# %% ####################################################################################
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import os
import urllib.request
import gzip
from tqdm import tqdm
import gzip
import shutil
from scipy.stats import spearmanr
import warnings


# %% ####################################################################################
def simulate_data(
    scRNA_df : pd.DataFrame,
    n_mixtures : int,
    n_cells_in_mix : int,
    n_genes : int = None,
    seed : int = 1,
    coverage_threshold : int = 5):

    """
    Function that generates artificial bulk mixtures Y from a collection of single cell profiles and saves the ground truth of their composition C.
    Furthermore, it derives a reference matrix X by averaging over provided example profiles per cell type.

    Parameters
    ----------
    scRNA_df : pd.DataFrame
        A matrix containing labeled single cell RNA profiles from multiple cell types.
        Please provide multiple examples per cell type and label each sample with a column label stating the cell type.
        Shape: genes x n_samples
    n_mixtures : int
        How many artificial bulks should be created.
    n_cells_in_mix : int
        How many profiles from scRNA_df should be randomly mixed together in one artifical bulk.
    n_genes : int
        Filters genes from scRNA_df by variance across celltypes and restricts the gene space to the n_genes most variable genes.
    seed : int
        Random seed for the drawing process.
    coverage_threshold : int
        Minimum number of examples per cell type, if number of examples is below the threshold a warning will be thrown. (Default = 5).

    Returns
    -------
    X_ref : pd.DataFrame
        Single cell reference matrix X calculated by averaging over all examples of each cell type in scRNA_df.
        Shape: genes x celltypes
    Y_mat : pd.DataFrame
        Y matrix containing the generated artificial bulk profiles. Each column represents one artificial bulk, normalized by n_cells_in_mix
        Shape: genes x n_mixtures
    C_mat : pd.DataFrame
        Composition matrix containing the true cellular abundance in percent for each artificial mixture according to the drawing process.
        Shape: cell types x n_mixtures
    """

    # Input Handling
    # Check if n_genes is plausible
    if n_genes:
        if scRNA_df.shape[0] < n_genes:
            raise ValueError(f"Number of genes to select needs to be equal or lower than the amount of genes provided in scRNA_df. "
                             f"Found {scRNA_df.shape[0]} genes in scRNA_df but n_genes is {n_genes}")
    
    # Coverage check
    ct_unique = set(scRNA_df.columns)
    n_celltypes = len(ct_unique)
    n_examples = len(scRNA_df.columns)
    
    # Issue warnings for low coverage and check for single-example cell types
    celltype_counts = scRNA_df.columns.value_counts()
    for celltype, count in celltype_counts.items():
        if count == 1:
            warnings.warn(f"Warning: Only one example provided for cell type '{celltype}'. The mean will be the same as the single example.")
        elif count <= coverage_threshold:  # Assuming coverage_threshold is defined
            warnings.warn(f"Warning: Low coverage for cell type '{celltype}' ({count} examples).")

    # Gene filter function
    def gene_filter(scRNA_df, n_genes=1000):
        # Calculate average expression per cell type
        X_ref = scRNA_df.T.groupby(level=0).mean().T

        # Calculate variance across cell types for each gene
        gene_variances = X_ref.var(axis=1)

        # Select the top n_genes based on variance
        selected_genes = gene_variances.nlargest(n_genes).index

        return selected_genes

    # Filter genes by variance across cell types if n_genes is specified
    if n_genes:
        gene_selection = gene_filter(scRNA_df, n_genes)
        scRNA_df = scRNA_df.loc[gene_selection, :]

    
    # Creating X_ref as average profiles of the unique cell types
    X_ref = scRNA_df.T.groupby(level=0).mean().T

    # Generate random samples all at once via random index choice
    np.random.seed(seed)
    bulk_indices = np.random.choice(scRNA_df.columns.size, size=(n_mixtures, n_cells_in_mix), replace=True)

    # Create Y_mat and C_mat without a for loop
    Y_mat = pd.DataFrame(
        np.add.reduceat(scRNA_df.values[:, bulk_indices.flatten()], np.arange(0, n_cells_in_mix * n_mixtures, n_cells_in_mix), axis=1) / n_cells_in_mix,
        index=scRNA_df.index
    )

    # Create C_mat by counting occurrences of each cell type in the sampled indices
    C_mat = pd.DataFrame(0, index=scRNA_df.columns.unique(), columns=range(n_mixtures))

    for i in range(n_mixtures):
        sampled_celltypes = scRNA_df.columns[bulk_indices[i, :]]
        celltype_counts = sampled_celltypes.value_counts()
        C_mat.loc[celltype_counts.index, i] = celltype_counts.values

    C_mat = C_mat.div(n_cells_in_mix)
    C_mat = C_mat.reindex(X_ref.columns)


    return X_ref, Y_mat, C_mat


# %% ####################################################################################
def calculate_estimated_composition(
        X_mat: pd.DataFrame,
        Y_mat: pd.DataFrame,
        gamma: pd.DataFrame):
    r"""
    Function that calculates the estimated composition of Y based on a
    reference matrix X and the gene weight vector gamma generated by the DTD training.

    Parameters
    ----------
    X_mat : pd.DataFrame
        Single cell reference matrix X containing (average) profiles as columns per celltype.
        Shape: genes x celltypes
    Y_mat : pd.DataFrame
        Y matrix containing the generated artificial bulk profiles.
        Shape: genes x n_mixtures
    gamma : pd.DataFrame
        DataFrame containing the weights for the genes featured in X_mat and Y_mat.
        Usually generated by Loss-function Learning for Digital Tissue Deconvolution.
        Shape: genes x 1

    Returns
    -------
    C_estimated : pd.DataFrame
        Matrix that contains estimated cellular abundances for the bulk profiles of the Y matrix.

    """

    # Input Handling

    # Check if the number of genes (rows) matches between X_mat and Y_mat
    if X_mat.shape[0] != Y_mat.shape[0]:
        raise ValueError(f"Number of genes (rows) must match between X_mat and Y_mat. "
                         f"Found {X_mat.shape[0]} in X_mat and {Y_mat.shape[0]} in Y_mat.")

    # Check if the number of genes (rows) matches between X_mat and gamma
    if X_mat.shape[0] != gamma.shape[0]:
        raise ValueError(f"Number of genes (rows) must match between X_mat and gamma. "
                         f"Found {X_mat.shape[0]} in X_mat and {gamma.shape[0]} in gamma.")

    # Check if the gene labels (index) are consistent between X_mat and Y_mat
    if not X_mat.index.equals(Y_mat.index):
        raise ValueError("Gene labels (index) must be identical between X_mat and Y_mat.")

    # Check if the gene labels (index) are consistent between X_mat and gamma
    if not X_mat.index.equals(gamma.index):
        raise ValueError("Gene labels (index) must be identical between X_mat and gamma.")

    # Calculations

    X = torch.from_numpy(X_mat.values.astype(np.double))
    Y = torch.from_numpy(Y_mat.values.astype(np.double))
    gamma = torch.from_numpy(gamma.values.flatten().astype(np.double))
    Gamma = torch.diag(gamma)
    C_e = torch.linalg.inv(X.T @ Gamma @ X) @ X.T @ Gamma @ Y
    C_e[C_e < 0] = 0
    C_estimated = pd.DataFrame(C_e, index=X_mat.columns, columns=Y_mat.columns)
    return (C_estimated)


# %% ####################################################################################
def XYCtoTorch(X_mat, Y_mat, C_mat):
    X_torch = torch.Tensor(X_mat.to_numpy())
    Y_torch = torch.Tensor(Y_mat.to_numpy())
    C_torch = torch.Tensor(C_mat.to_numpy())
    return X_torch, Y_torch, C_torch


# %% ####################################################################################
def load_example():
    r"""
    Function providing an example test and training set for ADTD.
    Data is downloaded from NCBI (Tirosh et al., https://doi.org/10.1126/science.aad0501) once and cached for further usage.

    Returns
    -------
    tir_train : pd.DataFrame
        Matrix that contains multiple single cell RNA profiles for different cell types from a set of tumors.
        Column labels equals cell type. Shape: genes x n_samples
    tir_test : pd.DataFrame
        Matrix that contains multiple single cell RNA profiles for different cell types from a different set of tumors.
        Column labels equals cell type. Shape: genes x n_samples
    """

    # Internal functions:

    def download_and_extract_gz(url, extract_path):
        # creating data folder
        if not os.path.exists("example/data"):
            os.makedirs("example/data")
        # Download the file if it doesn't exist
        if not os.path.exists(extract_path):
            print(f"Downloading Tirosh data from {url}...")
            with urllib.request.urlopen(url) as response, open("example/data/downloaded_file.gz", 'wb') as out_file:
                content_length = int(response.headers.get('Content-Length'))
                with tqdm(total=content_length, unit='B', unit_scale=True) as pbar:
                    while True:
                        data = response.read(1024)
                        if not data:
                            break
                        out_file.write(data)
                        pbar.update(len(data))

            # Extract the file
            print("Extracting file...")
            with gzip.open("example/data/downloaded_file.gz", "rb") as f_in, open(extract_path, "wb") as f_out:
                f_out.write(f_in.read())

            # Remove the downloaded file
            os.remove("example/data/downloaded_file.gz")

            print(f"File extracted to {extract_path}")

        else:
            print("Raw data already downloaded")

    def importing_tirosh():
        # reading data
        # if os.path.exists("example/data/train.feather"):
        #    print("Loading already existing preprocessed data...")
        #    tir_train = pd.read_feather("example/data/train.feather")
        #    tir_test = pd.read_feather("example/data/test.feather")
        #    return tir_train, tir_test

        # else:

        tirosh = pd.read_csv("example/data/tirosh_raw.txt", sep="\t", index_col=0)
        # Drop weird gene 'MARCH1' which is featured TWICE in the dataset?!
        tirosh = tirosh.drop('MARCH1')
        # list of cell types to filter
        cell_types = [0, 1, 2, 3, 4, 5, 6]
        celltype_dict = {0: 'Malignant', 1: 'T', 2: 'B', 3: 'Macro', 4: 'Endo', 5: 'CAF', 6: 'NK'}

        # Transformation to tpm
        # Formula: 10*(2^Ei,j-1)
        def backtransformation(x):
            return 10 * (2 ** x - 1)

        tirosh[3:].map(backtransformation)

        # Restricting gene space (OLD)
        # temp_var = tirosh.iloc[:,:].var(axis = 1)
        # temp_var[0:3] = None
        # tirosh['Variance'] = temp_var
        # tirosh.sort_values('Variance', ascending = False, na_position = 'first', inplace = True)
        # tirosh.drop('Variance', axis = 1, inplace = True)
        # tirosh = tirosh[0:1003]

        ##### Restricting gene space (NEW) ####
        # Step 1: Create Reference Matrix from whole dataset
        scRNA_df = tirosh.iloc[3:, :]
        scRNA_df.index = tirosh.index[3:]
        scRNA_df.columns = tirosh.iloc[2, :]
        # display(scRNA_df)
        ct_unique = set(scRNA_df.columns)
        X_ref = pd.DataFrame()
        for celltype in ct_unique:
            curr_avg = scRNA_df.loc[:, celltype].mean(axis=1)
            X_ref[celltype] = curr_avg

        # Step 2: Calculate Variance over average expression levels of genes between different celltypes
        temp_var = X_ref.iloc[:, 1:].var(axis=1)  # Ignore malignant cells here?!
        X_ref['Variance'] = temp_var
        X_ref.sort_values('Variance', ascending=False, na_position='first', inplace=True)
        # display(X_ref)
        genes_top1k = list(X_ref.index[0:1000])
        # print(len(genes_top1k))

        # Step 3: Choose the calculated top 1k genes in the full dataset
        head_index = list(tirosh.index[0:3])
        restricted_selection = head_index + genes_top1k
        tirosh = tirosh.loc[restricted_selection]
        # display(tirosh)

        #### Splitting Data by Tumor Ids #####
        tumor_ids = np.unique(tirosh.iloc[0].values)
        i_split = int(len(tumor_ids) / 2)
        train_ids = tumor_ids[:i_split]
        test_ids = tumor_ids[i_split:]

        tir_train = tirosh.loc[:, tirosh.iloc[0].isin(train_ids)]
        tir_test = tirosh.loc[:, tirosh.iloc[0].isin(test_ids)]

        def choose_celltypes(tir_raw, cell_types):
            # Chooses celltypes and drops non gene expression rows
            tir_filtered = tir_raw.loc[:, tir_raw.iloc[2].isin(cell_types)]
            tir_filtered = tir_filtered.iloc[2:]
            tir_filtered.columns = tir_filtered.iloc[0]
            tir_filtered = tir_filtered[1:]
            tir_filtered = tir_filtered.rename(columns=celltype_dict)
            tir_filtered.index.name = "Genes"
            tir_filtered.columns.name = "Celltypes"
            return tir_filtered

        # filtering the sets
        tir_train = choose_celltypes(tir_train, cell_types)
        tir_test = choose_celltypes(tir_test, cell_types)

        return tir_train, tir_test

    # Actual Import
    # Tirosh URL
    url = 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE72056' \
          '&format=file&file=GSE72056%5Fmelanoma%5Fsingle%5Fcell%5Frevised%5Fv2%2Etxt%2Egz'
    extract_path = os.path.join(os.getcwd(), "example/data/tirosh_raw.txt")
    download_and_extract_gz(url, extract_path)
    print("Importing Data to Python ...")

    if os.path.exists("example/data/train.pkl"):
        # Found preprocessed cached data, skip preprocessing and load from csv instead:
        print("Using cached, preprocessed data")
        tir_train = pd.read_pickle("example/data/train.pkl")
        tir_test = pd.read_pickle("example/data/test.pkl")
    else:
        tirosh = pd.read_csv("example/data/tirosh_raw.txt", sep="\t", index_col=0)
        print("Splitting and filtering data ...")
        tir_train, tir_test = importing_tirosh()
        # Drop malignant samples
        tir_train = tir_train.drop("Malignant", axis=1)
        tir_test = tir_test.drop("Malignant", axis=1)
        # save the preprocessed data
        tir_train.to_pickle("example/data/train.pkl")
        tir_test.to_pickle("example/data/test.pkl")
    print("Done")
    # drop last row (1001 instead of 1000 genes...)
    # tir_train.drop(tir_train.tail(1).index,inplace=True)
    # tir_test.drop(tir_test.tail(1).index,inplace=True)
    return tir_train, tir_test


def plot_corr(C_true, C_est, title="", color="grey", hidden_ct=None, c_est=None, path=None):
    """
    Function that visualizes the correlation of estimated composition for multiple bulks with the ground truth.

    Parameters
    ----------
    C_true : pd.DataFrame
        True cellular compositions for all mixtures analyzed in C_est.
        Features all cell types, even if one cell type is hidden.
        Shape: n_cell_types x n_mixtures
    C_est : pd.DataFrame
        Estimated cellular compositions for multiple bulk mixtures.
        Features all non-hidden cell types from the ground truth.
        Shape: b_cell_types x n_mixtures
    color : String
        Sets the color of the correlation plots for the non-hidden cell types.
    hidden_ct : String (Optional)
        Needs to be set to the label of the cell type in C_true that is hidden, which means its not featured in C_est.
    c_est : pd.DataFrame (Optional)
        Estimated hidden background contributions for all mixtures featured in C_est (ADTD).
        Shape: 1 x n_mixtures
    path : String (Optional)
        Specify a path for saving the figure as pdf.

    Returns
    -------
    correlations : pd.DataFrame
        Spearman correlation values for C_true vs. C_est for each cell type.
        Shape: n_cell_types
    """

    # Inititialize pd.Series for correlation values
    correlations = pd.Series()

    # Calculate bounds & plots
    rows = int(np.ceil(len(C_true.index) / 3))
    empty = rows * 3 - len(C_true.index)
    i = 0
    j = 0
    fig, axes = plt.subplots(rows, 3, figsize=(15, rows * 5), squeeze=False)
    fig.suptitle(title)
    for celltype in C_est.index:
        r_val, _ = spearmanr(C_true.loc[celltype], C_est.loc[celltype])
        correlations[celltype] = r_val
        axes[j, i].set_title(celltype + " (R: %1.2f)" % (r_val))
        axes[j, i].scatter(C_true.loc[celltype], C_est.loc[celltype], alpha=0.6, c=color)
        axes[j, i].set_xlabel("true abundance")
        axes[j, i].set_ylabel("estimated abundance")
        if i == 2:
            j += 1
            i = 0
        else:
            i += 1
    if hidden_ct:
        r_val, _ = spearmanr(C_true.loc[hidden_ct], c_est.loc["hidden"])
        correlations['hidden'] = r_val
        axes[j, i].set_title("Hidden (R: %1.2f)" % (r_val))
        axes[j, i].scatter(C_true.loc[hidden_ct], c_est.loc["hidden"], alpha=0.6, c="black")
        axes[j, i].set_xlabel("true abundance")
        axes[j, i].set_ylabel("estimated abundance")

    # remove empty subplots
    if empty == 2:
        axes.flat[-1].set_visible(False)
        axes.flat[-2].set_visible(False)
    elif empty == 1:
        axes.flat[-1].set_visible(False)

    if path:
        plt.savefig(path + '.pdf')

    plt.show()

    return correlations
##########################################################
#
# Various utility functions for DTD/ADTD pipelines
#
##########################################################




# %% ####################################################################################
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, linregress, pearsonr
from scipy.io import mmread
import numpy as np
import pandas as pd

# %% ####################################################################################
def process_composition(C, subtypes_dict, main_type_to_ignore):
    new_rows = []
    
    # Iterate over each main cell type in the dictionary
    for cell_type, subtypes in subtypes_dict.items():
        if cell_type == main_type_to_ignore:
            # If the current cell type is the one we want to keep the subtypes for
            for subtype in subtypes:
                if subtype in C.index:
                    # Add rows corresponding to subtypes directly
                    new_rows.append(C.loc[subtype])
        else:
            # Sum up all the subtypes to get the main type
            main_type_sum = C.loc[subtypes].sum()
            main_type_sum.name = cell_type
            new_rows.append(main_type_sum)
    
    # Convert the list of Series objects back into a DataFrame
    process_C = pd.DataFrame(new_rows)
    
    return process_C


# %% ####################################################################################
def create_reference_matrices(scRNA_df, subtypes_dict, selected_main_type):

    X_ref = pd.DataFrame()
    for celltype in subtypes_dict.keys():
        curr_avg = scRNA_df.loc[:, subtypes_dict[celltype]].mean(axis=1)
        X_ref[celltype] = curr_avg
    
    X_ref_with_subtypes = pd.DataFrame()
    
    for celltype in subtypes_dict.keys():
        if celltype == selected_main_type:
            for subtype in subtypes_dict[selected_main_type]:
                curr_avg = scRNA_df.loc[:, subtype].mean(axis=1)
                X_ref_with_subtypes[subtype] = curr_avg
        else:
            curr_avg = scRNA_df.loc[:, subtypes_dict[celltype]].mean(axis=1)
            X_ref_with_subtypes[celltype] = curr_avg
    
    X_subtypes = pd.DataFrame()
    
    for subtype in subtypes_dict[selected_main_type]:
        curr_avg = scRNA_df.loc[:, subtype].mean(axis=1)
        X_subtypes[subtype] = curr_avg

    return X_ref, X_ref_with_subtypes, X_subtypes


# %% ####################################################################################
def create_complete_reference(scRNA_df):

    X_ref = pd.DataFrame()

    celltypes = scRNA_df.columns.unique()
    for celltype in celltypes:
        curr_avg = scRNA_df.loc[:, celltype].mean(axis=1)
        X_ref[celltype] = curr_avg
    
    return X_ref


# %% ####################################################################################
def estimate_corr(C_true, C_est, title="", color="grey", fPlot=False, hidden_ct=None, c_est=None, savePath=None, nan_policy='omit'):

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
        #r_val, _ = spearmanr(C_true.loc[celltype], C_est.loc[celltype], nan_policy=nan_policy)
        r_val, _ = pearsonr(C_true.loc[celltype], C_est.loc[celltype])
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
        #r_val, _ = spearmanr(C_true.loc[hidden_ct], c_est.loc["hidden"])
        r_val, _ = pearsonr(C_true.loc[hidden_ct], c_est.loc["hidden"])
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

    if savePath:
        plt.savefig(savePath + '.pdf')

    if fPlot:
        plt.show()
    else:
        plt.close()

    return correlations.fillna(0)


# %% ####################################################################################
def tirosh_create_subtype_labelled_set(sc_data, metadata, column_name_main, column_name_sub, types_to_split, verbose=False):

    if isinstance(types_to_split, str): types_to_split=[types_to_split]

    maintype_names = metadata[column_name_main].unique()
    subtype_names = metadata[metadata[column_name_main].isin(types_to_split)][column_name_sub].unique()

    # Create a dictionary mapping column names to major/minor cell types
    meta_dict = metadata.set_index(metadata.columns[0])[[column_name_main, column_name_sub]].to_dict(orient="index")
    
     # Create a dictionary for renaming columns
    cell_celltype_dic = {}
    for column in sc_data.columns:
          
        cell_info = meta_dict.get(column, {})
        celltype_major = cell_info.get(column_name_main)
        celltype_minor = cell_info.get(column_name_sub)

        # Celltype can either be minor or major celltype
        if celltype_minor in subtype_names:
            cell_celltype_dic[column] = celltype_minor
        elif celltype_major in maintype_names:
            cell_celltype_dic[column] = celltype_major
        else:
            if verbose: print(f"Warning!: {celltype_major} -> not found in XRef")

    # Rename columns using the constructed dictionary
    sc_data_filtered = sc_data.rename(columns=cell_celltype_dic)
    sc_data_filtered.index.name = "Genes"
    sc_data_filtered.columns.name = "Celltypes"

    return sc_data_filtered.drop(columns=[col for col in sc_data_filtered if col not in np.concatenate((maintype_names, subtype_names))])



# %% ####################################################################################
def tirosh_split_train_val(metadata, split, column_name_subtype, column_name_patient_id, verbose=False, random_state=42):
    # All subtypes 
    subtypes = metadata[column_name_subtype].unique()
    all_patients = metadata[column_name_patient_id].unique()

    train_patients = []
    test_patients = []

    for subtype in subtypes:
        num_patients = len(metadata[metadata[column_name_subtype] == subtype][column_name_patient_id].unique())
        if verbose: print(f"{subtype}:{num_patients}")

        patients = metadata[metadata[column_name_subtype] == subtype].sample(frac=1, random_state=random_state)
        num_train = int(split * num_patients)
        train_patients.extend(patients[column_name_patient_id].unique()[:num_train])
        test_patients.extend(patients[column_name_patient_id].unique()[num_train:])

    return train_patients, test_patients

# Usage:
#train_patId, test_patId = split_train_val(metadata_cancer, train_val_split, "subtype", "orig.ident", verbose=False, random_state=random_state)


# %% ####################################################################################
def restrict_to_n_genes(data_train, data_val, n_genes):

    gene_variances = data_train.var(axis=1)
    top_n_genes = gene_variances.nlargest(n_genes).index
    data_train = data_train.loc[top_n_genes]
    data_val = data_val.loc[top_n_genes]

    return {
        'data_train' : data_train,
        'data_val' : data_val
    }



# %% ####################################################################################
def log(txt, log_file=None):

    if log_file is None:
        print(txt)
    else:
        with open(log_file, "a+") as f:
            print(txt)
            f.write("\n")
            f.write(txt)


# %% ####################################################################################
def simulate_data(
        scRNA_df: pd.DataFrame,
        n_mixtures: int,
        n_cells_in_mix: int,
        n_genes: int = None,
        seed: int = 1):
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

    Returns
    -------
    X_ref : pd.DataFrame
        Single cell reference matrix X calculated by averaging over all examples of each cell type in scRNA_df.
        Shape: genes x celltypes
    Y_mat : pd.DataFrame
        Y matrix containing the generated artificial bulk profiles. Each column represents one artificial bulk.
        Shape: genes x n_mixtures
    C_mat : pd.DataFrame
        Composition matrix containing the true cellular abundance for each artificial mixture according to the drawing process.
        Shape: cell types x n_mixtures
    """

    # Input Handling

    # Check if n_genes is plausible
    if n_genes:
        if scRNA_df.shape[0] < n_genes:
            raise ValueError(
                f"Number of genes to select needs to be equal or lower then the amount of genes provided in scRNA_df. "
                f"Found {scRNA_df.shape[0]} genes in scRNA_df but n_genes is {n_genes}")

    # coverage check
    ct_unique = set(scRNA_df.columns)
    n_celltypes = len(ct_unique)
    n_examples = len(scRNA_df.columns)
    if (n_examples / n_celltypes <= 5):
        print("Warning: Low Average Coverage (Few Examples per Celltype)")

    # gene filter help function
    def gene_filter(scRNA_df, n_genes=1000):

        # Calculate average expression per cell type
        ct_unique = set(scRNA_df.columns)
        X_ref = pd.DataFrame()
        for celltype in ct_unique:
            if type(scRNA_df.loc[:, celltype]) is not pd.Series:
                curr_avg = scRNA_df.loc[:, celltype].mean(axis=1)
            else:
                curr_avg = scRNA_df.loc[:, celltype]
            X_ref[celltype] = curr_avg
        X_ref.index = scRNA_df.index

        # Calculate Variance over average expression levels of genes between different celltypes
        temp_var = X_ref.var(axis=1)
        X_ref['Variance'] = temp_var
        X_ref.sort_values('Variance', ascending=False, na_position='first', inplace=True)

        # Return sorted index , cut to n_genes
        return list(X_ref.index)[:n_genes]

    # Filtering genes by variance across cell types to select n_genes, if wanted
    if n_genes:
        gene_selection = gene_filter(scRNA_df, n_genes)
        scRNA_df = scRNA_df.loc[gene_selection, :]

    # Creating X_ref as average profiles of the unique celltypes and counts per celltype as dictionary
    X_ref = pd.DataFrame()
    celltype_counts = {}
    for celltype in ct_unique:
        if type(scRNA_df.loc[:, celltype]) is not pd.Series:
            curr_avg = scRNA_df.loc[:, celltype].mean(axis=1)
            celltype_counts[celltype] = len(scRNA_df.loc[:,celltype].columns)
        else:
            curr_avg = scRNA_df.loc[:, celltype]
            celltype_counts[celltype] = 1
        X_ref[celltype] = curr_avg

    # Preparing Dataframes
    Y_mat = pd.DataFrame(columns=range(0, n_mixtures),
                         index=X_ref.index)
    C_mat = pd.DataFrame(columns=range(0, n_mixtures),
                         index=X_ref.columns)

    # Creating artificial mixtures
    for i in range(0, n_mixtures):
        # Sample random reference profiles
        bulk = scRNA_df.sample(n_cells_in_mix, axis=1, replace=True,
                               random_state=seed + i)
        # Calculate corresponding C vector
        C_vec = pd.Series(data=np.zeros(len(ct_unique)), index=ct_unique)
        for celltype in ct_unique:
            for used_celltype in bulk.columns:
                if used_celltype == celltype:
                    C_vec.loc[celltype] += 1
        # Sum up the sample into a bulk profile
        Y_vec = bulk.sum(axis=1) / n_cells_in_mix
        # Add the results to the dataframes
        Y_mat[i] = Y_vec
        C_mat[i] = C_vec / n_cells_in_mix

    return X_ref, Y_mat, C_mat, celltype_counts



# %% ####################################################################################
def filter_subtypes_by_dataframe_columns(cell_dict, X_ref):

    filtered_dict = {}
    
    for main_cell, subtypes in cell_dict.items():
        valid_subtypes = [subtype for subtype in subtypes if subtype in X_ref.columns]
        
        if valid_subtypes:
            filtered_dict[main_cell] = valid_subtypes
    
    return filtered_dict


# %% ####################################################################################
def merge_celltypes(sub_celltypes, subset_celltypes):
    merged_structure = {}

    for main_cell, subtypes in sub_celltypes.items():
        merged_structure[main_cell] = {}

        for subtype in subtypes:
            if subtype in subset_celltypes:
                merged_structure[main_cell][subtype] = subset_celltypes[subtype]
            else:
                merged_structure[main_cell][subtype] = []

    return merged_structure


# %% ####################################################################################
def create_reverse_mapping(merged_celltypes):
    reverse_mapping = {}
    
    # Iterate over the main cell types and their nested subtypes
    for _, subtypes in merged_celltypes.items():
        for subtype, subset_subtypes in subtypes.items():
            # Map the subset subtypes to the mid-level (subtype) cell
            for subset_subtype in subset_subtypes:
                reverse_mapping[subset_subtype] = subtype

    return reverse_mapping


# %% ####################################################################################
def flatten_nested_dict(merged_celltypes):
    flattened_dict = {}

    for main_cell, subtypes in merged_celltypes.items():
        all_subtypes = []
        
        for mid_level_subtype, subset_subtypes in subtypes.items():
            all_subtypes.append(mid_level_subtype)
        
        flattened_dict[main_cell] = all_subtypes

    return flattened_dict


# %% ####################################################################################
def split_train_test_half(sc_data, random_state=42):
    genes = sc_data.index

    data_train = pd.DataFrame(index=genes)
    data_val = pd.DataFrame(index=genes)
    celltypes = sc_data.columns.unique()

    for celltype in celltypes:
        if type(sc_data[celltype]) is not pd.Series:
            idx_half = int(sc_data[celltype].shape[1] / 2)
        else:
            # Catch case where only one cell of a specific type is included and do not include this in
            # the train/test set, as this makes obviously no sense
            continue

        celltype_shuffled = sc_data[celltype].sample(frac=1, random_state=random_state)

        add_to_train = celltype_shuffled.iloc[:,0:idx_half]
        add_to_test = celltype_shuffled.iloc[:,idx_half:]

        data_train = pd.concat([data_train, add_to_train], axis=1)
        data_val = pd.concat([data_val, add_to_test], axis=1)

    return data_train, data_val

# %% ####################################################################################
def split_train_test_half_exact_half(sc_data):
    genes = sc_data.index

    data_train = pd.DataFrame(index=genes)
    data_val = pd.DataFrame(index=genes)
    celltypes = sc_data.columns.unique()

    for celltype in celltypes:
        if type(sc_data[celltype]) is not pd.Series:
            idx_half = int(sc_data[celltype].shape[1] / 2)
        else:
            # Catch case where only one cell of a specific type is included and do not include this in
            # the train/test set, as this makes obviously no sense
            continue
        
        last_idx = sc_data[celltype].shape[1]
        indices = np.arange(last_idx) 
        even_indices = indices[indices % 2 == 0]
        odd_indices = indices[indices % 2 != 0]

        #celltype_shuffled = sc_data[celltype].sample(frac=1)

        add_to_train = sc_data[celltype].iloc[:,even_indices]
        add_to_test = sc_data[celltype].iloc[:,odd_indices]

        data_train = pd.concat([data_train, add_to_train], axis=1)
        data_val = pd.concat([data_val, add_to_test], axis=1)

    return data_train, data_val

# %% ####################################################################################
def linReg(C_true, C_est):
    # Get celltypes
    celltypes = C_true.index.unique()

    linReg_results = pd.DataFrame(index=celltypes, columns=['slope', 'intercept', 'p'])

    for celltype in celltypes:
        slope, intercept, _, p, _ = linregress(C_est.loc[celltype,:], C_true.loc[celltype,:])

        linReg_results.loc[celltype, 'slope'] = slope
        linReg_results.loc[celltype, 'intercept'] = intercept
        linReg_results.loc[celltype, 'p'] = p

    return linReg_results

# %% ####################################################################################
def adjustToLinReg(C_est, linReg_results):
    # Get celltypes
    celltypes = linReg_results.index.unique()

    for celltype in celltypes:
        # y = mx + t
        C_est.loc[celltype] = linReg_results.loc[celltype, 'slope'] * C_est.loc[celltype] + linReg_results.loc[celltype, 'intercept']

    return C_est
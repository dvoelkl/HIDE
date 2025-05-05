##########################################################
#
# Various functions loading data from different sources
# and processing it such that they can be used
# with DTD/ADTD pipelines
#
##########################################################

# %% ####################################################################################
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.io import mmread
import numpy as np
import pandas as pd
import os


# %% ####################################################################################
def disco_read_data_mtx(mtx_path, barcodes_path, genes_path):

    # Read barcodes
    barcodes = pd.read_csv(barcodes_path, header=None)

    # Read genes
    genes = pd.read_csv(genes_path, header=None)
    
    # Read matrix
    data_mtx = mmread(mtx_path)
    data_df = pd.DataFrame.sparse.from_spmatrix(data_mtx)#, index=genes.iloc[:,0], columns=barcodes.iloc[:,0])
    data_df = data_df.sparse.to_dense()

    data_df = data_df.set_index(genes.iloc[:,0])
    data_df.columns = barcodes.iloc[:,0]

    return data_df


# %% ####################################################################################
def disco_read_metadata(metadata_path, major_column_name, minor_column_name):
    metadata = pd.read_csv(metadata_path)

    main_celltypes = metadata[major_column_name].unique()

    sub_celltypes = {}
    for main_celltype in main_celltypes:
        sub_celltypes[main_celltype] = []
        subtypes = metadata[metadata[major_column_name] == main_celltype][minor_column_name].unique()
        sub_celltypes[main_celltype].extend(subtypes)
    
    return {
        'metadata' : metadata,
        'main_celltypes' : main_celltypes,
        'sub_celltypes' : sub_celltypes
    }


# %% ####################################################################################
def disco_read_split_sampleId(mtx_path, barcodes_path, genes_path, metadata_path, seed=941, train_test_split=0.7):

    # Read barcodes
    barcodes = pd.read_csv(barcodes_path, header=None)

    # Read genes
    genes = pd.read_csv(genes_path, header=None)

    # Read metadata
    metadata = pd.read_csv(metadata_path)
    
    sampleIds = pd.Series(metadata['sample_id'].unique())
    
    # Read matrix
    data_mtx = mmread(mtx_path)
    data_df = pd.DataFrame.sparse.from_spmatrix(data_mtx)#, index=genes.iloc[:,0], columns=barcodes.iloc[:,0])
    data_df = data_df.sparse.to_dense()

    data_df = data_df.set_index(genes.iloc[:,0])
    data_df.columns = barcodes.iloc[:,0]
    data_df.loc['sample_id'] = metadata['sample_id'].values

    # Select sample_ids used for training
    sampleIds_shuffeled = sampleIds.sample(n=len(sampleIds), replace=False, random_state=seed)
    train_Ids = sampleIds_shuffeled[:int(train_test_split*len(sampleIds))]
    test_Ids = sampleIds_shuffeled[int(train_test_split*len(sampleIds)):]

    data_df_T = data_df.T
    train_df = data_df_T[data_df_T['sample_id'].isin(train_Ids)].T.drop('sample_id', axis=0).astype('float64')
    test_df = data_df_T[data_df_T['sample_id'].isin(test_Ids)].T.drop('sample_id', axis=0).astype('float64')

    return train_df, test_df



# %% ####################################################################################
def tirosh_load_data(path_to_csv):

    if os.path.exists(path_to_csv):
        data_df = pd.read_csv(path_to_csv)
        data_df = data_df.set_index(data_df.columns[0])
    else:
        raise FileNotFoundError(f"File {path_to_csv} does not exist!")
    
    return data_df


# %% ####################################################################################
def tirosh_read_metadata(path_to_csv, major_column_name, minor_column_name):

    if os.path.exists(path_to_csv):
        metadata = pd.read_csv(path_to_csv)
    else:
        raise FileNotFoundError(f"File {path_to_csv} does not exist!")
    
    main_celltypes = metadata[major_column_name].unique()

    sub_celltypes = {}
    for main_celltype in main_celltypes:
        sub_celltypes[main_celltype] = []
        subtypes = metadata[metadata[major_column_name] == main_celltype][minor_column_name].unique()
        sub_celltypes[main_celltype].extend(subtypes)

    return {
        'metadata' : metadata,
        'main_celltypes' : main_celltypes,
        'sub_celltypes' : sub_celltypes
    }
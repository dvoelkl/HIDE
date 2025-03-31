##########################################################
#
#
#
#
##########################################################

### Imports
import pandas as pd
import numpy as np
from hDTD import HIDE
from pipelines_dataloader import disco_read_metadata
from pipelines_utils import merge_celltypes, filter_subtypes_by_dataframe_columns

### Parameters
iterations_HIDE = 1000

path_to_data_folder = f"./data/"


### MAIN ###
def run(path_to_data_folder, iterations_HIDE, savePathCorrelation):
    print("########## HIDE ##########")

    # load metadata
    meta = disco_read_metadata(path_to_data_folder+'cell_hierarchy.csv', "celltype_major", 'celltype_minor')
    main_celltypes = meta['main_celltypes']
    sub_celltypes = meta['sub_celltypes']
    meta = disco_read_metadata(path_to_data_folder+'cell_hierarchy.csv', "celltype_minor", 'celltype_sub')
    subset_celltypes = meta['sub_celltypes']

    sub_celltypes = merge_celltypes(sub_celltypes, subset_celltypes)


    X_train = pd.read_csv(path_to_data_folder + "/X_train.csv", index_col=0)
    Y_train = pd.read_csv(path_to_data_folder+"/train_data.csv", index_col=0)
    
    C_train = pd.read_csv(path_to_data_folder+"/train_distribution.csv", index_col=0)

    # Calculate sum of each celltype
    celltype_counts_train = {}
    for celltype in X_train.columns.unique():
        celltype_counts_train[celltype] = C_train.sum(axis=1)[celltype]

    # Load the test set 
    print(f"--> Load precreated test set")
    Y_val = pd.read_csv(path_to_data_folder+f"/test_data.csv", index_col=0)
    C_val = pd.read_csv(path_to_data_folder+f"/test_distribution.csv", index_col=0)

    # Filter subtypes dictionary so that subtypes that are not in the training data do not appear
    for type in main_celltypes:
        sub_celltypes[type] |= filter_subtypes_by_dataframe_columns(sub_celltypes[type], X_train)

    # Run HIDE
    res_hide = HIDE(C_train, 
                        C_val, 
                        Y_train, 
                        Y_val, 
                        X_train, 
                        sub_celltypes, 
                        celltype_counts_train, 
                        iterations_dtd=iterations_HIDE,
                        savePath=savePathCorrelation, 
                        saveC=True)


if __name__ == '__main__':

    run(path_to_data_folder, iterations_HIDE, savePathCorrelation='./results/')
    
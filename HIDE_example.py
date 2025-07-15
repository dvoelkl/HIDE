##########################################################
#
#
#
#
##########################################################

### Imports
import pandas as pd
import numpy as np
from HIDE_class import HIDEModel
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

    # Create and train HIDE model
    hide_model = HIDEModel(
        subtypes_dict=sub_celltypes,
        count_celltypes=celltype_counts_train,
        iterations_dtd=iterations_HIDE,
        save_path=savePathCorrelation,
        save_compositions=True,
        verbose=True
    )
    
    # Train the model
    hide_model.train(C_train, C_val, Y_train, Y_val, X_train)
    
    # Get training summary
    training_summary = hide_model.get_training_summary()
    print(f"Training completed with correlation: {training_summary['training_correlation']:.4f}")
    print(f"Validation correlation: {training_summary['validation_correlation']:.4f}")
    
    return hide_model


if __name__ == '__main__':
    # Train the model
    trained_model = run(path_to_data_folder, iterations_HIDE, savePathCorrelation='./results/')
    
    # Optional: Example of making predictions on new data
    # In this case, we'll predict on the validation set as an example
    print("\n########## Making Predictions ##########")
    Y_val = pd.read_csv(path_to_data_folder + "/test_data.csv", index_col=0)
    
    # Make predictions
    predictions = trained_model.predict(Y_val)
    
    print(f"Predictions completed:")
    print(f"  - Major cell types: {list(predictions['major'].index)}")
    print(f"  - Minor cell types: {list(predictions['minor'].keys())}")
    print(f"  - Sub cell types: {list(predictions['sub'].keys())}")
    
    # Save the trained model for later use
    trained_model.save_model('./results/trained_hide_model.pkl')
    print("\nTrained model saved to './results/trained_hide_model.pkl'")
    
##########################################################
#
# HIDE Example
#
##########################################################

### Imports
import pandas as pd
import numpy as np
import os
from HIDE_class import HIDEModel

### Parameters
iterations_HIDE = 10

path_to_data_folder = f"./data/"


### MAIN ###
def run(path_to_data_folder, iterations_HIDE):
    print("########## HIDE ##########")

    # Load data files
    X_train = pd.read_csv(path_to_data_folder + "/X_train.csv", index_col=0)
    Y_train = pd.read_csv(path_to_data_folder+"/train_data.csv", index_col=0)
    C_train = pd.read_csv(path_to_data_folder+"/train_distribution.csv", index_col=0)

    # Load the test set 
    print(f"--> Load precreated test set")
    Y_val = pd.read_csv(path_to_data_folder+f"/test_data.csv", index_col=0)
    C_val = pd.read_csv(path_to_data_folder+f"/test_distribution.csv", index_col=0)

    # Create HIDE model using the simplified initialization
    print(f"--> Creating HIDE model from hierarchy file")
    hide_model = HIDEModel.from_hierarchy_file(
        hierarchy_file_path=path_to_data_folder + 'cell_hierarchy.csv',
        X_ref=X_train,
        iterations_dtd=iterations_HIDE,
        verbose=True
    )
    
    # Calculate actual cell type counts from training data and update the model
    print(f"--> Calculating actual cell type counts from training data")
    celltype_counts_train = {}
    for celltype in X_train.columns.unique():
        celltype_counts_train[celltype] = C_train.sum(axis=1)[celltype]
    
    hide_model.update_cell_counts(celltype_counts_train)
    
    # Train the model
    hide_model.train(C_train, C_val, Y_train, Y_val, X_train)
    
    # Get training summary
    training_summary = hide_model.get_training_summary()
    print(f"Training completed with correlation: {training_summary['training_correlation']:.4f}")
    print(f"Validation correlation: {training_summary['validation_correlation']:.4f}")
    
    return hide_model


if __name__ == '__main__':
    # Train the model
    trained_model = run(path_to_data_folder, iterations_HIDE)
    
    # Optional: Example of making predictions on new data
    # In this case, we'll predict on the validation set as an example
    Y_val = pd.read_csv(path_to_data_folder + "/test_data.csv", index_col=0)
    
    # Make predictions
    predictions = trained_model.predict(Y_val)

    # Access the prediction
    C_major = predictions['major']
    C_minor = predictions['minor']
    C_est = predictions['sub']
    
    # Save the trained model for later use
    trained_model.save_model('./results/trained_hide_model.pkl')
    print("\nTrained model saved to './results/trained_hide_model.pkl'")
    
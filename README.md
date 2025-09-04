# HIDE
 
Implementation of HIDE as proposed in Völkl et al (2025), HIDE: Hierarchical cell-type Deconvolution, https://doi.org/10.1093/bioinformatics/btaf179
<br>
<br>
HIDE (Hierarchical cell-type Deconvolution) is a computational approach on infering cellular distribution from bulk transcriptomic data. In comparison to other algorithms HIDE incorporates a hierarchical cell-type structure. For a detailed description of its advantages, we refer to our paper.

## Quick Start

```python
from HIDE_class import HIDEModel
import pandas as pd

# Load your data
X_ref = pd.read_csv("X_train.csv", index_col=0)  # Reference expression matrix
C_train = pd.read_csv("train_distribution.csv", index_col=0)  # Training composition
C_val = pd.read_csv("test_distribution.csv", index_col=0)     # Validation composition  
Y_train = pd.read_csv("train_data.csv", index_col=0)         # Training bulk data
Y_val = pd.read_csv("test_data.csv", index_col=0)            # Validation bulk data

# Create model with simplified initialization
model = HIDEModel.from_hierarchy_file('cell_hierarchy.csv', X_ref)

# Optionally update with actual cell counts for better performance
cell_counts = {celltype: C_train.sum(axis=1)[celltype] for celltype in X_ref.columns}
model.update_cell_counts(cell_counts)

# Train the model
model.train(C_train, C_val, Y_train, Y_val, X_ref)

# Make predictions on new data
predictions = model.predict(Y_new)

# Save the trained model
model.save_model('trained_model.pkl')
```

## Tutorial
In Tutorial.ipynb you find a detailed tutorial on how to use HIDE on an example breast cancer data set downloaded from DISCO (https://www.immunesinglecell.org/cell_type). HIDE_example.py also features the same example as how it could be used to integrate it into your pipeline.

## Requirements
All necessary requirements are frozen in requirements.txt. We recommend python version 3.13.2, as this was the version we used for freezing the requirements. For other version you must install the requirements on your own.

## Planned updates
- Make HIDE available to non-coding users by implementing it into our graphical suite of deconvolution tools (Deconomix)
- Automatic creation of the reference profiles out of single cells
- Improved usability of the class


## Further algorithms
If you are looking for other deconvolution algorithms without the need of a hierarchical cell type structure, we recommend you to explore the DeconomiX package (http://deconomix.bioinf.med.uni-goettingen.de)

## Bugs, Issues, Correspondence
If you find any bug, struggle with loading a specific dataset, have any question or comment, please don't hesitate to contact us either via GitHub or via Email to franziska(dot)gortler(at)uib(dot)no or dennis(dot)k(dot)volkl(at)uib(dot)no (replace (dot) with . and (at) with @)

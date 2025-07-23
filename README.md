# HIDE
 
Implementation of HIDE as proposed in Völkl et al (2025), HIDE: Hierarchical cell-type Deconvolution, https://doi.org/10.1093/bioinformatics/btaf179
<br>
<br>
HIDE (Hierarchical cell-type Deconvolution) is a computational approach on infering cellular distribution from bulk transcriptomic data. In comparison to other algorithms HIDE incorporates a hierarchical cell-type structure. For a detailed description of its advantages, we refer to our paper.

## Tutorial
In Tutorial.ipynb you find a detailed tutorial on how to use HIDE on an example breast cancer data set downloaded from DISCO (https://www.immunesinglecell.org/cell_type). HIDE_example.py also features the same example as how it could be used to integrate it into your pipeline.

## Requirements
All necessary requirements are frozen in requirements.txt. We recommend python version 3.9.6, as this was the version we used for development.

## Planned updates
- Make HIDE available to non-coding users by implementing it into our graphical suite of deconvolution tools (Deconomix)
- Automatic creation of the reference profiles out of single cells
- Improved usability of the class

## Further algorithms
If you are looking for other deconvolution algorithms without the need of a hierarchical cell type structure, we recommend you to explore the Deconomix package (http://deconomix.bioinf.med.uni-goettingen.de)

## Bugs, Issues, Correspondence
If you find any bug, struggle with loading a specific dataset, have any question or comment, please don't hesitate to contact us either via GitHub or via Email to franziska(dot)gortler(at)uib(dot)no or dennis(dot)voelkl(at)stud(dot)uni-regensburg(dot)de (replace (dot) with . and (at) with @)

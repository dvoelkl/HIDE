##########################################################
#
# Methods for HIDE
#
##########################################################




# %% ####################################################################################
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.io import mmread
import numpy as np
import pandas as pd
from methods import ADTD, DTD
from pipelines_utils import flatten_nested_dict, process_composition, estimate_corr, linReg, adjustToLinReg 
from utils import calculate_estimated_composition
import datetime


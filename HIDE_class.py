##########################################################
#
# HIDE Class - Simplified interface for HIDE algorithm
#
##########################################################

import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.io import mmread
import numpy as np
import pandas as pd
from methods import ADTD, DTD
from pipelines_utils import flatten_nested_dict, process_composition, estimate_corr, linReg, adjustToLinReg 
from utils import calculate_estimated_composition
import datetime
from hDTD import HIDE, subtypes_estimate_composition

class HIDEModel:
    """
    Simplified HIDE class for hierarchical cell-type deconvolution.
    
    This class provides a scikit-learn-like interface for the HIDE algorithm
    with separate train() and predict() methods.
    
    Parameters
    ----------
    subtypes_dict : dict
        Hierarchical dictionary defining cell type structure
    count_celltypes : dict
        Dictionary containing counts of each cell type in training data
    iterations_dtd : int, default=500
        Number of iterations for DTD training
    save_path : str, optional
        Path to save intermediate results and plots
    save_models : bool, default=False
        Whether to save model parameters (gamma, X, LinReg)
    save_compositions : bool, default=False
        Whether to save estimated compositions
    verbose : bool, default=True
        Whether to print progress information
    """
    
    def __init__(self, subtypes_dict, count_celltypes, iterations_dtd=500, 
                 save_path=None, save_models=False, save_compositions=False, verbose=True):
        
        # Set attributes first
        self.subtypes_dict = subtypes_dict
        self.count_celltypes = count_celltypes
        self.iterations_dtd = iterations_dtd
        self.save_path = save_path
        self.save_models = save_models
        self.save_compositions = save_compositions
        self.verbose = verbose
        
        # Model parameters (will be set after training)
        self.is_trained = False
        self.training_results = None
        self.model_parameters = None
        
        # Validate subtypes_dict structure after setting verbose
        self._validate_subtypes_dict(subtypes_dict)
    
    def _validate_subtypes_dict(self, subtypes_dict):
        """
        Validate the structure of subtypes_dict to ensure it's properly formatted.
        """
        if not isinstance(subtypes_dict, dict):
            raise ValueError("subtypes_dict must be a dictionary")
        
        for main_type, subtypes in subtypes_dict.items():
            if not isinstance(subtypes, dict):
                raise ValueError(f"Subtypes for '{main_type}' must be a dictionary, got {type(subtypes)}")
            
            for subtype, subsubtypes in subtypes.items():
                if not isinstance(subsubtypes, (list, set)):
                    raise ValueError(f"Sub-subtypes for '{main_type}' -> '{subtype}' must be a list or set, got {type(subsubtypes)}")
        
        if self.verbose:
            print(f"✓ Subtypes dictionary validated successfully")
            print(f"  - {len(subtypes_dict)} main cell types")
            total_subtypes = sum(len(subtypes) for subtypes in subtypes_dict.values())
            print(f"  - {total_subtypes} total subtypes")
        
    def train(self, C_train, C_val, Y_train, Y_val, X_ref):
        """
        Train the HIDE model on training and validation data.
        
        Parameters
        ----------
        C_train : pd.DataFrame
            Training composition matrix (cell types x samples)
        C_val : pd.DataFrame
            Validation composition matrix (cell types x samples)
        Y_train : pd.DataFrame
            Training bulk expression data (genes x samples)
        Y_val : pd.DataFrame
            Validation bulk expression data (genes x samples)
        X_ref : pd.DataFrame
            Reference expression matrix (genes x cell types)
            
        Returns
        -------
        self : HIDEModel
            Returns self for method chaining
        """
        
        if self.verbose:
            print("="*50)
            print("      Training HIDE Model")
            print("="*50)
        
        # Train using the original HIDE function
        self.training_results = HIDE(
            C_train_all=C_train,
            C_val_all=C_val,
            Y_train_all=Y_train,
            Y_val_all=Y_val,
            X_ref_all=X_ref,
            subtypes_dict=self.subtypes_dict,
            count_celltypes=self.count_celltypes,
            iterations_dtd=self.iterations_dtd,
            savePath=self.save_path,
            saveC=self.save_compositions,
            saveGammaAndX=self.save_models
        )
        
        # Extract model parameters for later prediction
        self._extract_model_parameters()
        
        self.is_trained = True
        
        if self.verbose:
            print("\n" + "="*50)
            print("      Training Complete")
            print("="*50)
            print(f"Training correlation: {self.training_results['corr_train']:.4f}")
            print(f"Validation correlation: {self.training_results['corr_val']:.4f}")
        
        return self
    
    def predict(self, Y_new):
        """
        Predict cell type compositions for new bulk data.
        
        Parameters
        ----------
        Y_new : pd.DataFrame
            New bulk expression data (genes x samples)
            
        Returns
        -------
        predictions : dict
            Dictionary containing predictions for each hierarchical level:
            - 'major': Main cell type predictions
            - 'minor': Sub-cell type predictions  
            - 'sub': Sub-sub-cell type predictions
        """
        
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions. Call train() first.")
        
        if self.verbose:
            print("="*50)
            print("      Making Predictions")
            print("="*50)
            print(f"Predicting for {Y_new.shape[1]} samples...")
        
        predictions = {}
        
        # 1. Predict major cell types
        if self.verbose:
            print("-> Predicting major cell types...")
        
        major_predictions = self._predict_major_celltypes(Y_new)
        predictions['major'] = major_predictions
        
        # 2. Predict minor cell types (subtypes)
        if self.verbose:
            print("-> Predicting minor cell types...")
        
        minor_predictions = {}
        for celltype in self.model_parameters['minor'].keys():
            if self.verbose:
                print(f"   -> Predicting {celltype} subtypes...")
            
            minor_pred = self._predict_subtypes(
                Y_new, celltype, major_predictions, level='minor'
            )
            minor_predictions[celltype] = minor_pred
        
        predictions['minor'] = minor_predictions
        
        # 3. Predict sub-subtypes if they exist
        if self.model_parameters['sub']:
            if self.verbose:
                print("-> Predicting sub-subtypes...")
            
            sub_predictions = {}
            for subtype in self.model_parameters['sub'].keys():
                if self.verbose:
                    print(f"   -> Predicting {subtype} sub-subtypes...")
                
                # Find parent celltype
                parent_celltype = self._find_parent_celltype(subtype)
                if parent_celltype:
                    sub_pred = self._predict_subtypes(
                        Y_new, subtype, minor_predictions[parent_celltype], level='sub'
                    )
                    sub_predictions[subtype] = sub_pred
            
            predictions['sub'] = sub_predictions
        else:
            predictions['sub'] = {}
        
        if self.verbose:
            print(f"-> Predictions complete for {Y_new.shape[1]} samples")
        
        return predictions
    
    def _extract_model_parameters(self):
        """Extract model parameters needed for prediction."""
        
        self.model_parameters = {
            'major': {
                'gamma': self.training_results['major']['model_main'].gamma,
                'X_ref': self.training_results['major']['X_main'],
                'LinReg': self.training_results['major']['LinReg']
            },
            'minor': {},
            'sub': {}
        }
        
        # Extract minor celltype parameters
        for celltype, results in self.training_results['minor'].items():
            self.model_parameters['minor'][celltype] = {
                'gamma': results['model'].gamma,
                'X_ref': results['X_sub'],
                'LinReg': results['LinReg']
            }
        
        # Extract sub-subtype parameters
        for subtype, results in self.training_results['sub'].items():
            self.model_parameters['sub'][subtype] = {
                'gamma': results['model'].gamma,
                'X_ref': results['X_sub'],
                'LinReg': results['LinReg']
            }
    
    def _predict_major_celltypes(self, Y_new):
        """Predict major cell type compositions."""
        
        params = self.model_parameters['major']
        
        # DTD estimation
        C_est = calculate_estimated_composition(
            params['X_ref'], Y_new, params['gamma']
        )
        
        # Apply linear regression adjustment
        C_est = adjustToLinReg(C_est, params['LinReg'])
        
        # Ensure non-negative and normalize
        C_est = C_est.clip(lower=0)
        C_est = C_est / C_est.sum(axis=0)
        
        return C_est
    
    def _predict_subtypes(self, Y_new, celltype, parent_composition, level='minor'):
        """Predict subtypes for a given cell type."""
        
        params = self.model_parameters[level][celltype]
        
        # Use the appropriate reference matrix for parent type
        if level == 'minor':
            X_main = self.model_parameters['major']['X_ref']
        else:
            # For sub-subtypes, find the parent's reference matrix
            parent_celltype = self._find_parent_celltype(celltype)
            X_main = self.model_parameters['minor'][parent_celltype]['X_ref']
        
        # Estimate subtype composition
        result = subtypes_estimate_composition(
            X_sub=params['X_ref'],
            X_main=X_main,
            Y_all=Y_new,
            type_to_extend=celltype,
            C_main=parent_composition,
            gamma=params['gamma'],
            linReg=params['LinReg']
        )
        
        return result['C_est']
    
    def _find_parent_celltype(self, subtype):
        """Find the parent cell type for a given subtype."""
        
        # Search in the minor results to find which celltype contains this subtype
        for celltype, results in self.training_results['minor'].items():
            if subtype in results['C_train'].index:
                return celltype
        
        return None
    
    def get_training_summary(self):
        """
        Get a summary of training results.
        
        Returns
        -------
        summary : dict
            Dictionary containing training metrics and information
        """
        
        if not self.is_trained:
            raise ValueError("Model must be trained first.")
        
        summary = {
            'training_correlation': self.training_results['corr_train'],
            'validation_correlation': self.training_results['corr_val'],
            'major_celltypes': list(self.model_parameters['major']['X_ref'].columns),
            'minor_celltypes': {
                celltype: list(params['X_ref'].columns) 
                for celltype, params in self.model_parameters['minor'].items()
            },
            'sub_celltypes': {
                subtype: list(params['X_ref'].columns) 
                for subtype, params in self.model_parameters['sub'].items()
            },
            'used_subset_types': self.training_results['used_subset_types']
        }
        
        return summary
    
    def save_model(self, filepath):
        """
        Save the trained model parameters to file.
        
        Parameters
        ----------
        filepath : str
            Path to save the model
        """
        
        if not self.is_trained:
            raise ValueError("Model must be trained before saving.")
        
        import pickle
        
        model_data = {
            'subtypes_dict': self.subtypes_dict,
            'count_celltypes': self.count_celltypes,
            'iterations_dtd': self.iterations_dtd,
            'model_parameters': self.model_parameters,
            'training_results': self.training_results
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        if self.verbose:
            print(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath, verbose=True):
        """
        Load a trained model from file.
        
        Parameters
        ----------
        filepath : str
            Path to the saved model
        verbose : bool, default=True
            Whether to print loading information
            
        Returns
        -------
        model : HIDEModel
            Loaded HIDE model
        """
        
        import pickle
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create new instance
        model = cls(
            subtypes_dict=model_data['subtypes_dict'],
            count_celltypes=model_data['count_celltypes'],
            iterations_dtd=model_data['iterations_dtd'],
            verbose=verbose
        )
        
        # Restore trained state
        model.model_parameters = model_data['model_parameters']
        model.training_results = model_data['training_results']
        model.is_trained = True
        
        if verbose:
            print(f"Model loaded from {filepath}")
        
        return model

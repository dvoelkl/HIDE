
##########################################################
#
# HIDE Class
#
##########################################################

import matplotlib.pyplot as plt
from scipy.stats import linregress
import numpy as np
import pandas as pd
from deconomix.methods import DTD
from HIDE_utils import flatten_nested_dict, process_composition, estimate_corr, estimate_nmae
from deconomix.utils import calculate_estimated_composition
from HIDE_dataloader import disco_read_metadata
from HIDE_utils import merge_celltypes
import pickle

class HIDEModel:
    """
    HIDE class for hierarchical cell-type deconvolution.
    
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
        Whether to save model parameters (gamma, X, LinReg)
    save_compositions : bool, default=False
        Whether to save estimated compositions
    verbose : bool, default=True
        Whether to print progress information
    """
    
    def __init__(self, subtypes_dict, count_celltypes, iterations_dtd=500, 
                 save_path=None, save_compositions=False, verbose=True):
        
        # Set attributes first
        self.subtypes_dict = subtypes_dict
        self.count_celltypes = count_celltypes
        self.iterations_dtd = iterations_dtd
        self.save_path = save_path
        self.save_compositions = save_compositions
        self.verbose = verbose
        
        self.is_trained = False
        self.training_results = None
        self.model_parameters = None
        
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
            print(f"Subtypes dictionary validated")
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
        
        # Train using the HIDE function
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
            saveC=self.save_compositions
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
        Predict cell type compositions for bulk data.
        
        Parameters
        ----------
        Y_new : pd.DataFrame
            Bulk expression data (genes x samples)
            
        Returns
        -------
        predictions : dict
            - 'major': Main cell type predictions
            - 'minor': Minor cell type predictions  
            - 'sub': Sub cell type predictions
        """
        
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions. Call train() first.")
        

        major_predictions = self._predict_major_celltypes(Y_new)


        minor_predictions = {}
        for celltype in self.model_parameters['minor'].keys():

            minor_pred = self._predict_subtypes(
                Y_new, celltype, major_predictions, level='minor'
            )
            minor_predictions[celltype] = minor_pred

        sub_predictions = {}
        if self.model_parameters['sub']:

            for subtype in self.model_parameters['sub'].keys():

                parent_celltype = self._find_parent_celltype(subtype)
                
                if parent_celltype and parent_celltype in minor_predictions:
                    sub_pred = self._predict_subtypes(
                        Y_new, subtype, minor_predictions[parent_celltype], level='sub'
                    )
                    sub_predictions[subtype] = sub_pred

        C_major = major_predictions.copy()

        sample_cols = Y_new.columns
        minor_rows = []
        major_rows = list(C_major.index)
        for main_ct in major_rows:
            if main_ct in minor_predictions:
                minor_rows.extend(list(minor_predictions[main_ct].index))
            else:
                minor_rows.append(main_ct)

        C_minor = pd.DataFrame(0.0, index=minor_rows, columns=sample_cols)

        for main_ct in major_rows:
            if main_ct in minor_predictions:
                df = minor_predictions[main_ct]

                for r in df.index:
                    C_minor.loc[r] = df.loc[r]
            else:
                C_minor.loc[main_ct] = C_major.loc[main_ct]


        deep_rows = []
        for minor_name in minor_rows:
            if minor_name in self.model_parameters['sub']:
                sub_cols = list(self.model_parameters['sub'][minor_name]['X_ref'].columns)
                deep_rows.extend(sub_cols)
            else:
                deep_rows.append(minor_name)

        C_est = pd.DataFrame(0.0, index=deep_rows, columns=sample_cols)

        for minor_name in minor_rows:
            if minor_name in sub_predictions:
                df = sub_predictions[minor_name]
                for r in df.index:
                    C_est.loc[r] = df.loc[r]
            else:
                C_est.loc[minor_name] = C_minor.loc[minor_name]

        return {
            'major': C_major,
            'minor': C_minor,
            'sub': C_est
        }
    
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

        # Extract relationship between subtypes to minor cell type for reduced memory consumption of saved model
        self.subtype_to_parent = {}

        # Extract minor celltype parameters
        for celltype, results in self.training_results['minor'].items():
            self.model_parameters['minor'][celltype] = {
                'gamma': results['model'].gamma,
                'X_ref': results['X_sub'],
                'LinReg': results['LinReg']
            }

            for subtype in results['C_train'].index:
                self.subtype_to_parent[subtype] = celltype

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
        C_est = HIDEModel.adjustToLinReg(C_est, params['LinReg'])
        
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
        """Find the parent cell type for a given subtype using the Zuordnung."""
        return self.subtype_to_parent.get(subtype, None)
    
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
    
    def get_metrics(self):
        '''Function for benchmarking'''

        if not self.is_trained:
            raise ValueError("Model must be trained first.")

        # Major
        major_types = list(self.model_parameters['major']['X_ref'].columns)
        train_major_corr = self.training_results['major']['train_main_corr']
        train_major_nmae = self.training_results['major']['train_main_nmae']
        val_major_corr = self.training_results['major']['val_main_corr']
        val_major_nmae = self.training_results['major']['val_main_nmae']

        # Add prefixes to celltype, such that it can't be a duplicate index later on
        train_major_corr = train_major_corr.add_prefix('major_', axis=0)
        train_major_nmae = train_major_nmae.add_prefix('major_', axis=0)
        val_major_corr = val_major_corr.add_prefix('major_', axis=0)
        val_major_nmae = val_major_nmae.add_prefix('major_', axis=0)

        # Minor
        minor_types = []
        train_minor_corr = pd.Series(dtype=float)
        train_minor_nmae = pd.Series(dtype=float)
        val_minor_corr = pd.Series(dtype=float)
        val_minor_nmae = pd.Series(dtype=float)
        for celltype, params in self.model_parameters['minor'].items():
            for subtype in params['X_ref'].columns:
                minor_types.append(subtype)
                train_minor_corr[subtype] = self.training_results['minor'][celltype]['train_corr'].get(subtype, float('nan'))
                train_minor_nmae[subtype] = self.training_results['minor'][celltype]['train_nmae'].get(subtype, float('nan'))
                val_minor_corr[subtype] = self.training_results['minor'][celltype]['val_corr'].get(subtype, float('nan'))
                val_minor_nmae[subtype] = self.training_results['minor'][celltype]['val_nmae'].get(subtype, float('nan'))

        # Add prefixes to celltype, such that it can't be a duplicate index later on
        train_minor_corr = train_minor_corr.add_prefix('minor_', axis=0)
        train_minor_nmae = train_minor_nmae.add_prefix('minor_', axis=0)
        val_minor_corr = val_minor_corr.add_prefix('minor_', axis=0)
        val_minor_nmae = val_minor_nmae.add_prefix('minor_', axis=0)

        # Sub
        sub_types = []
        train_sub_corr = pd.Series(dtype=float)
        train_sub_nmae = pd.Series(dtype=float)
        val_sub_corr = pd.Series(dtype=float)
        val_sub_nmae = pd.Series(dtype=float)
        for subtype, params in self.model_parameters['sub'].items():
            for subsubtype in params['X_ref'].columns:
                sub_types.append(subsubtype)
                train_sub_corr[subsubtype] = self.training_results['sub'][subtype]['train_corr'].get(subsubtype, float('nan'))
                train_sub_nmae[subsubtype] = self.training_results['sub'][subtype]['train_nmae'].get(subsubtype, float('nan'))
                val_sub_corr[subsubtype] = self.training_results['sub'][subtype]['val_corr'].get(subsubtype, float('nan'))
                val_sub_nmae[subsubtype] = self.training_results['sub'][subtype]['val_nmae'].get(subsubtype, float('nan'))

        # Add prefixes to celltype, such that it can't be a duplicate index later on
        train_sub_corr = train_sub_corr.add_prefix('sub_', axis=0)
        train_sub_nmae = train_sub_nmae.add_prefix('sub_', axis=0)
        val_sub_corr = val_sub_corr.add_prefix('sub_', axis=0)
        val_sub_nmae = val_sub_nmae.add_prefix('sub_', axis=0)

        train_corr_all = pd.concat([train_major_corr, train_minor_corr, train_sub_corr])
        train_nmae_all = pd.concat([train_major_nmae, train_minor_nmae, train_sub_nmae])
        val_corr_all = pd.concat([val_major_corr, val_minor_corr, val_sub_corr])
        val_nmae_all = pd.concat([val_major_nmae, val_minor_nmae, val_sub_nmae])

        train_metrics = pd.DataFrame({
            'Correlation': train_corr_all,
            'NMAE': train_nmae_all
        })
        val_metrics = pd.DataFrame({
            'Correlation': val_corr_all,
            'NMAE': val_nmae_all
        })

        return train_metrics, val_metrics

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
        
        model_data = {
            'subtypes_dict': self.subtypes_dict,
            'count_celltypes': self.count_celltypes,
            'iterations_dtd': self.iterations_dtd,
            'model_parameters': self.model_parameters,
            'subtype_to_parent': self.subtype_to_parent
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        if self.verbose:
            print(f"Model saved to {filepath}")
    
    @classmethod
    def from_hierarchy_file(cls, hierarchy_file_path, X_ref, iterations_dtd=500, 
                           save_path=None, save_compositions=False, verbose=True):
        """
        Create a HIDEModel from a cell hierarchy CSV file.
        
        This method simplifies the model creation by automatically parsing the hierarchy file
        and calculating cell type counts from the reference matrix.
        
        Parameters
        ----------
        hierarchy_file_path : str
            Path to the CSV file containing cell hierarchy with columns:
            'celltype_major', 'celltype_minor', 'celltype_sub'
        X_ref : pd.DataFrame
            Reference expression matrix (genes x cell types)
        iterations_dtd : int, default=500
            Number of iterations for DTD training
        save_path : str, optional
            Path to save intermediate results and plots
        save_compositions : bool, default=False
            Whether to save estimated compositions
        verbose : bool, default=True
            Whether to print progress information
            
        Returns
        -------
        model : HIDEModel
            Configured HIDE model ready for training
            
        Examples
        --------
        >>> # Simple usage
        >>> model = HIDEModel.from_hierarchy_file('cell_hierarchy.csv', X_ref)
        >>> model.train(C_train, C_val, Y_train, Y_val, X_ref)
        
        >>> # With custom parameters
        >>> model = HIDEModel.from_hierarchy_file(
        ...     'cell_hierarchy.csv', 
        ...     X_ref,
        ...     iterations_dtd=1000,
        ...     save_path='./results/',
        ...     verbose=True
        ... )
        """
        
        if verbose:
            print(f"-> Loading cell hierarchy from: {hierarchy_file_path}")
        
        try:
            # Load metadata using the existing function
            meta_major = disco_read_metadata(hierarchy_file_path, "celltype_major", 'celltype_minor')
            main_celltypes = meta_major['main_celltypes']
            sub_celltypes = meta_major['sub_celltypes']
            
            meta_minor = disco_read_metadata(hierarchy_file_path, "celltype_minor", 'celltype_sub')
            subset_celltypes = meta_minor['sub_celltypes']
            
            # Merge the cell type dictionaries
            merged_celltypes = merge_celltypes(sub_celltypes, subset_celltypes)
            
            if verbose:
                print(f"-> Found {len(main_celltypes)} major cell types:")
                for i, celltype in enumerate(main_celltypes):
                    print(f"   {i+1}. {celltype}")
            
        except Exception as e:
            raise ValueError(f"Failed to load hierarchy file '{hierarchy_file_path}': {str(e)}")
        
        # Filter subtypes dictionary to only include cell types present in X_ref
        if verbose:
            print(f"-> Filtering cell types based on reference matrix...")
            
        original_count = sum(len(list(subtypes.values())[0]) if subtypes and isinstance(list(subtypes.values())[0], list) 
                           else len(subtypes) for subtypes in merged_celltypes.values())
        
        # Filter the hierarchical structure
        filtered_celltypes = {}
        for main_celltype in main_celltypes:
            if main_celltype in merged_celltypes:
                filtered_celltypes[main_celltype] = {}
                for subtype, sub_subtypes in merged_celltypes[main_celltype].items():
                    # Filter sub-subtypes to only include those present in X_ref
                    valid_sub_subtypes = [sst for sst in sub_subtypes if sst in X_ref.columns]
                    if valid_sub_subtypes:
                        filtered_celltypes[main_celltype][subtype] = valid_sub_subtypes
                
                # Remove empty main celltypes
                if not filtered_celltypes[main_celltype]:
                    del filtered_celltypes[main_celltype]
        
        merged_celltypes = filtered_celltypes
        
        filtered_count = sum(len(sub_subtypes) for subtypes in merged_celltypes.values() 
                           for sub_subtypes in subtypes.values())
        
        if verbose:
            print(f"-> Filtered cell types: {original_count} → {filtered_count}")
            if original_count > filtered_count:
                print(f"-> {original_count - filtered_count} cell types were not found in reference matrix")
        
        # Calculate cell type counts from reference matrix
        if verbose:
            print(f"-> Calculating cell type counts from reference matrix...")
            
        count_celltypes = {}
        available_celltypes = set(X_ref.columns)
        
        for celltype in X_ref.columns.unique():
            if celltype in available_celltypes:
                # Use a default count of 1 for each cell type if no other information is available
                # This can be overridden later if actual counts are available
                count_celltypes[celltype] = 1
        
        if verbose:
            print(f"-> Calculated counts for {len(count_celltypes)} cell types")
            print(f"-> Note: Using uniform counts (1) for all cell types. You can update these later with actual counts.")
        
        # Create and return the model
        model = cls(
            subtypes_dict=merged_celltypes,
            count_celltypes=count_celltypes,
            iterations_dtd=iterations_dtd,
            save_path=save_path,
            save_compositions=save_compositions,
            verbose=verbose
        )
        
        if verbose:
            print(f"HIDEModel created successfully!")
            print(f"   - Major cell types: {len([ct for ct in merged_celltypes.keys() if merged_celltypes[ct]])}")
            total_subtypes = sum(len(subtypes) for subtypes in merged_celltypes.values())
            print(f"   - Total subtypes: {total_subtypes}")
            print(f"   - Available cell types in reference: {len(count_celltypes)}")
        
        return model
    
    def update_cell_counts(self, count_celltypes):
        """
        Update cell type counts after model creation.
        
        This is useful when you create a model using from_hierarchy_file() with default counts
        and later want to provide actual cell counts from your training data.
        
        Parameters
        ----------
        count_celltypes : dict
            Dictionary containing counts of each cell type
            
        Examples
        --------
        >>> model = HIDEModel.from_hierarchy_file('hierarchy.csv', X_ref)
        >>> # Calculate actual counts from training data
        >>> actual_counts = {celltype: C_train.sum(axis=1)[celltype] 
        ...                  for celltype in X_ref.columns.unique()}
        >>> model.update_cell_counts(actual_counts)
        """
        if self.verbose:
            print(f"-> Updating cell type counts...")
            print(f"   - Previous count keys: {len(self.count_celltypes)}")
            print(f"   - New count keys: {len(count_celltypes)}")
        
        self.count_celltypes.update(count_celltypes)
        
        if self.verbose:
            print(f"-> Cell type counts updated successfully!")

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
        model.is_trained = True
        model.subtype_to_parent = model_data.get('subtype_to_parent', {})

        if verbose:
            print(f"Model loaded from {filepath}")

        return model
    

    @staticmethod
    def linReg(C_true, C_est):
        # Get celltypes
        celltypes = C_true.index.unique()

        linReg_results = pd.DataFrame(index=celltypes, columns=['slope', 'intercept', 'p'])

        for celltype in celltypes:
            try:
                slope, intercept, _, p, _ = linregress(C_est.loc[celltype,:], C_true.loc[celltype,:])
            except:
                # Catch case where some cell type is constantly zero, then f(x) = id(x)
                print(f"-> Warning: {celltype} proportions were constant")
                slope = 1 #
                intercept = 0
                p = 0

            linReg_results.loc[celltype, 'slope'] = slope
            linReg_results.loc[celltype, 'intercept'] = intercept
            linReg_results.loc[celltype, 'p'] = p

        return linReg_results

    @staticmethod
    def adjustToLinReg(C_est, linReg_results):
        # Get celltypes
        celltypes = linReg_results.index.unique()

        for celltype in celltypes:
            # y = mx + t
            C_est.loc[celltype] = linReg_results.loc[celltype, 'slope'] * C_est.loc[celltype] + linReg_results.loc[celltype, 'intercept']

        return C_est 

# %% ####################################################################################
def HIDE(C_train_all, C_val_all, 
                    Y_train_all, Y_val_all, 
                    X_ref_all, 
                    subtypes_dict, count_celltypes,
                    iterations_dtd=500,
                    savePath=None, saveC=False, saveC_prefix=''): 

    # Ensure everything is in correct order and normalization is done
    
    X_ref_all = X_ref_all.reindex(C_train_all.index.values, axis=1)

    # Normalize everything spotwise to one
    X_ref_all = X_ref_all / X_ref_all.sum(axis=0)

    Y_train_all = Y_train_all / Y_train_all.sum(axis=0)
    Y_val_all = Y_val_all / Y_val_all.sum(axis=0)

    C_train_all = C_train_all / C_train_all.sum(axis=0)
    C_val_all = C_val_all / C_val_all.sum(axis=0)
    
    # Variables to hold correlation info
    corr_train_dtd_main = 0
    corr_train_dtd_sub = []
    corr_train_dtd_subset = []

    corr_val_dtd_main = 0
    corr_val_dtd_sub = []
    corr_val_dtd_subset = []

    used_subsettypes = []

    X_ref_subtypes = pd.DataFrame()
    subtype_counts = {}
    for celltype in list(subtypes_dict.keys()):
        for subtype in list(subtypes_dict[celltype]):
            tot_cells_of_type = 0
            weight_sum_of_type = pd.Series(0, index=X_ref_all.index)
            for subcelltype in list(subtypes_dict[celltype][subtype]):
                if subcelltype in X_ref_all.columns:
                    weight_sum_of_type += count_celltypes[subcelltype] * X_ref_all[subcelltype]
                    tot_cells_of_type += count_celltypes[subcelltype]
                else:
                    # Cleanup dictionary
                    print(f"!!! WARNING !!!")
                    print(f"!!! {subcelltype} not included in reference profile !!!")
                    print(f"!!! Dropping {subcelltype} from dictionary !!!")
                    subtypes_dict[celltype][subtype].remove(subcelltype)
                    if len(subtypes_dict[celltype][subtype]) == 0:
                        del subtypes_dict[celltype][subtype]
                        # If no subtypes remain for the cell type, clean up the cell type
                        if len(subtypes_dict[celltype]) == 0:
                            del subtypes_dict[celltype]

            subtype_counts[subtype] = tot_cells_of_type
            X_ref_subtypes[subtype] = weight_sum_of_type / tot_cells_of_type

    subtypes_only_dict = flatten_nested_dict(subtypes_dict)

    new_rows_train = []
    new_rows_val = []
    for celltype in subtypes_dict.keys():
        for subtype in subtypes_dict[celltype]:
            sub_type_train_sum = C_train_all.loc[subtypes_dict[celltype][subtype]].sum()
            sub_type_train_sum.name = subtype
            new_rows_train.append(sub_type_train_sum)

            sub_type_val_sum = C_val_all.loc[subtypes_dict[celltype][subtype]].sum()
            sub_type_val_sum.name = subtype
            new_rows_val.append(sub_type_val_sum)

    C_train_subtypes = pd.DataFrame(new_rows_train)
    C_val_subtypes = pd.DataFrame(new_rows_val)

    # Train and validate DTD on main celltypes
    results_maintype = subtypes_pipeline_main(C_train_subtypes, 
                                            C_val_subtypes, 
                                            Y_train_all, 
                                            Y_val_all, 
                                            X_ref_subtypes, 
                                            subtypes_only_dict, 
                                            subtype_counts,
                                            iterations_dtd, savePath)
    
    # Add results to corr variables
    corr_val_dtd_main = results_maintype['val_main_corr'].mean()
    corr_train_dtd_main = results_maintype['train_main_corr'].mean()
    
    X_main = results_maintype['X_main']

    if saveC:
        results_maintype['C_main_val_est'].to_csv(savePath+f'_C_main_' + saveC_prefix + f'.csv')

    # Loop through the subtypes, adjust the reference matrices and compositions each time 
    # and store the results into a dictionary
    results_subtypes = {}
    results_subsettype = {}

    for i, celltype in enumerate(subtypes_only_dict.keys()):
        
        if len(subtypes_only_dict[celltype]) > 1:
            result_sub= subtypes_pipeline_sub(C_train_subtypes, 
                                            C_val_subtypes, 
                                            results_maintype['Y_train_main'], 
                                            results_maintype['Y_val_main'], 
                                            X_ref_subtypes, 
                                            X_main,
                                            subtypes_only_dict, 
                                            celltype, 
                                            results_maintype['C_main_train_est'],
                                            results_maintype['C_main_val_est'],
                                            results_maintype['C_main_train'],
                                            results_maintype['model_main'],
                                            iterations_dtd, savePath)
            results_subtypes.update({celltype:result_sub})

            # Add results to corr variables
            corr_val_dtd_sub.extend(result_sub['val_corr']) #[].mean()
            corr_train_dtd_sub.extend(result_sub['train_corr']) #[].mean()

            if saveC:
                result_sub['C_val_est'].to_csv(savePath+f'_C_{celltype}_' + saveC_prefix + f'.csv')

            # Now loop through the subset types
            for j, subtype in enumerate(subtypes_dict[celltype].keys()):
                
                if len(subtypes_dict[celltype][subtype]) > 1:

                    result_subset = subtypes_pipeline_sub(C_train_all, 
                                                C_val_all, 
                                                result_sub['Y_train'], 
                                                result_sub['Y_val'], 
                                                X_ref_all,
                                                result_sub['X_sub'],
                                                subtypes_dict[celltype], 
                                                subtype, 
                                                result_sub['C_train_est'],
                                                result_sub['C_val_est'],
                                                result_sub['C_train'],
                                                result_sub['model'],
                                                iterations_dtd, savePath)
                    results_subsettype.update({subtype:result_subset})

                    # Add results to corr variables
                    corr_val_dtd_subset.extend(result_subset['val_corr'])
                    corr_train_dtd_subset.extend(result_subset['train_corr'])

                    if saveC:
                        result_subset['C_val_est'].to_csv(savePath+f'_C_{subtype}_' + saveC_prefix + f'.csv')

                    used_subsettypes.extend(subtypes_dict[celltype][subtype])
                else:
                    pass
        else:
            pass

    corr_train_dtd_tot = corr_train_dtd_sub
    corr_train_dtd_sub = np.array(corr_train_dtd_sub).mean()

    corr_train_dtd_tot.extend([corr_train_dtd_main])
    corr_train_dtd_tot.extend(corr_train_dtd_subset)
    corr_train_dtd_subset = np.mean(np.array(corr_train_dtd_subset))

    corr_train_dtd_tot = np.mean(np.array(corr_train_dtd_tot))

    corr_val_dtd_tot = corr_val_dtd_sub.copy()
    corr_val_dtd_sub = np.mean(np.array(corr_val_dtd_sub))

    corr_val_dtd_tot.extend([corr_val_dtd_main])
    corr_val_dtd_tot.extend(corr_val_dtd_subset)
    corr_val_dtd_subset = np.mean(np.array(corr_val_dtd_subset))
    
    corr_val_dtd_tot = np.mean(np.array(corr_val_dtd_tot))

    return {
    'major' : results_maintype,
    'minor' : results_subtypes,
    'sub' : results_subsettype,
    'corr_train' : corr_train_dtd_tot,
    'corr_val' : corr_val_dtd_tot,
    'used_subset_types' : used_subsettypes
    }





# %% ####################################################################################
def subtypes_pipeline_main(C_train_all, C_val_all, 
                        Y_train_all, Y_val_all, 
                        X_ref_all, 
                        subtypes_dict, counts_celltypes,
                        iterations_dtd=500, savePath=None):


    savePathTrain = None if savePath is None else savePath + f'/corr_train_dtd_main'
    savePathVal = None if savePath is None else savePath + f'/corr_train_dtd_val'


    X_ref = pd.DataFrame()

    
    for celltype in subtypes_dict.keys():
        tot_cells_of_type = 0
        weight_sum_of_type = pd.Series(0, index=X_ref_all.index)
        for subcelltype in subtypes_dict[celltype]:
            weight_sum_of_type += counts_celltypes[subcelltype] * X_ref_all[subcelltype]
            tot_cells_of_type += counts_celltypes[subcelltype]

        X_ref[celltype] = weight_sum_of_type / tot_cells_of_type
    

    C_train = process_composition(C_train_all, subtypes_dict, '')
    C_val = process_composition(C_val_all, subtypes_dict, '')

    # Ensure everything is in correct order
    X_ref = X_ref.reindex(C_train.index.values, axis=1)
    C_val = C_val.reindex(C_train.index.values, axis=0)

    # Norm everything
    X_ref = X_ref / X_ref.sum(axis=0)
    C_train = (C_train / C_train.sum(axis=0)).fillna(0)
    Y_train_all = Y_train_all / Y_train_all.sum(axis=0)
    Y_val_all = Y_val_all / Y_val_all.sum(axis=0)

    model_dtd = DTD(X_ref, Y_train_all, C_train)

    model_dtd.run(iterations=iterations_dtd)

    
    C_train_est = calculate_estimated_composition(X_ref, Y_train_all, model_dtd.gamma)

    
    C_train_est = C_train_est / C_train_est.sum(axis=0)

    

    train_corr = estimate_corr(C_train, 
                            C_train_est,
                            title='HIDE Maintypes Training', 
                            savePath=savePathTrain)
    train_nmae = estimate_nmae(C_train, C_train_est)
    

    C_val_est = calculate_estimated_composition(X_ref, Y_val_all, model_dtd.gamma)

    linReg_results = HIDEModel.linReg(C_train, C_train_est)
    if savePathVal is not None:
        pass
        #linReg_results.to_csv(savePathVal+f'_LinReg_main.csv')

    C_val_est = HIDEModel.adjustToLinReg(C_val_est, linReg_results)

    C_val_est = C_val_est / C_val_est.sum(axis=0)

    # Ensure that spots are non-negative
    C_val_est = C_val_est.clip(lower=0)

    val_corr = estimate_corr(C_val, 
                            C_val_est,
                            title='HIDE Maintypes Validation', 
                            savePath=savePathVal)
    val_nmae = estimate_nmae(C_val, C_val_est)

    return {
        'train_main_corr' : train_corr,
        'val_main_corr' : val_corr,
        'train_main_nmae' : train_nmae,
        'val_main_nmae' : val_nmae,
        'C_main_train' : C_train,
        'C_main_train_est' : C_train_est,
        'C_main_val_est' : C_val_est,
        'C_main_val' : C_val,
        'X_main' : X_ref,
        'model_main' : model_dtd,
        'LinReg' : linReg_results,
        'Y_train_main' : Y_train_all,
        'Y_val_main' : Y_val_all
    }


# %% ####################################################################################
def subtypes_pipeline_sub(C_train_all, C_val_all, 
                        Y_train_all, Y_val_all, 
                        X_ref_all, X_main,
                        subtypes_dict, type_to_extend,
                        C_est_train_main, C_est_val_main, C_train_main, model_main,
                        iterations_dtd=500,
                        savePath=None):

    savePathTrain = None if savePath is None else savePath + f'/corr_train_dtd_{type_to_extend}'
    savePathVal = None if savePath is None else savePath + f'/corr_val_dtd_{type_to_extend}'

    # Only keep entries of the selected cell maintype
    X_ref = X_ref_all[subtypes_dict[type_to_extend]]
    
    C_train = C_train_all.loc[subtypes_dict[type_to_extend]]
    C_val = C_val_all.loc[subtypes_dict[type_to_extend]]

    # Ensure everything is in correct order
    X_ref = X_ref.reindex(C_train.index.values, axis=1)
    C_val = C_val.reindex(C_train.index.values, axis=0)


    #
    # Training
    #

    # Remove Bulks over other maintypes in Training

    Y_train_to_remove = X_main[X_main.columns.difference([type_to_extend])] @ C_est_train_main.loc[C_est_train_main.index.difference([type_to_extend])]
    Y_train_to_remove.columns = Y_train_all.columns
    Y_train_reduced = (Y_train_all - Y_train_to_remove).clip(lower=0)

    X_ref = X_ref / X_ref.sum(axis=0)
    Y_train_reduced = C_train_main.loc[type_to_extend].to_numpy() * Y_train_reduced / Y_train_reduced.sum(axis=0)

    model_dtd = DTD(X_ref, Y_train_reduced, C_train)
    model_dtd.run(iterations=iterations_dtd)


    estimation_train = subtypes_estimate_composition(X_ref, 
                                X_main, 
                                Y_train_all, 
                                type_to_extend, 
                                C_est_train_main, 
                                model_dtd.gamma, 
                                None)

    train_corr = estimate_corr(C_train, 
                            estimation_train['C_est'],
                            title=f'HIDE {type_to_extend} Training', 
                            savePath=savePathTrain)
    train_nmae = estimate_nmae(C_train, estimation_train['C_est'])
    
    linReg_results = HIDEModel.linReg(C_train, estimation_train['C_est'])
    if savePathVal is not None:
        pass
        #linReg_results.to_csv(savePathVal+f'_LinReg_{type_to_extend}.csv')


    #
    # Validation
    #
    

    estimation_val = subtypes_estimate_composition(X_ref, 
                                X_main, 
                                Y_val_all, 
                                type_to_extend, 
                                C_est_val_main, 
                                model_dtd.gamma, 
                                linReg_results)

    val_corr = estimate_corr(C_val, 
                            estimation_val['C_est'],
                            title=f'HIDE {type_to_extend} Validation', 
                            savePath=savePathVal)
    val_nmae = estimate_nmae(C_val, estimation_val['C_est'])

    return {
        'train_corr' : train_corr,
        'val_corr' : val_corr,
        'train_nmae' : train_nmae,
        'val_nmae' : val_nmae,
        'C_train' : C_train,
        'C_train_est' : estimation_train['C_est'],
        'C_val_est' : estimation_val['C_est'],
        'C_val' : C_val,
        'X_sub' : X_ref,
        'LinReg' : linReg_results,
        'model' : model_dtd,
        'Y_train' : estimation_train['Y_reduced'],
        'Y_val' : estimation_val['Y_reduced']
    }



# %% ####################################################################################
def subtypes_estimate_composition(X_sub, X_main, 
                                Y_all, type_to_extend, 
                                C_main, gamma, 
                                linReg=None):

    #
    # Remove contributions of other celltypes
    #

    Y_to_remove = X_main[X_main.columns.difference([type_to_extend])] @ C_main.loc[C_main.index.difference([type_to_extend])]
    Y_to_remove.columns = Y_all.columns

    Y_reduced = (Y_all - Y_to_remove).clip(lower=0)

    # Catch case where Y_reduced is predicted to be zero at a spot
    zero_sum_indices = (Y_reduced.sum(axis=0)== 0)
    if zero_sum_indices.any():
        print(zero_sum_indices)
        Y_reduced.loc[:, zero_sum_indices] = 0.0001 / Y_reduced.shape[0]

    #
    # Calculate estimated composition
    #
    C_est = calculate_estimated_composition(X_sub, Y_reduced, gamma)

    #
    # Perform linear 
    #
    if linReg is not None:
        C_est = HIDEModel.adjustToLinReg(C_est, linReg)
        C_est = C_est.clip(lower=0) 

    #
    # Renormalize estimations in accordance with the respective maintype
    #
    if len(C_main.columns) != len(C_est.columns):
        raise Exception("Length of spots in C_main and C_est not the same!")
    
    # Ensure columns of c_main and c_est have the same name
    C_est.columns = C_main.columns

    # Set spots, where no transcripts where left to zero
    if zero_sum_indices.any():
        zero_sum_indices.index = C_est.columns
        C_est.loc[:, zero_sum_indices] = 0.0

    xi_val = (C_main.loc[type_to_extend].to_numpy() / C_est.sum(axis=0).to_numpy())
    xi_val = np.nan_to_num(xi_val) # Ensure no nans are in xi

    C_est_xi = pd.DataFrame(xi_val * C_est.to_numpy(), index=C_est.index, columns=C_est.columns)

    return {
        'C_est' : C_est_xi,
        'Y_reduced' : Y_reduced
    }


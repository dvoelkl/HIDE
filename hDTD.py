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

# %% ####################################################################################
def HIDE(C_train_all, C_val_all, 
                      Y_train_all, Y_val_all, 
                      X_ref_all, 
                      subtypes_dict, count_celltypes,
                      iterations_dtd=500,
                      savePath=None, saveC=False, saveC_prefix='', saveGammaAndX=False): 

    print("##################################")
    print("###       HIDE pipeline       ###")
    print("##################################")
    print(f"-> Number of used genes: {len(X_ref_all.index.unique())}")
    print(f"-> list of all celltypes:")
    for i, celltype in enumerate(X_ref_all.columns.unique()): 
        print(f"\t {i}: {celltype}")
    start_time = datetime.datetime.now()
    print(f"-> Started Pipeline at {start_time.time()}")

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
                                              iterations_dtd, savePath, saveGammaAndX=saveGammaAndX)
    
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
                                        iterations_dtd, savePath, saveGammaAndX=saveGammaAndX)
        results_subtypes.update({celltype:result_sub})
        print("") # Just for readabilty

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
                                            iterations_dtd, savePath,
                                            saveGammaAndX=saveGammaAndX)
                results_subsettype.update({subtype:result_subset})
                print("") # Just for readabilty

                # Add results to corr variables
                corr_val_dtd_subset.extend(result_subset['val_corr'])
                corr_train_dtd_subset.extend(result_subset['train_corr'])

                if saveC:
                    result_subset['C_val_est'].to_csv(savePath+f'_C_{subtype}_' + saveC_prefix + f'.csv')

                used_subsettypes.extend(subtypes_dict[celltype][subtype])
            else:
                print(f"-> No subset types of {subtype}\n")
            
          
    end_time = datetime.datetime.now()
    print(f"-> Ended Pipeline at {end_time.time()}")
    print(f"-> Total duration: {end_time - start_time}")

    print(f"### Correlations ###")
    print(f"--- HIDE Training ---")
    print(f"-> Main correlation: {corr_train_dtd_main}")
    corr_train_dtd_tot = corr_train_dtd_sub
    corr_train_dtd_sub = np.array(corr_train_dtd_sub).mean()
    print(f"-> Sub correlation: {corr_train_dtd_sub}")
    corr_train_dtd_tot.extend([corr_train_dtd_main])
    corr_train_dtd_tot.extend(corr_train_dtd_subset)
    corr_train_dtd_subset = np.mean(np.array(corr_train_dtd_subset))
    print(f"-> Subset correlation: {corr_train_dtd_subset}")
    corr_train_dtd_tot = np.mean(np.array(corr_train_dtd_tot))
    print(f"-> Total correlation: {corr_train_dtd_tot}\n")

    print(f"--- HIDE Validation ---")
    print(f"-> Main correlation: {corr_val_dtd_main}")
    corr_val_dtd_tot = corr_val_dtd_sub.copy()
    corr_val_dtd_sub = np.mean(np.array(corr_val_dtd_sub))
    print(f"-> Sub correlation: {corr_val_dtd_sub}")
    corr_val_dtd_tot.extend([corr_val_dtd_main])
    corr_val_dtd_tot.extend(corr_val_dtd_subset)
    corr_val_dtd_subset = np.mean(np.array(corr_val_dtd_subset))
    print(f"-> Subset correlation: {corr_val_dtd_subset}")
    corr_val_dtd_tot = np.mean(np.array(corr_val_dtd_tot))
    print(f"-> Total correlation: {corr_val_dtd_tot}\n")
    print("##################################")
    
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
                           iterations_dtd=500, savePath=None, saveGammaAndX=False):

    print("### HIDE on maintypes ###")

    savePathTrain = None if savePath is None else savePath + f'/corr_train_dtd_main'
    savePathVal = None if savePath is None else savePath + f'/corr_train_dtd_val'

    print(f"-> list of all maintypes:")
    for i, celltype in enumerate(subtypes_dict.keys()): 
        print(f"\t {i}: {celltype}")

    print(f"-> Creating cell maintype reference matrix")
    X_ref = pd.DataFrame()

    
    for celltype in subtypes_dict.keys():
        tot_cells_of_type = 0
        weight_sum_of_type = pd.Series(0, index=X_ref_all.index)
        for subcelltype in subtypes_dict[celltype]:
            weight_sum_of_type += counts_celltypes[subcelltype] * X_ref_all[subcelltype]
            tot_cells_of_type += counts_celltypes[subcelltype]

        X_ref[celltype] = weight_sum_of_type / tot_cells_of_type
    
    print(f"-> Processing compositions")
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

    print(f"-> Training HIDE")
    model_dtd = DTD(X_ref, Y_train_all, C_train)
    model_dtd.run(iterations=iterations_dtd)

    
    C_train_est = calculate_estimated_composition(X_ref, Y_train_all, model_dtd.gamma)

    
    C_train_est = C_train_est / C_train_est.sum(axis=0)

    

    train_corr = estimate_corr(C_train, 
                                                    C_train_est,
                                                    title='HIDE Maintypes Training', 
                                                    savePath=savePathTrain)
    
    print(f"-> Average train correlation: {train_corr.mean()}")

    print(f"-> Validating HIDE")

    C_val_est = calculate_estimated_composition(X_ref, Y_val_all, model_dtd.gamma)

    linReg_results = linReg(C_train, C_train_est)
    if savePathVal is not None:
        pass
        #linReg_results.to_csv(savePathVal+f'_LinReg_main.csv')

    C_val_est = adjustToLinReg(C_val_est, linReg_results)

    C_val_est = C_val_est / C_val_est.sum(axis=0)

    # Ensure that spots are non-negative
    C_val_est = C_val_est.clip(lower=0)

    val_corr = estimate_corr(C_val, 
                            C_val_est,
                            title='HIDE Maintypes Validation', 
                            savePath=savePathVal)
    
    print(f"-> Average val correlation: {val_corr.mean()}")

    if saveGammaAndX:
            model_dtd.gamma.to_csv(savePathVal+f'_gamma_main.csv')
            X_ref.to_csv(savePathVal+f'_X_main.csv')
            linReg_results.to_csv(savePathVal+f'_LinReg_main.csv')

    return {
        'train_main_corr' : train_corr,
        'val_main_corr' : val_corr,
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
                        savePath=None,
                          saveGammaAndX=False):
    
    print(f"### HIDE on {type_to_extend} subtypes ###")
    print(f"-> list of {type_to_extend} subtypes:")
    for i, celltype in enumerate(subtypes_dict[type_to_extend]): 
        print(f"\t {i}: {celltype}")

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
    print("-> Clearing training bulks")
    Y_train_to_remove = X_main[X_main.columns.difference([type_to_extend])] @ C_est_train_main.loc[C_est_train_main.index.difference([type_to_extend])]
    Y_train_to_remove.columns = Y_train_all.columns
    Y_train_reduced = (Y_train_all - Y_train_to_remove).clip(lower=0)

    X_ref = X_ref / X_ref.sum(axis=0)
    Y_train_reduced = C_train_main.loc[type_to_extend].to_numpy() * Y_train_reduced / Y_train_reduced.sum(axis=0)

    print(f"-> Training HIDE")
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
    
    linReg_results = linReg(C_train, estimation_train['C_est'])
    if savePathVal is not None:
        pass
        #linReg_results.to_csv(savePathVal+f'_LinReg_{type_to_extend}.csv')

    print(f"-> Average train correlation: {train_corr.mean()}")

    #
    # Validation
    #

    print(f"-> Validating HIDE")
    

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
    
    print(f"-> Average val correlation: {val_corr.mean()}")

    if saveGammaAndX:
        model_dtd.gamma.to_csv(savePathVal+f'_gamma_{type_to_extend}.csv')
        X_ref.to_csv(savePathVal+f'_X_{type_to_extend}.csv')
        linReg_results.to_csv(savePathVal+f'_LinReg_{type_to_extend}.csv')

    return {
        'train_corr' : train_corr,
        'val_corr' : val_corr,
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
    print("-> clearing bulks")
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
        C_est = adjustToLinReg(C_est, linReg)
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

    C_est_xi = pd.DataFrame(xi_val*C_est.to_numpy(), index=C_est.index)

    return {
        'C_est' : C_est_xi,
        'Y_reduced' : Y_reduced
    }

#!/usr/bin/env python3
"""
Beispiel für die Verwendung von HIDE mit eigenen Daten
Zeigt den kompletten Workflow: Training -> Validierung -> Anwendung
"""

import pandas as pd
import numpy as np
from hDTD import HIDE, subtypes_estimate_composition
from pipelines_dataloader import disco_read_metadata
from pipelines_utils import merge_celltypes, filter_subtypes_by_dataframe_columns, adjustToLinReg
from utils import calculate_estimated_composition

def prepare_data():
    """
    Lade und bereite die Daten vor
    """
    print("=== Daten vorbereiten ===")
    
    # 1. Hierarchie-Struktur laden
    path_to_data = "./data/"
    meta = disco_read_metadata(path_to_data + 'cell_hierarchy.csv', 
                              "celltype_major", 'celltype_minor')
    main_celltypes = meta['main_celltypes']
    sub_celltypes = meta['sub_celltypes']
    
    # Erweiterte Hierarchie für Untertypen
    meta_sub = disco_read_metadata(path_to_data + 'cell_hierarchy.csv', 
                                  "celltype_minor", 'celltype_sub')
    subset_celltypes = meta_sub['sub_celltypes']
    
    # Hierarchie zusammenführen
    sub_celltypes = merge_celltypes(sub_celltypes, subset_celltypes)
    
    # 2. Trainingsdaten laden
    X_train = pd.read_csv(path_to_data + "X_train.csv", index_col=0)
    Y_train = pd.read_csv(path_to_data + "train_data.csv", index_col=0)
    C_train = pd.read_csv(path_to_data + "train_distribution.csv", index_col=0)
    
    # 3. Validierungsdaten laden
    Y_val = pd.read_csv(path_to_data + "test_data.csv", index_col=0)
    C_val = pd.read_csv(path_to_data + "test_distribution.csv", index_col=0)
    
    # 4. Zelltyp-Häufigkeiten berechnen
    celltype_counts_train = {}
    for celltype in X_train.columns.unique():
        celltype_counts_train[celltype] = C_train.sum(axis=1)[celltype]
    
    # 5. Hierarchie-Dictionary bereinigen
    for celltype in main_celltypes:
        sub_celltypes[celltype] = filter_subtypes_by_dataframe_columns(
            sub_celltypes[celltype], X_train)
    
    return {
        'X_train': X_train,
        'Y_train': Y_train,
        'C_train': C_train,
        'Y_val': Y_val,
        'C_val': C_val,
        'sub_celltypes': sub_celltypes,
        'celltype_counts_train': celltype_counts_train
    }

def train_and_validate_hide(data, iterations=1000):
    """
    Trainiere und validiere HIDE
    """
    print("=== HIDE Training und Validierung ===")
    
    # HIDE ausführen
    results = HIDE(
        C_train_all=data['C_train'],
        C_val_all=data['C_val'],
        Y_train_all=data['Y_train'],
        Y_val_all=data['Y_val'],
        X_ref_all=data['X_train'],
        subtypes_dict=data['sub_celltypes'],
        count_celltypes=data['celltype_counts_train'],
        iterations_dtd=iterations,
        savePath='./results/',
        saveC=True,
        saveGammaAndX=True
    )
    
    print(f"\n=== Trainingsergebnisse ===")
    print(f"Gesamtkorrelation Training: {results['corr_train']:.4f}")
    print(f"Gesamtkorrelation Validierung: {results['corr_val']:.4f}")
    
    return results

def apply_to_new_data(results, Y_application):
    """
    Wende trainierte HIDE-Parameter auf neue Daten an
    """
    print("=== Anwendung auf neue Daten ===")
    
    # 1. Hauptzelltypen schätzen
    print("-> Schätze Hauptzelltypen...")
    gamma_main = results['major']['model_main'].gamma
    X_main = results['major']['X_main']
    LinReg_main = results['major']['LinReg']
    
    # DTD-Schätzung
    C_main = calculate_estimated_composition(X_main, Y_application, gamma_main)
    
    # Lineare Regression anwenden
    C_main = adjustToLinReg(C_main, LinReg_main)
    
    # Negative Werte auf 0 setzen und normalisieren
    C_main = C_main.clip(lower=0)
    C_main = C_main / C_main.sum(axis=0)
    
    print(f"   Hauptzelltypen geschätzt: {list(C_main.index)}")
    
    # 2. Subtypen für jeden Hauptzelltyp schätzen
    subtype_predictions = {}
    
    for celltype in results['minor'].keys():
        print(f"-> Schätze {celltype} Subtypen...")
        
        # Parameter für diesen Zelltyp extrahieren
        gamma_sub = results['minor'][celltype]['model'].gamma
        X_sub = results['minor'][celltype]['X_sub']
        LinReg_sub = results['minor'][celltype]['LinReg']
        
        # Subtyp-Komposition schätzen
        result_sub = subtypes_estimate_composition(
            X_sub=X_sub,
            X_main=X_main,
            Y_all=Y_application,
            type_to_extend=celltype,
            C_main=C_main,
            gamma=gamma_sub,
            linReg=LinReg_sub
        )
        
        subtype_predictions[celltype] = result_sub['C_est']
        print(f"   {celltype} Subtypen: {list(result_sub['C_est'].index)}")
    
    # 3. Weitere Untertypen schätzen (falls vorhanden)
    sub_subtype_predictions = {}
    
    for subtype in results['sub'].keys():
        print(f"-> Schätze {subtype} Untertypen...")
        
        # Parameter für diesen Untertyp extrahieren
        gamma_subsub = results['sub'][subtype]['model'].gamma
        X_subsub = results['sub'][subtype]['X_sub']
        LinReg_subsub = results['sub'][subtype]['LinReg']
        
        # Parent-Zelltyp finden
        parent_celltype = None
        for ct, subtypes in results['minor'].items():
            if subtype in subtypes['C_train'].index:
                parent_celltype = ct
                break
        
        if parent_celltype:
            # Untertyp-Komposition schätzen
            result_subsub = subtypes_estimate_composition(
                X_sub=X_subsub,
                X_main=subtype_predictions[parent_celltype],
                Y_all=subtype_predictions[parent_celltype],
                type_to_extend=subtype,
                C_main=subtype_predictions[parent_celltype],
                gamma=gamma_subsub,
                linReg=LinReg_subsub
            )
            
            sub_subtype_predictions[subtype] = result_subsub['C_est']
            print(f"   {subtype} Untertypen: {list(result_subsub['C_est'].index)}")
    
    return {
        'major_celltypes': C_main,
        'minor_celltypes': subtype_predictions,
        'sub_celltypes': sub_subtype_predictions
    }

def save_predictions(predictions, output_path='./predictions/'):
    """
    Speichere Vorhersagen
    """
    print("=== Speichere Vorhersagen ===")
    
    # Hauptzelltypen speichern
    predictions['major_celltypes'].to_csv(output_path + 'major_celltypes_predictions.csv')
    print(f"Hauptzelltypen gespeichert in: {output_path}major_celltypes_predictions.csv")
    
    # Subtypen speichern
    for celltype, composition in predictions['minor_celltypes'].items():
        composition.to_csv(output_path + f'{celltype}_subtypes_predictions.csv')
        print(f"{celltype} Subtypen gespeichert in: {output_path}{celltype}_subtypes_predictions.csv")
    
    # Untertypen speichern
    for subtype, composition in predictions['sub_celltypes'].items():
        composition.to_csv(output_path + f'{subtype}_sub_subtypes_predictions.csv')
        print(f"{subtype} Untertypen gespeichert in: {output_path}{subtype}_sub_subtypes_predictions.csv")

def main():
    """
    Hauptfunktion - Vollständiger Workflow
    """
    print("####################################")
    print("### HIDE Vollständiger Workflow ###")
    print("####################################")
    
    # 1. Daten vorbereiten
    data = prepare_data()
    
    # 2. HIDE trainieren und validieren
    results = train_and_validate_hide(data, iterations=1000)
    
    # 3. Auf neue Daten anwenden (hier verwenden wir Validierungsdaten als Beispiel)
    # In der Praxis würden Sie hier Ihre eigenen neuen Bulk-Daten laden
    Y_application = data['Y_val'].iloc[:, 0:10]  # Beispiel: erste 10 Proben
    
    print(f"\n=== Anwendung auf {Y_application.shape[1]} neue Proben ===")
    predictions = apply_to_new_data(results, Y_application)
    
    # 4. Vorhersagen speichern
    save_predictions(predictions)
    
    # 5. Zusammenfassung
    print("\n=== Zusammenfassung ===")
    print(f"Trainingsergebnisse:")
    print(f"  - Trainingskorrelation: {results['corr_train']:.4f}")
    print(f"  - Validierungskorrelation: {results['corr_val']:.4f}")
    print(f"Vorhersagen für neue Daten:")
    print(f"  - Anzahl Proben: {Y_application.shape[1]}")
    print(f"  - Hauptzelltypen: {len(predictions['major_celltypes'].index)}")
    print(f"  - Subtypen: {sum(len(comp.index) for comp in predictions['minor_celltypes'].values())}")
    print(f"  - Untertypen: {sum(len(comp.index) for comp in predictions['sub_celltypes'].values())}")

if __name__ == '__main__':
    main()

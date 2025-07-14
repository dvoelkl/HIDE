#!/usr/bin/env python3
"""
Beispiel für die Verwendung der HIDEModel Klasse
Zeigt die vereinfachte Anwendung mit train() und predict() Methoden
"""

import pandas as pd
import numpy as np
from HIDE_class import HIDEModel, HIDE
from pipelines_dataloader import disco_read_metadata
from pipelines_utils import merge_celltypes, filter_subtypes_by_dataframe_columns

def demonstrate_hide_class():
    """
    Demonstriert die Verwendung der HIDEModel Klasse
    """
    
    print("="*60)
    print("    HIDEModel Klasse - Vereinfachte Anwendung")
    print("="*60)
    
    # 1. Daten laden und vorbereiten
    print("\n1. Lade Daten...")
    
    data = load_hierarchy_and_data()
    
    print(f"   Geladene Daten:")
    print(f"   - Trainingsdaten: {data['Y_train'].shape[1]} Proben, {data['Y_train'].shape[0]} Gene")
    print(f"   - Validierungsdaten: {data['Y_val'].shape[1]} Proben")
    print(f"   - Referenzmatrix: {data['X_train'].shape[1]} Zelltypen")
    
    # 2. HIDEModel erstellen
    print("\n2. Erstelle HIDEModel...")
    
    model = HIDEModel(
        subtypes_dict=data['sub_celltypes'],
        count_celltypes=data['celltype_counts_train'],
        iterations_dtd=100,  # Reduziert für Demo
        save_path="./results/",
        save_models=True,
        save_compositions=True,
        verbose=True
    )
    
    print(f"   Modell erstellt mit {len(data['sub_celltypes'])} Hauptzelltypen")
    
    # 3. Modell trainieren
    print("\n3. Trainiere Modell...")
    
    model.train(
        C_train=data['C_train'],
        C_val=data['C_val'],
        Y_train=data['Y_train'],
        Y_val=data['Y_val'],
        X_ref=data['X_train']
    )
    
    # 4. Trainingsresultate anzeigen
    print("\n4. Trainingsresultate:")
    summary = model.get_training_summary()
    print(f"   - Trainingskorrelation: {summary['training_correlation']:.4f}")
    print(f"   - Validierungskorrelation: {summary['validation_correlation']:.4f}")
    print(f"   - Hauptzelltypen: {len(summary['major_celltypes'])}")
    print(f"   - Minor Zelltypen: {sum(len(v) for v in summary['minor_celltypes'].values())}")
    print(f"   - Sub-Zelltypen: {sum(len(v) for v in summary['sub_celltypes'].values())}")
    
    # 5. Vorhersagen für neue Daten
    print("\n5. Erstelle Vorhersagen für neue Daten...")
    
    # Verwende einen Teil der Validierungsdaten als "neue" Daten
    Y_new = data['Y_val'].iloc[:, 0:5]  # Erste 5 Proben
    
    predictions = model.predict(Y_new)
    
    print(f"   Vorhersagen für {Y_new.shape[1]} Proben erstellt")
    
    # 6. Vorhersagen analysieren
    print("\n6. Analyse der Vorhersagen:")
    
    # Hauptzelltypen
    major_pred = predictions['major']
    print(f"   Hauptzelltypen (erste 3 Proben):")
    print(major_pred.iloc[:, 0:3].round(4))
    
    # Ein Beispiel für Subtypen
    first_minor_type = list(predictions['minor'].keys())[0]
    minor_pred = predictions['minor'][first_minor_type]
    print(f"\n   {first_minor_type} Subtypen (erste 3 Proben):")
    print(minor_pred.iloc[:, 0:3].round(4))
    
    # 7. Modell speichern
    print("\n7. Speichere Modell...")
    model.save_model("./results/hide_model.pkl")
    
    # 8. Modell laden (Demonstration)
    print("\n8. Lade Modell (Demonstration)...")
    loaded_model = HIDEModel.load_model("./results/hide_model.pkl")
    
    # Teste geladenes Modell
    test_predictions = loaded_model.predict(Y_new.iloc[:, 0:2])
    print(f"   Geladenes Modell funktioniert: {test_predictions['major'].shape[1]} Vorhersagen")
    
    return model, predictions

def compare_with_original():
    """
    Vergleicht die Ergebnisse der HIDEModel Klasse mit der ursprünglichen HIDE Funktion
    """
    
    print("\n" + "="*60)
    print("    Vergleich: HIDEModel vs. Original HIDE")
    print("="*60)
    
    # Lade die gleichen Daten
    data = load_hierarchy_and_data()
    
    # Test mit HIDEModel
    print("\n-> Teste HIDEModel Klasse...")
    model = HIDEModel(
        subtypes_dict=data['sub_celltypes'],
        count_celltypes=data['celltype_counts_train'],
        iterations_dtd=50,  # Reduziert für schnelleren Vergleich
        verbose=False
    )
    
    model.train(data['C_train'], data['C_val'], data['Y_train'], data['Y_val'], data['X_train'])
    
    # Vorhersage für eine kleine Probe
    Y_test = data['Y_val'].iloc[:, 0:3]
    predictions_class = model.predict(Y_test)
    
    print(f"   HIDEModel Korrelation: {model.training_results['corr_val']:.4f}")
    print(f"   Vorhersagen erstellt für {Y_test.shape[1]} Proben")
    
    # Test mit Original HIDE
    print("\n-> Teste Original HIDE...")
    
    results_original = HIDE(
        C_train_all=data['C_train'],
        C_val_all=data['C_val'],
        Y_train_all=data['Y_train'],
        Y_val_all=data['Y_val'],
        X_ref_all=data['X_train'],
        subtypes_dict=data['sub_celltypes'],
        count_celltypes=data['celltype_counts_train'],
        iterations_dtd=50,
        savePath=None,
        saveC=False,
        saveGammaAndX=False
    )
    
    print(f"   Original HIDE Korrelation: {results_original['corr_val']:.4f}")
    
    # Vergleiche Ergebnisse
    print("\n-> Vergleiche Ergebnisse:")
    correlation_diff = abs(model.training_results['corr_val'] - results_original['corr_val'])
    print(f"   Korrelationsunterschied: {correlation_diff:.6f}")
    
    if correlation_diff < 0.001:
        print("   ✓ Ergebnisse sind praktisch identisch!")
    else:
        print(f"   ⚠ Ergebnisse unterscheiden sich um {correlation_diff:.6f}")

def usage_example():
    """
    Einfaches Anwendungsbeispiel
    """
    
    print("\n" + "="*60)
    print("    Einfaches Anwendungsbeispiel")
    print("="*60)
    
    # Daten laden (vollständig)
    data = load_hierarchy_and_data()
    
    # Modell erstellen und trainieren
    model = HIDEModel(
        data['sub_celltypes'], 
        data['celltype_counts_train'], 
        iterations_dtd=100, 
        verbose=False
    )
    model.train(data['C_train'], data['C_val'], data['Y_train'], data['Y_val'], data['X_train'])
    
    # Vorhersagen für neue Daten
    Y_new = data['Y_val'].iloc[:, 0:5]
    predictions = model.predict(Y_new)
    
    # Ergebnisse anzeigen
    print("\nHauptzelltypen (erste 2 Proben):")
    print(predictions['major'].iloc[:, 0:2])
    
    # Modell speichern
    model.save_model("./results/simple_model.pkl")
    
    print("\nModell gespeichert und einsatzbereit!")

def simple_hide_example():
    """
    Sehr einfaches Beispiel für die Verwendung von HIDEModel
    """
    print("\n" + "="*60)
    print("    Einfachstes HIDEModel Beispiel")
    print("="*60)
    
    try:
        # Überprüfe und erstelle results-Verzeichnis
        import os
        if not os.path.exists("./results/"):
            os.makedirs("./results/")
            print("-> Results-Verzeichnis erstellt")
        
        # Daten laden
        print("-> Lade Daten...")
        data = load_hierarchy_and_data()
        print(f"   ✓ Daten geladen: {data['Y_train'].shape[0]} Gene, {data['Y_train'].shape[1]} Trainingsproben")
        
        # Modell erstellen
        print("-> Erstelle HIDEModel...")
        model = HIDEModel(
            data['sub_celltypes'], 
            data['celltype_counts_train'], 
            iterations_dtd=25, 
            verbose=False
        )
        print("   ✓ Modell erstellt")
        
        # Training
        print("-> Trainiere Modell...")
        model.train(data['C_train'], data['C_val'], data['Y_train'], data['Y_val'], data['X_train'])
        print(f"   ✓ Training abgeschlossen (Korrelation: {model.training_results['corr_val']:.4f})")
        
        # Vorhersagen
        print("-> Erstelle Vorhersagen...")
        predictions = model.predict(data['Y_val'].iloc[:, 0:3])
        print(f"   ✓ Vorhersagen für {predictions['major'].shape[1]} Proben erstellt")
        
        print(f"\n✓ Erfolgreich! HIDEModel funktioniert korrekt")
        print(f"✓ Modell-Performance: {model.training_results['corr_val']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Fehler beim Ausführen des Beispiels: {e}")
        import traceback
        traceback.print_exc()
        return False

def load_hierarchy_and_data(path_to_data="./data/"):
    """
    Hilfsfunktion zum korrekten Laden der Hierarchie und Daten.
    
    Returns
    -------
    dict : Dictionary mit allen benötigten Daten
    """
    
    # Hierarchie vollständig laden
    meta = disco_read_metadata(path_to_data + 'cell_hierarchy.csv', 
                              "celltype_major", 'celltype_minor')
    main_celltypes = meta['main_celltypes']
    sub_celltypes = meta['sub_celltypes']
    
    # Erweiterte Hierarchie für Untertypen
    meta_sub = disco_read_metadata(path_to_data + 'cell_hierarchy.csv', 
                                  "celltype_minor", 'celltype_sub')
    subset_celltypes = meta_sub['sub_celltypes']
    sub_celltypes = merge_celltypes(sub_celltypes, subset_celltypes)
    
    # Daten laden
    X_train = pd.read_csv(path_to_data + "X_train.csv", index_col=0)
    Y_train = pd.read_csv(path_to_data + "train_data.csv", index_col=0)
    C_train = pd.read_csv(path_to_data + "train_distribution.csv", index_col=0)
    Y_val = pd.read_csv(path_to_data + "test_data.csv", index_col=0)
    C_val = pd.read_csv(path_to_data + "test_distribution.csv", index_col=0)
    
    # Zelltyp-Häufigkeiten
    celltype_counts_train = {}
    for celltype in X_train.columns.unique():
        celltype_counts_train[celltype] = C_train.sum(axis=1)[celltype]
    
    # Hierarchie bereinigen
    for celltype in main_celltypes:
        sub_celltypes[celltype] = filter_subtypes_by_dataframe_columns(
            sub_celltypes[celltype], X_train)
    
    return {
        'sub_celltypes': sub_celltypes,
        'main_celltypes': main_celltypes,
        'X_train': X_train,
        'Y_train': Y_train,
        'C_train': C_train,
        'Y_val': Y_val,
        'C_val': C_val,
        'celltype_counts_train': celltype_counts_train
    }

if __name__ == '__main__':
    
    # Erst einfaches Beispiel testen
    print("Teste einfaches Beispiel...")
    success = simple_hide_example()
    
    if success:
        print("\n" + "="*60)
        print("    Führe vollständige Demonstrationen aus...")
        print("="*60)
        
        # Hauptdemonstration
        model, predictions = demonstrate_hide_class()
        
        # Vergleich mit Original
        compare_with_original()
        
        # Einfaches Beispiel
        usage_example()
        
        print("\n" + "="*60)
        print("    Alle Demonstrationen abgeschlossen!")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("    Fehler beim einfachen Beispiel - Demo abgebrochen")
        print("="*60)

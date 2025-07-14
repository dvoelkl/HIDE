# HIDEModel Klasse - Vereinfachte Anwendung

Die `HIDEModel` Klasse bietet eine benutzerfreundliche, scikit-learn-ähnliche Schnittstelle für den HIDE-Algorithmus.

## Hauptmerkmale

- **Einfache API**: Separate `train()` und `predict()` Methoden
- **Modell-Persistierung**: Speichern und Laden von trainierten Modellen
- **Hierarchische Vorhersagen**: Automatische Vorhersage aller Hierarchieebenen
- **Trainings-Übersicht**: Detaillierte Informationen über das Training

## Schnellstart

```python
from HIDE_class import HIDEModel

# 1. Modell erstellen
model = HIDEModel(
    subtypes_dict=your_hierarchy_dict,
    count_celltypes=your_celltype_counts,
    iterations_dtd=1000
)

# 2. Trainieren
model.train(C_train, C_val, Y_train, Y_val, X_ref)

# 3. Vorhersagen für neue Daten
predictions = model.predict(Y_new)

# 4. Modell speichern
model.save_model("my_model.pkl")

# 5. Modell laden
loaded_model = HIDEModel.load_model("my_model.pkl")
```

## Detaillierte Anwendung

### 1. Modell-Initialisierung

```python
model = HIDEModel(
    subtypes_dict=subtypes_dict,        # Hierarchie-Dictionary
    count_celltypes=celltype_counts,    # Zelltyp-Häufigkeiten
    iterations_dtd=1000,                # DTD-Iterationen
    save_path="./results/",             # Speicherpfad für Ergebnisse
    save_models=True,                   # Modellparameter speichern
    save_compositions=True,             # Geschätzte Kompositionen speichern
    verbose=True                        # Fortschrittsmeldungen
)
```

### 2. Training

```python
model.train(
    C_train=C_train,    # Trainings-Komposition (Zelltypen x Proben)
    C_val=C_val,        # Validierungs-Komposition
    Y_train=Y_train,    # Trainings-Bulk-Daten (Gene x Proben)
    Y_val=Y_val,        # Validierungs-Bulk-Daten
    X_ref=X_ref         # Referenz-Expressionsmatrix (Gene x Zelltypen)
)
```

### 3. Vorhersagen

```python
predictions = model.predict(Y_new)

# Ergebnisse:
# predictions['major']           -> Hauptzelltypen
# predictions['minor'][celltype] -> Subtypen für jeden Hauptzelltyp
# predictions['sub'][subtype]    -> Sub-Subtypen
```

### 4. Trainings-Übersicht

```python
summary = model.get_training_summary()
print(f"Training correlation: {summary['training_correlation']:.4f}")
print(f"Validation correlation: {summary['validation_correlation']:.4f}")
print(f"Major cell types: {summary['major_celltypes']}")
```

### 5. Modell-Persistierung

```python
# Speichern
model.save_model("trained_model.pkl")

# Laden
loaded_model = HIDEModel.load_model("trained_model.pkl")
new_predictions = loaded_model.predict(Y_new)
```

## Datenstrukturen

### Eingabe-Datenformate

- **C_train/C_val**: Kompositionsmatrizen (Zelltypen als Zeilen, Proben als Spalten)
- **Y_train/Y_val/Y_new**: Bulk-Expressionsdaten (Gene als Zeilen, Proben als Spalten)
- **X_ref**: Referenzmatrix (Gene als Zeilen, Zelltypen als Spalten)

### Hierarchie-Dictionary

```python
subtypes_dict = {
    'T cell': {
        'CD4 T cell': ['CD4 memory T', 'CD4 naive T'],
        'CD8 T cell': ['CD8 memory T', 'CD8 naive T']
    },
    'B cell': {
        'B cell memory': ['B cell memory 1', 'B cell memory 2'],
        'B cell naive': ['B cell naive 1']
    }
}
```

### Zelltyp-Häufigkeiten

```python
celltype_counts = {
    'CD4 memory T': 1500,
    'CD4 naive T': 800,
    'CD8 memory T': 1200,
    # ... für alle Zelltypen
}
```

## Vorteile gegenüber der Original-HIDE Funktion

1. **Einfachere Anwendung**: Klare Trennung zwischen Training und Vorhersage
2. **Wiederverwendbarkeit**: Einmal trainiert, mehrfach anwendbar
3. **Modell-Persistierung**: Speichern und Laden von Modellen
4. **Bessere Struktur**: Organisierte Ausgabe-Struktur
5. **Fehlerbehandlung**: Bessere Validierung und Fehlermeldungen

## Beispiel-Workflow

```python
# Daten laden
X_train = pd.read_csv("X_train.csv", index_col=0)
Y_train = pd.read_csv("train_data.csv", index_col=0)
C_train = pd.read_csv("train_distribution.csv", index_col=0)
Y_val = pd.read_csv("test_data.csv", index_col=0)
C_val = pd.read_csv("test_distribution.csv", index_col=0)

# Hierarchie und Häufigkeiten definieren
subtypes_dict = load_hierarchy()
celltype_counts = calculate_counts(C_train)

# Modell erstellen und trainieren
model = HIDEModel(subtypes_dict, celltype_counts, iterations_dtd=1000)
model.train(C_train, C_val, Y_train, Y_val, X_train)

# Für neue Daten vorhersagen
Y_new = load_new_data()
predictions = model.predict(Y_new)

# Ergebnisse analysieren
major_types = predictions['major']
t_cell_subtypes = predictions['minor']['T cell']
```

Die `HIDEModel` Klasse macht die Anwendung von HIDE deutlich benutzerfreundlicher und strukturierter!

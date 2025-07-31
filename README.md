# Traffic Prediction with Graph Neural Networks

Ein Projekt zur Vorhersage von Verkehrsflüssen mit Graph Neural Networks (GNNs) auf den Datensätzen **METR‑LA** und **PEMS‑BAY**.

## Überblick

Implementierte und verglichene GNN‑Architekturen für räumlich‑zeitliche Verkehrsprognosen:

* **STGCN** – Spatial‑Temporal Graph Convolution Network
* **GraphWaveNet** – Dilated Temporal Convolution + Graph Convolution
* **GraphAttentionNetwork** – Multi‑Head Graph Attention
* **GraphTransformer** – TransformerConv‑Layer

### Merkmale

* Zwei Arbeitsweisen: Jupyter‑Notebook oder Kommandozeilen‑Skript
* Analyse mehrerer Datensätze
* Cross‑Dataset‑Evaluation
* Interaktive Plotly‑Dashboards
* Vollständiger CSV‑Export
* Automatisches Management aller Ergebnisse

## Quick Start

### Installation

```bash
# Repository klonen
git clone https://github.com/ChipCracker/GSP.git
cd GSP

# Virtuelle Umgebung anlegen
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# oder
.venv\Scripts\activate    # Windows

# Abhängigkeiten installieren (umfangreiche ML/AI-Bibliotheken)
pip install -r requirements.txt

# Für minimale Installation nur für GNN-Traffic-Prediction:
# pip install torch torch-geometric pandas numpy matplotlib plotly seaborn scikit-learn jupyter
```
### Environment‑Setup

```bash
# .env für das Ausgabe‑Verzeichnis anlegen
echo "OUT_DIR=/pfad/zum/output" > .env
```

### Datensätze

```
data/
├── METR-LA.csv              # Verkehrsdaten METR‑LA
├── PEMS-BAY.csv             # Verkehrsdaten PEMS‑BAY (optional)
├── adj_mx_METR-LA.pkl       # Adjazenzmatrix METR‑LA
└── adj_mx_PEMS-BAY.pkl      # Adjazenzmatrix PEMS‑BAY
```

## Workflows

### Notebook‑basierter Workflow

```bash
jupyter lab traffic_data_analysis.ipynb       # Datenanalyse
jupyter lab traffic_gnn_evaluation.ipynb      # Model-Evaluation und Visualisierung
jupyter lab traffic_graph_ablation_analysis.ipynb  # Graph-Konstruktions-Ablationsstudie
```

### Script‑basierter Workflow

```bash
# Haupttraining
python traffic_gnn_training.py --experiment single          # Einzelnes Modell
python traffic_gnn_training.py --experiment comparison      # Architekturvergleich
python traffic_gnn_training.py --experiment hyperparameter  # Hyperparameter‑Tuning
python traffic_gnn_training.py --experiment multi_dataset   # Multi‑Datensatz‑Analyse
python traffic_gnn_training.py --experiment all             # Alle Experimente

# Graph-Ablationsstudien
python traffic_graph_ablation_training.py                   # Graph-Konstruktions-Experimente
```

### Graph‑Ablationsstudien

* Systematischer Vergleich verschiedener Graph‑Konstruktionsmethoden
* k‑NN vs. korrelationsbasierte vs. vorberechnete Adjazenzmatrizen
* Evaluation der Graph‑Topologie‑Auswirkungen auf Model‑Performance
* Detaillierte Analyse mit `traffic_graph_ablation_analysis.ipynb`
* Automatisiertes Training mit `traffic_graph_ablation_training.py`

### Automatisches Ergebnis‑Management

```
{OUT_DIR}/traffic_gnn_results/
├── traffic_gnn_experiment_YYYYMMDD_HHMMSS/
│   ├── models/                    # Trainierte Modelle
│   ├── plots/                     # Visualisierungen
│   ├── csv/                       # CSV‑Exports
│   │   ├── training_history/
│   │   ├── predictions/
│   │   ├── metrics/
│   │   └── analysis/
│   ├── data/                      # Dataset‑Info
│   └── logs/                      # Training‑Logs
```


## Evaluation

### Model‑Evaluation (`traffic_gnn_evaluation.ipynb`)

* Automatisches Laden trainierter Modelle
* Visualisierung des Trainingsverlaufs
* Fehler‑Analyse der Vorhersagen
* Cross‑Dataset‑Performance‑Heatmaps
* Bewertung der Architektur‑Robustheit

### Graph‑Ablationsanalyse (`traffic_graph_ablation_analysis.ipynb`)

* Vergleichende Analyse verschiedener Graph‑Konstruktionsmethoden
* Visualisierung der Graph‑Topologie‑Auswirkungen
* Performance‑Metriken nach Graph‑Typ
* Interaktive Dashboards für Graph‑Vergleiche

## Konfiguration

### Environment‑Variablen

```bash
# .env
OUT_DIR=/pfad/zum/output       # Ausgabe‑Verzeichnis
```

### Modell‑Hyperparameter

```python
SEQUENCE_LENGTH = 12      # Eingabezeitpunkte
PREDICTION_LENGTH = 3     # Vorhersagezeitpunkte
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EARLY_STOPPING = 15
```
---

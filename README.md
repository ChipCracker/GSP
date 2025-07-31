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
git clone <repository-url>
cd traffic-prediction-gnn

# Virtuelle Umgebung anlegen
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# oder
.venv\Scripts\activate    # Windows

# Abhängigkeiten installieren
pip install -r requirements.txt
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
jupyter lab traffic_pred_gnn.ipynb            # Training und Experimente
jupyter lab traffic_data_analysis.ipynb       # Datenanalyse
jupyter lab traffic_gnn_evaluation.ipynb      # Evaluation und Visualisierung
```

### Script‑basierter Workflow

```bash
python traffic_gnn_training.py --experiment single          # Einzelnes Modell
python traffic_gnn_training.py --experiment comparison      # Architekturvergleich
python traffic_gnn_training.py --experiment hyperparameter  # Hyperparameter‑Tuning
python traffic_gnn_training.py --experiment multi_dataset   # Multi‑Datensatz‑Analyse
python traffic_gnn_training.py --experiment all             # Alle Experimente
```

## Experimentelle Funktionen

### Multi‑Dataset‑Comparison

* Performance‑Vergleich METR‑LA vs PEMS‑BAY
* Bewertung der Architektur‑Robustheit
* Transfer‑Learning‑Analyse
* Interaktive Heatmaps zur Ergebnisdarstellung

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

## Modell‑Performance (METR‑LA)

| Modell           | Test Loss | Test MAE | Trainingszeit |
| ---------------- | --------- | -------- | ------------- |
| GraphAttention   | 0.1121    | 0.2674   | ≈ 15 min      |
| STGCN            | 0.1124    | 0.2684   | ≈ 12 min      |
| GraphTransformer | 0.1127    | 0.2686   | ≈ 18 min      |
| GraphWaveNet     | 0.1135    | 0.2695   | ≈ 20 min      |

## Zentrale Erkenntnisse

* GraphAttention liefert die beste Gesamt‑Performance.
* k‑NN‑Graph‑Konstruktion (k = 8) übertrifft korrelationsbasierte Ansätze.
* Die Datensätze unterscheiden sich im Schwierigkeitsgrad.
* Transfer‑Learning zwischen METR‑LA und PEMS‑BAY zeigt nur begrenzte Generalisierung.

## Technische Details

### Architektur‑Schnittstelle

```python
def forward(x, edge_index, edge_weight):
    """Vorwärtsdurchlauf
    Args:
        x: Eingabetensor [batch, seq_len, nodes]
        edge_index: Kantenindizes
        edge_weight: Kantengewichte
    Returns:
        Tensor [batch, pred_len, nodes]
    """
```

### Daten‑Pipeline

1. **TrafficDataLoader** – CSV‑Ladung mit Zeitzonen‑Handling
2. **TemporalDataProcessor** – Sliding‑Window‑Sequenzen (12 → 3 Schritte)
3. **TrafficGraphBuilder** – Graph‑Konstruktion
4. **TrafficGNNTrainer** – Einheitliches Training mit Early Stopping

#### Unterstützte Graph‑Konstruktionen

* Vorberechnete Adjazenzmatrizen
* Korrelationsbasierte Schwellenwerte
* k‑Nearest‑Neighbors (empfohlen: k = 8)

## Evaluation

Das Notebook `traffic_gnn_evaluation.ipynb` bietet:

* Automatisches Laden trainierter Modelle
* Visualisierung des Trainingsverlaufs
* Fehler‑Analyse der Vorhersagen
* Cross‑Dataset‑Performance‑Heatmaps
* Bewertung der Architektur‑Robustheit

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

## Fehlersuche

1. **CUDA "out of memory"**

   ```bash
   python traffic_gnn_training.py --experiment single --batch_size 16
   ```
2. **Fehlende Datensätze**

   * Prüfen, ob die CSV‑Dateien in `data/` vorhanden sind.
   * Bei Bedarf auf synthetische Daten zurückgreifen.
3. **Encoding‑Probleme beim CSV‑Import**

   * Mehrere Encodings werden automatisch ausprobiert.
   * Bei fortbestehenden Fehlern die CSVs in UTF‑8 konvertieren.

---

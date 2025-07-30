# Traffic Prediction with Graph Neural Networks

[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Ein umfassendes Research-Projekt für die Vorhersage von Verkehrsflüssen mit Graph Neural Networks (GNNs) auf den METR-LA und PEMS-BAY Datensätzen.

## 🎯 Überblick

Dieses Projekt implementiert und vergleicht mehrere GNN-Architekturen für räumlich-zeitliche Verkehrsprognosen:

- **STGCN** (Spatial-Temporal Graph Convolution Network)
- **GraphWaveNet** (Dilated TCN + GCN)
- **GraphAttentionNetwork** (Multi-head GAT)
- **GraphTransformer** (TransformerConv layers)

### Kernfeatures

- ✅ **Dual Workflow**: Sowohl Jupyter Notebook als auch CLI-Script Support
- ✅ **Multi-Dataset Analysis**: Vergleich zwischen METR-LA und PEMS-BAY
- ✅ **Cross-Dataset Evaluation**: Robustheitsbewertung von Modellarchitekturen
- ✅ **Interactive Dashboards**: Umfassende Plotly-Visualisierungen
- ✅ **Comprehensive CSV Export**: Detaillierte Ergebnisexporte für weitere Analyse
- ✅ **Automated Result Management**: Organisierte Speicherung aller Experimente

## 🚀 Quick Start

### Installation

```bash
# Repository klonen
git clone <repository-url>
cd traffic-prediction-gnn

# Python Virtual Environment erstellen
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# oder
.venv\Scripts\activate     # Windows

# Dependencies installieren
pip install -r requirements.txt
```

### Environment Setup

```bash
# .env Datei erstellen für Output-Verzeichnis
echo "OUT_DIR=/path/to/your/output/directory" > .env
```

### Datasets

Platziere die Datensätze im `data/` Verzeichnis:
```
data/
├── METR-LA.csv              # METR-LA Verkehrsdaten
├── PEMS-BAY.csv             # PEMS-BAY Verkehrsdaten (optional)
├── adj_mx_METR-LA.pkl       # Vorberechnete Adjazenzmatrix für METR-LA
└── adj_mx_PEMS-BAY.pkl      # Vorberechnete Adjazenzmatrix für PEMS-BAY
```

## 📊 Workflows

### 1. Notebook-basierter Workflow (Interaktiv)

#### Training und Experimente
```bash
jupyter lab traffic_pred_gnn.ipynb
```

#### Datenanalyse
```bash
jupyter lab traffic_data_analysis.ipynb
```

#### Evaluation und Visualisierung
```bash
jupyter lab traffic_gnn_evaluation.ipynb
```

### 2. Script-basierter Workflow (Automatisiert)

#### Single Model Training
```bash
python traffic_gnn_training.py --experiment single
```

#### Model Architecture Comparison
```bash
python traffic_gnn_training.py --experiment comparison
```

#### Hyperparameter Tuning
```bash
python traffic_gnn_training.py --experiment hyperparameter
```

#### Multi-Dataset Analysis
```bash
python traffic_gnn_training.py --experiment multi_dataset
```

#### All Experiments
```bash
python traffic_gnn_training.py --experiment all
```

## 🔬 Experimentelle Features

### Multi-Dataset Comparison

Das Projekt bietet erweiterte Cross-Dataset-Analyse-Capabilities:

- **Performance Comparison**: METR-LA vs PEMS-BAY Modellvergleich
- **Architecture Robustness**: Bewertung der Konsistenz verschiedener GNN-Architekturen
- **Transfer Learning Insights**: Analyse der Generalisierungsfähigkeiten
- **Interactive Heatmaps**: Visualisierung der Cross-Dataset-Performance

### Automated Result Management

Alle Experimente werden automatisch organisiert:

```
{OUT_DIR}/traffic_gnn_results/
├── traffic_gnn_experiment_YYYYMMDD_HHMMSS/
│   ├── models/                    # Trainierte Modelle
│   ├── plots/                     # Visualisierungen
│   ├── csv/                       # Detaillierte CSV-Exports
│   │   ├── training_history/      # Training-Verläufe
│   │   ├── predictions/           # Modell-Vorhersagen
│   │   ├── metrics/               # Performance-Metriken
│   │   └── analysis/              # Datenanalysen
│   ├── data/                      # Dataset-Informationen
│   └── logs/                      # Training-Logs
```

## 📈 Model Performance

### Benchmark Results (METR-LA)

| Modell | Test Loss | Test MAE | Training Zeit |
|--------|-----------|----------|---------------|
| **GraphAttention** | 0.1121 | 0.2674 | ~15 min |
| STGCN | 0.1124 | 0.2684 | ~12 min |
| GraphTransformer | 0.1127 | 0.2686 | ~18 min |
| GraphWaveNet | 0.1135 | 0.2695 | ~20 min |

### Key Findings

- **GraphAttention** zeigt die beste Overall-Performance
- **k-NN Graph Construction** (k=8) übertrifft korrelationsbasierte Methoden
- **Multi-Dataset Analysis** zeigt unterschiedliche Schwierigkeitsgrade der Datensätze
- **Transfer Learning** zwischen METR-LA und PEMS-BAY zeigt begrenzte Generalisierung

## 🛠️ Technische Details

### Architektur

```python
# Einheitliche Model Interface
def forward(x, edge_index, edge_weight):
    # Input:  [batch, seq_len, nodes]
    # Output: [batch, pred_len, nodes]
```

### Data Pipeline

1. **TrafficDataLoader**: Robuste CSV-Ladung mit Timezone-Handling
2. **TemporalDataProcessor**: Sliding Window Sequenzen (12 → 3 Schritte)
3. **TrafficGraphBuilder**: Priority-basierte Graph-Konstruktion
4. **TrafficGNNTrainer**: Einheitliches Training mit Early Stopping

### Supported Graph Construction Methods

- **Precomputed Adjacency**: Vorberechnete Adjazenzmatrizen
- **Correlation-based**: Korrelations-threshold basierte Kanten
- **k-NN Graph**: k-Nearest Neighbors (empfohlen: k=8)

## 📋 Evaluation Features

### Comprehensive Analysis

Das `traffic_gnn_evaluation.ipynb` Notebook bietet:

- **Automatic Model Loading**: Erkennt und lädt alle trainierten Modelle
- **Training History Visualization**: Interaktive Plotly-Dashboards
- **Prediction Analysis**: Detaillierte Error-Analyse
- **Multi-Dataset Comparison**: Cross-Dataset Performance Heatmaps
- **Architecture Robustness**: Konsistenz-Bewertung über Datensätze
- **CSV Data Analysis**: Robuste Datenladung mit encoding-fallbacks

### Interactive Dashboards

- Performance Ranking und Vergleich
- Training Progress Monitoring
- Error Distribution Analysis
- Cross-Dataset Transfer Learning Insights

## 🔧 Konfiguration

### Environment Variables

```bash
# .env file
OUT_DIR=/path/to/output/directory  # Output-Verzeichnis für alle Ergebnisse
```

### Model Hyperparameters

```python
# Standard Configuration
SEQUENCE_LENGTH = 12      # Input Zeitschritte (1 Stunde)
PREDICTION_LENGTH = 3     # Vorhersage Zeitschritte (15 Minuten)
BATCH_SIZE = 32          # Batch-Größe
LEARNING_RATE = 0.001    # Lernrate
EARLY_STOPPING = 15      # Early Stopping Patience
```

## 📚 Documentation

- **CLAUDE.md**: Detaillierte Entwicklerdokumentation für Claude Code
- **Notebooks**: Ausführliche Kommentare und Markdown-Zellen
- **Code Comments**: Umfassende Inline-Dokumentation

## 🐛 Troubleshooting

### Häufige Probleme

1. **CUDA Out of Memory**
   ```bash
   # Reduziere Batch Size
   python traffic_gnn_training.py --experiment single --batch_size 16
   ```

2. **Missing Dataset Files**
   - Stelle sicher, dass CSV-Dateien im `data/` Verzeichnis sind
   - Das System fällt automatisch auf synthetische Daten zurück

3. **Encoding Errors beim CSV-Loading**
   - Das System versucht automatisch mehrere Encodings
   - Bei anhaltenden Problemen, konvertiere CSV zu UTF-8

### Performance Optimization

- **GPU Usage**: CUDA wird automatisch erkannt und verwendet
- **Memory Management**: Effiziente Batching für große Datensätze
- **Parallel Processing**: Multi-threading für Datenladung

## 📄 License

Dieses Projekt ist unter der MIT License lizensiert - siehe [LICENSE](LICENSE) Datei für Details.

## 🏆 Acknowledgments

- **METR-LA Dataset**: Los Angeles Metropolitan Transportation Authority
- **PEMS-BAY Dataset**: Caltrans Performance Measurement System
- **PyTorch Geometric**: Für GNN-Implementierungen
- **Plotly**: Für interaktive Visualisierungen

---

**🚀 Happy Traffic Predicting!** 🚦📊
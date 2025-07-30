#!/usr/bin/env python3
"""
Traffic GNN Training Script - Complete Implementation
=====================================

Comprehensive script for training Graph Neural Networks on traffic data.
Supports multiple architectures, extensive result management, and CSV exports.

Usage:
    python traffic_gnn_training.py --experiment single
    python traffic_gnn_training.py --experiment comparison
    python traffic_gnn_training.py --experiment hyperparameter
    python traffic_gnn_training.py --experiment all

Author: Generated from traffic_pred_gnn.ipynb
"""

import os
import json
import pickle
import logging
import argparse
import warnings
from datetime import datetime
from itertools import product
import random

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # For headless plotting
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch_geometric
from torch_geometric.nn import GCNConv, ChebConv, GATConv, TransformerConv
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_networkx

from tqdm import tqdm
from dotenv import load_dotenv

# Suppress warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# =============================================================================
# CONFIGURATION AND SETUP
# =============================================================================

def setup_environment():
    """Setup output directory and basic configuration"""
    OUT_DIR = os.getenv('OUT_DIR', '.')
    
    # Create base directories
    os.makedirs(OUT_DIR, exist_ok=True)
    
    print(f"🔧 Environment Setup:")
    print(f"   Output Directory: {OUT_DIR}")
    print(f"   PyTorch Version: {torch.__version__}")
    print(f"   PyTorch Geometric Version: {torch_geometric.__version__}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Device: {device}")
    
    return OUT_DIR, device

# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

class TrafficDataLoader:
    """Enhanced data loader for METR-LA traffic dataset with timezone handling"""

    def __init__(self, file_path=None, timezone='US/Pacific'):
        self.data = None
        self.sensor_ids = None
        self.timestamps = None
        self.scaler = StandardScaler()
        self.timezone = timezone

    def load_csv_data(self, file_path):
        """Load traffic data from CSV and localize timestamps"""
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        # Zeitzone zuweisen und DST-Lücken verschieben
        df.index = df.index.tz_localize(
            self.timezone,
            ambiguous='infer',
            nonexistent='shift_forward'
        )
        df.columns = df.columns.astype(str)
        df = df.loc[:, (df != 0).any(axis=0)]
        df = df.dropna(axis=1, how='all')

        self.data = df
        self.sensor_ids = list(df.columns)
        self.timestamps = df.index
        return df

    def create_sample_data(self, n_sensors=50, n_timesteps=2000):
        """Create sample data if CSV not available"""
        timestamps = pd.date_range(
            '2012-03-01',
            periods=n_timesteps,
            freq='5min',
            tz=self.timezone
        )
        sensor_ids = [f"sensor_{i:03d}" for i in range(n_sensors)]
        base = np.sin(np.linspace(0, 8*np.pi, n_timesteps)) * 20 + 50
        data = []
        for _ in sensor_ids:
            noise = np.random.normal(0, 5, n_timesteps)
            pattern = np.maximum(base + noise + np.random.normal(0,10), 0)
            data.append(pattern)
        df = pd.DataFrame(np.array(data).T, index=timestamps, columns=sensor_ids)

        self.data = df
        self.sensor_ids = sensor_ids
        self.timestamps = timestamps
        return df

class TrafficGraphBuilder:
    """Build spatial graph for traffic sensors"""
    
    def __init__(self, method='distance', threshold=0.7):
        self.method = method
        self.threshold = threshold
        self.adjacency_matrix = None
        self.edge_index = None
        self.edge_weight = None
        
    def load_precomputed_adjacency(self, file_path, data):
        """Load pre-computed adjacency matrix from pickle file"""
        print(f"Loading pre-computed adjacency matrix from {file_path}...")
        
        # Load the adjacency matrix data
        with open(file_path, 'rb') as f:
            adj_data = pickle.load(f, encoding='latin1')
        
        # Extract components: [sensor_ids, id_to_index_map, adjacency_matrix]
        sensor_ids, id_to_index, adj_matrix = adj_data
        
        # Create mapping from data columns to adjacency matrix indices
        data_sensor_indices = []
        for col in data.columns:
            if col in id_to_index:
                data_sensor_indices.append(id_to_index[col])
            else:
                print(f"Warning: Sensor {col} not found in adjacency matrix")
        
        # Filter adjacency matrix to match available sensors
        if len(data_sensor_indices) < len(data.columns):
            print(f"Using {len(data_sensor_indices)} out of {len(data.columns)} sensors")
        
        # Extract submatrix for available sensors
        adj_sub = adj_matrix[np.ix_(data_sensor_indices, data_sensor_indices)]
        
        # Create adjacency matrix DataFrame
        available_sensors = [data.columns[i] for i in range(len(data.columns)) 
                           if data.columns[i] in id_to_index]
        
        self.adjacency_matrix = pd.DataFrame(adj_sub, 
                                           index=available_sensors, 
                                           columns=available_sensors)
        
        # Convert to edge format for PyTorch Geometric
        threshold = 0.1  # Adjust this threshold as needed
        adj_binary = (adj_sub > threshold).astype(int)
        np.fill_diagonal(adj_binary, 0)  # Remove self-loops
        
        edge_indices = np.where(adj_binary == 1)
        self.edge_index = torch.tensor(np.vstack(edge_indices), dtype=torch.long)
        
        # Edge weights are the original adjacency values
        edge_weights = adj_sub[edge_indices]
        self.edge_weight = torch.tensor(edge_weights, dtype=torch.float)
        
        print(f"Pre-computed graph loaded: {len(available_sensors)} nodes, {len(edge_weights)} edges")
        
        return self.edge_index, self.edge_weight
        
    def build_correlation_graph(self, data):
        """Build graph based on traffic correlation"""
        print("Building correlation-based graph...")
        
        # Calculate correlation matrix
        corr_matrix = data.corr().abs()
        
        # Create adjacency matrix (threshold correlation)
        adj_matrix = (corr_matrix > self.threshold).astype(int)
        np.fill_diagonal(adj_matrix.values, 0)  # Remove self-loops
        
        self.adjacency_matrix = adj_matrix
        
        # Convert to edge list format for PyTorch Geometric
        edge_indices = np.where(adj_matrix.values == 1)
        self.edge_index = torch.tensor(np.vstack(edge_indices), dtype=torch.long)
        
        # Edge weights are correlation values
        edge_weights = corr_matrix.values[edge_indices]
        self.edge_weight = torch.tensor(edge_weights, dtype=torch.float)
        
        print(f"Graph created: {len(data.columns)} nodes, {len(edge_weights)} edges")
        return self.edge_index, self.edge_weight
    
    def build_knn_graph(self, data, k=5):
        """Build k-nearest neighbors graph"""
        print(f"Building {k}-NN graph...")
        
        # Calculate correlation matrix
        corr_matrix = data.corr().abs()
        
        # For each node, keep only k strongest connections
        adj_matrix = np.zeros_like(corr_matrix.values)
        
        for i in range(len(corr_matrix)):
            # Get k nearest neighbors (excluding self)
            neighbors = np.argsort(corr_matrix.iloc[i].values)[-k-1:-1]
            adj_matrix[i, neighbors] = 1
        
        # Make symmetric
        adj_matrix = np.maximum(adj_matrix, adj_matrix.T)
        
        self.adjacency_matrix = pd.DataFrame(adj_matrix, 
                                           index=corr_matrix.index, 
                                           columns=corr_matrix.columns)
        
        # Convert to edge format
        edge_indices = np.where(adj_matrix == 1)
        self.edge_index = torch.tensor(np.vstack(edge_indices), dtype=torch.long)
        
        edge_weights = corr_matrix.values[edge_indices]
        self.edge_weight = torch.tensor(edge_weights, dtype=torch.float)
        
        print(f"Graph created: {len(data.columns)} nodes, {len(edge_weights)} edges")
        return self.edge_index, self.edge_weight

class TemporalDataProcessor:
    """Prepare temporal sequences for GNN training"""
    
    def __init__(self, seq_length=12, pred_length=3, stride=1):
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.stride = stride
        self.scaler = StandardScaler()
        
    def create_sequences(self, data, normalize=True):
        """Create sliding window sequences"""
        print(f"Creating temporal sequences...")
        print(f"   Input length: {self.seq_length}, Prediction length: {self.pred_length}")
        
        # Normalize data
        if normalize:
            data_scaled = self.scaler.fit_transform(data.values)
            data_df = pd.DataFrame(data_scaled, index=data.index, columns=data.columns)
        else:
            data_df = data.copy()
        
        # Create sequences
        X, y = [], []
        
        for i in range(0, len(data_df) - self.seq_length - self.pred_length + 1, self.stride):
            # Input sequence
            seq_x = data_df.iloc[i:i + self.seq_length].values  # [seq_len, n_nodes]
            # Target sequence
            seq_y = data_df.iloc[i + self.seq_length:i + self.seq_length + self.pred_length].values
            
            X.append(seq_x)
            y.append(seq_y)
        
        X = np.array(X)  # [n_samples, seq_len, n_nodes]
        y = np.array(y)  # [n_samples, pred_len, n_nodes]
        
        print(f"Created {len(X)} sequences")
        print(f"   Input shape: {X.shape}, Target shape: {y.shape}")
        
        return X, y
    
    def train_test_split(self, X, y, train_ratio=0.7, val_ratio=0.2):
        """Split data chronologically"""
        n_samples = len(X)
        
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))
        
        X_train = X[:train_end]
        y_train = y[:train_end]
        
        X_val = X[train_end:val_end]
        y_val = y[train_end:val_end]
        
        X_test = X[val_end:]
        y_test = y[val_end:]
        
        print(f"Data split:")
        print(f"   Train: {len(X_train)} samples ({len(X_train)/n_samples*100:.1f}%)")
        print(f"   Val:   {len(X_val)} samples ({len(X_val)/n_samples*100:.1f}%)")
        print(f"   Test:  {len(X_test)} samples ({len(X_test)/n_samples*100:.1f}%)")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# =============================================================================
# RESULT MANAGEMENT SYSTEM
# =============================================================================

class ComprehensiveResultManager:
    """Comprehensive system for saving all results, logs, CSV exports"""
    
    def __init__(self, base_dir='traffic_gnn_results', experiment_name=None, out_dir='.'):
        self.out_dir = out_dir
        self.base_dir = os.path.join(out_dir, base_dir)
        self.experiment_name = experiment_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_dir = os.path.join(self.base_dir, self.experiment_name)
        
        # Create directory structure
        self.dirs = {
            'models': os.path.join(self.experiment_dir, 'models'),
            'plots': os.path.join(self.experiment_dir, 'plots'),
            'data': os.path.join(self.experiment_dir, 'data'),
            'logs': os.path.join(self.experiment_dir, 'logs'),
            'csv': os.path.join(self.experiment_dir, 'csv'),
            'csv_training': os.path.join(self.experiment_dir, 'csv', 'training_history'),
            'csv_predictions': os.path.join(self.experiment_dir, 'csv', 'predictions'),
            'csv_metrics': os.path.join(self.experiment_dir, 'csv', 'metrics'),
            'csv_analysis': os.path.join(self.experiment_dir, 'csv', 'analysis')
        }
        
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Initialize storage
        self.results = {}
        self.metadata = {
            'experiment_name': self.experiment_name,
            'timestamp': datetime.now().isoformat(),
            'dataset_info': {},
            'model_configs': {},
            'training_configs': {}
        }
        
        self.logger.info(f"Initialized ComprehensiveResultManager for experiment: {self.experiment_name}")
    
    def setup_logging(self):
        """Setup comprehensive logging system"""
        # Create logger
        self.logger = logging.getLogger(f'traffic_gnn_{self.experiment_name}')
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # File handler
        log_file = os.path.join(self.dirs['logs'], 'experiment.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def save_dataset_info(self, traffic_data, processor):
        """Save dataset information with CSV export"""
        dataset_info = {
            'shape': traffic_data.shape,
            'sensors': list(traffic_data.columns),
            'time_range': {
                'start': str(traffic_data.index[0]),
                'end': str(traffic_data.index[-1])
            },
            'sequence_length': processor.seq_length,
            'prediction_length': processor.pred_length,
            'missing_values': int(traffic_data.isnull().sum().sum()),
            'memory_usage_mb': float(traffic_data.memory_usage(deep=True).sum() / 1024**2)
        }
        
        self.metadata['dataset_info'] = dataset_info
        
        # Save to JSON
        with open(os.path.join(self.dirs['data'], 'dataset_info.json'), 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        # CSV Export: Dataset statistics per sensor
        sensor_stats = traffic_data.describe()
        sensor_stats.to_csv(os.path.join(self.dirs['csv'], 'sensor_statistics.csv'))
        
        # CSV Export: Dataset summary
        summary_data = {
            'metric': ['num_sensors', 'num_timesteps', 'seq_length', 'pred_length', 'missing_values', 'memory_mb'],
            'value': [dataset_info['shape'][1], dataset_info['shape'][0], dataset_info['sequence_length'],
                     dataset_info['prediction_length'], dataset_info['missing_values'], dataset_info['memory_usage_mb']]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(self.dirs['csv'], 'dataset_summary.csv'), index=False)
        
        self.logger.info(f"Dataset info saved: {traffic_data.shape[0]} samples, {traffic_data.shape[1]} sensors")
    
    def save_model_and_results(self, model_name, trainer, model_config, test_results):
        """Save model, training results, and generate comprehensive CSV exports"""
        model_dir = os.path.join(self.dirs['models'], model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model state
        model_path = os.path.join(model_dir, 'model.pth')
        torch.save(trainer.model.state_dict(), model_path)
        
        # Save training history
        training_history = {
            'train_losses': trainer.train_losses,
            'val_losses': trainer.val_losses,
            'train_maes': trainer.train_maes,
            'val_maes': trainer.val_maes,
            'test_loss': test_results['test_loss'],
            'test_mae': test_results['test_mae']
        }
        
        with open(os.path.join(model_dir, 'training_history.json'), 'w') as f:
            json.dump(training_history, f, indent=2)
        
        # Save model configuration
        with open(os.path.join(model_dir, 'model_config.json'), 'w') as f:
            json.dump(model_config, f, indent=2)
        
        # CSV Export: Training history
        self.save_training_history_csv(model_name, trainer)
        
        # Store in results for comparison
        self.results[model_name] = {
            'trainer': trainer,
            'test_results': test_results,
            'model_config': model_config,
            'training_history': training_history
        }
        
        self.logger.info(f"Model {model_name} saved - Test Loss: {test_results['test_loss']:.4f}, Test MAE: {test_results['test_mae']:.4f}")
    
    def save_training_history_csv(self, model_name, trainer):
        """Save detailed training history as CSV"""
        # Training metrics per epoch
        epochs = list(range(1, len(trainer.train_losses) + 1))
        training_df = pd.DataFrame({
            'epoch': epochs,
            'train_loss': trainer.train_losses,
            'val_loss': trainer.val_losses,
            'train_mae': trainer.train_maes,
            'val_mae': trainer.val_maes,
            'model': [model_name] * len(epochs)
        })
        
        training_df.to_csv(os.path.join(self.dirs['csv_training'], f'{model_name}_training_history.csv'), index=False)
    
    def save_experiment_summary(self, results_dict):
        """Save comprehensive experiment summary"""
        
        # Update metadata
        self.metadata['num_models'] = len(results_dict)
        if results_dict:
            self.metadata['best_model'] = min(results_dict.keys(), 
                                            key=lambda x: results_dict[x]['test_results']['test_loss'])
        
        # Save metadata
        with open(os.path.join(self.experiment_dir, 'experiment_metadata.json'), 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        # Save all results with robust null-checking
        results_summary = {}
        for model_name, results in results_dict.items():
            summary = {
                'test_loss': results.get('test_results', {}).get('test_loss', 'N/A'),
                'test_mae': results.get('test_results', {}).get('test_mae', 'N/A')
            }
            
            # Handle training history with null checks
            if 'training_history' in results and results['training_history']:
                training_hist = results['training_history']
                if training_hist.get('train_losses') and len(training_hist['train_losses']) > 0:
                    summary['final_train_loss'] = training_hist['train_losses'][-1]
                    summary['epochs'] = len(training_hist['train_losses'])
                else:
                    summary['final_train_loss'] = 'N/A'
                    summary['epochs'] = 0
                    
                if training_hist.get('val_losses') and len(training_hist['val_losses']) > 0:
                    summary['final_val_loss'] = training_hist['val_losses'][-1]
                else:
                    summary['final_val_loss'] = 'N/A'
            else:
                summary['final_train_loss'] = 'N/A'
                summary['final_val_loss'] = 'N/A'
                summary['epochs'] = 'N/A'
            
            # Add model config if available
            if 'model_config' in results:
                summary['model_config'] = results['model_config']
            else:
                summary['model_config'] = {}
                
            results_summary[model_name] = summary
        
        with open(os.path.join(self.experiment_dir, 'results_summary.json'), 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        self.logger.info(f"Experiment summary saved to {self.experiment_dir}")
        if results_dict:
            self.logger.info(f"Best model: {self.metadata['best_model']}")
        
        return self.experiment_dir

# =============================================================================
# GNN MODEL ARCHITECTURES
# =============================================================================

class STGCN(nn.Module):
    """Spatial-Temporal Graph Convolutional Network"""
    
    def __init__(self, num_nodes, in_channels=1, hidden_channels=64, out_channels=1, num_layers=2, seq_length=12, pred_length=3):
        super(STGCN, self).__init__()
        
        self.num_nodes = num_nodes
        self.seq_length = seq_length
        self.pred_length = pred_length
        
        # Input projection
        self.input_proj = nn.Linear(seq_length, hidden_channels)
        
        # Spatial-temporal blocks
        self.gconv_layers = nn.ModuleList()
        self.temporal_layers = nn.ModuleList()
        
        for i in range(num_layers):
            self.gconv_layers.append(ChebConv(hidden_channels, hidden_channels, K=3))
            self.temporal_layers.append(nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1))
        
        # Output projection
        self.output_proj = nn.Linear(hidden_channels, pred_length)
        
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(hidden_channels)
        
    def forward(self, x, edge_index, edge_weight=None):
        # x: [batch, seq_len, nodes]
        batch_size, seq_len, num_nodes = x.shape
        
        # Input projection: [batch, seq_len, nodes] -> [batch, nodes, hidden_channels]
        x = x.transpose(1, 2)  # [batch, nodes, seq_len]
        x = self.input_proj(x)  # [batch, nodes, hidden_channels]
        
        # Apply spatial-temporal layers
        for gconv, temp_conv in zip(self.gconv_layers, self.temporal_layers):
            residual = x
            
            # Spatial modeling: Graph convolution
            x_list = []
            for b in range(batch_size):
                x_b = gconv(x[b], edge_index, edge_weight)
                x_list.append(x_b)
            x = torch.stack(x_list, dim=0)  # [batch, nodes, hidden_channels]
            
            # Temporal modeling: 1D convolution
            x = x.transpose(1, 2)  # [batch, hidden_channels, nodes]
            x = temp_conv(x)
            x = x.transpose(1, 2)  # [batch, nodes, hidden_channels]
            
            # Residual connection and normalization
            x = self.layer_norm(x + residual)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Output projection: [batch, nodes, hidden_channels] -> [batch, nodes, pred_length]
        x = self.output_proj(x)
        
        # Reshape to [batch, pred_length, nodes]
        x = x.transpose(1, 2)
        
        return x

class GraphWaveNet(nn.Module):
    """Graph WaveNet for traffic forecasting"""
    
    def __init__(self, num_nodes, in_channels=1, residual_channels=32, dilation_channels=32, 
                 skip_channels=256, end_channels=512, layers=1, seq_length=12, pred_length=3):
        super(GraphWaveNet, self).__init__()
        
        self.num_nodes = num_nodes
        self.seq_length = seq_length
        self.pred_length = pred_length
        
        # Start convolution
        self.start_conv = nn.Conv2d(in_channels, residual_channels, kernel_size=(1, 1))
        
        # Dilated convolution layers
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv = nn.ModuleList()
        
        for i in range(layers):
            dilation = 2 ** i
            
            self.filter_convs.append(nn.Conv2d(residual_channels, dilation_channels, 
                                               kernel_size=(1, 2), dilation=(1, dilation)))
            self.gate_convs.append(nn.Conv2d(residual_channels, dilation_channels, 
                                             kernel_size=(1, 2), dilation=(1, dilation)))
            self.residual_convs.append(nn.Conv2d(dilation_channels, residual_channels, kernel_size=(1, 1)))
            self.skip_convs.append(nn.Conv2d(dilation_channels, skip_channels, kernel_size=(1, 1)))
            self.gconv.append(GCNConv(dilation_channels, dilation_channels))
        
        # End layers
        self.end_conv_1 = nn.Conv2d(skip_channels, end_channels, kernel_size=(1, 1), bias=True)
        self.end_conv_2 = nn.Conv2d(end_channels, pred_length, kernel_size=(1, 1), bias=True)
        
    def forward(self, x, edge_index, edge_weight=None):
        # x: [batch, seq_len, nodes] -> [batch, 1, nodes, seq_len]
        x = x.transpose(1, 2).unsqueeze(1)
        
        x = self.start_conv(x)
        skip = 0
        
        # TCN layers with graph convolution
        for i in range(len(self.filter_convs)):
            residual = x
            
            # Dilated convolution
            filter_out = self.filter_convs[i](x)
            gate_out = self.gate_convs[i](x)
            
            # Graph convolution on the filtered output
            batch_size, channels, num_nodes, time_steps = filter_out.shape
            filter_out = filter_out.permute(0, 3, 2, 1).contiguous()
            filter_out = filter_out.view(-1, num_nodes, channels)
            
            graph_out = []
            for j in range(filter_out.size(0)):
                graph_out.append(self.gconv[i](filter_out[j], edge_index, edge_weight))
            filter_out = torch.stack(graph_out, dim=0)
            
            filter_out = filter_out.view(batch_size, time_steps, num_nodes, channels)
            filter_out = filter_out.permute(0, 3, 2, 1).contiguous()
            
            # Gated activation
            x = torch.tanh(filter_out) * torch.sigmoid(gate_out)
            
            # Residual and skip connections
            s = x
            if x.size(3) != residual.size(3):
                residual = residual[:, :, :, -x.size(3):]
            
            x = self.residual_convs[i](x)
            x = x + residual[:, :, :, -x.size(3):]
            
            skip = skip + self.skip_convs[i](s)
        
        # End convolutions
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)  # [batch, pred_len, nodes, remaining_time]
        
        # Global pooling over time dimension
        x = F.adaptive_avg_pool2d(x, (self.num_nodes, 1))
        x = x.squeeze(-1)  # [batch, pred_len, nodes]
        
        return x

class GraphAttentionNetwork(nn.Module):
    """Graph Attention Network for traffic forecasting"""
    
    def __init__(self, num_nodes, in_channels=1, hidden_channels=64, out_channels=1, 
                 num_layers=3, heads=8, seq_length=12, pred_length=3, dropout=0.1):
        super(GraphAttentionNetwork, self).__init__()
        
        self.num_nodes = num_nodes
        self.seq_length = seq_length
        self.pred_length = pred_length
        
        # Input projection
        self.input_proj = nn.Linear(seq_length, hidden_channels)
        
        # GAT layers
        self.gat_layers = nn.ModuleList()
        
        # First layer
        self.gat_layers.append(GATConv(hidden_channels, hidden_channels // heads, heads=heads, dropout=dropout))
        
        # Middle layers
        for _ in range(num_layers - 2):
            self.gat_layers.append(GATConv(hidden_channels, hidden_channels // heads, heads=heads, dropout=dropout))
        
        # Last layer
        self.gat_layers.append(GATConv(hidden_channels, hidden_channels, heads=1, dropout=dropout))
        
        # Temporal modeling
        self.temporal_conv = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_channels, pred_length)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index, edge_weight=None):
        # x: [batch, seq_len, nodes]
        batch_size, seq_len, num_nodes = x.shape
        
        # Project input
        x = x.transpose(1, 2)  # [batch, nodes, seq_len]
        x = self.input_proj(x)  # [batch, nodes, hidden_channels]
        
        # Apply GAT layers
        for i, gat_layer in enumerate(self.gat_layers):
            x_list = []
            for b in range(batch_size):
                x_b = gat_layer(x[b], edge_index, edge_weight)
                x_list.append(x_b)
            x = torch.stack(x_list, dim=0)
            
            if i < len(self.gat_layers) - 1:
                x = F.elu(x)
                x = self.dropout(x)
        
        # Temporal convolution
        x = x.transpose(1, 2)  # [batch, hidden_channels, nodes]
        x = self.temporal_conv(x)
        x = F.relu(x)
        x = x.transpose(1, 2)  # [batch, nodes, hidden_channels]
        
        # Output projection
        x = self.output_proj(x)  # [batch, nodes, pred_length]
        x = x.transpose(1, 2)  # [batch, pred_length, nodes]
        
        return x

class GraphTransformer(nn.Module):
    """Graph Transformer for traffic forecasting"""
    
    def __init__(self, num_nodes, in_channels=1, hidden_channels=64, out_channels=1, 
                 num_layers=3, heads=8, seq_length=12, pred_length=3, dropout=0.1):
        super(GraphTransformer, self).__init__()
        
        self.num_nodes = num_nodes
        self.seq_length = seq_length
        self.pred_length = pred_length
        
        # Input projection
        self.input_proj = nn.Linear(seq_length, hidden_channels)
        
        # Graph Transformer layers
        self.transformer_layers = nn.ModuleList()
        
        for _ in range(num_layers):
            self.transformer_layers.append(
                TransformerConv(hidden_channels, hidden_channels // heads, heads=heads, dropout=dropout)
            )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_channels, pred_length)
        
        self.layer_norm = nn.LayerNorm(hidden_channels)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index, edge_weight=None):
        # x: [batch, seq_len, nodes]
        batch_size, seq_len, num_nodes = x.shape
        
        # Project input
        x = x.transpose(1, 2)  # [batch, nodes, seq_len]
        x = self.input_proj(x)  # [batch, nodes, hidden_channels]
        
        # Apply Transformer layers
        for transformer_layer in self.transformer_layers:
            x_list = []
            for b in range(batch_size):
                x_b = transformer_layer(x[b], edge_index)
                x_list.append(x_b)
            residual = x
            x = torch.stack(x_list, dim=0)
            x = self.layer_norm(x + residual)
            x = self.dropout(x)
        
        # Output projection
        x = self.output_proj(x)  # [batch, nodes, pred_length]
        x = x.transpose(1, 2)  # [batch, pred_length, nodes]
        
        return x

# =============================================================================
# TRAINING FRAMEWORK
# =============================================================================

class TrafficGNNTrainer:
    """Training framework for traffic GNN models with progress bars and comprehensive logging"""
    
    def __init__(self, model, device='cpu', learning_rate=0.001, weight_decay=1e-4, 
                 show_progress=True, result_manager=None):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.7)
        self.show_progress = show_progress
        self.result_manager = result_manager
        
        self.train_losses = []
        self.val_losses = []
        self.train_maes = []
        self.val_maes = []
        
        # Log training configuration
        if self.result_manager:
            self.result_manager.logger.info(f"TrafficGNNTrainer initialized - LR: {learning_rate}, Device: {device}")
        
    def train_epoch(self, X_train, y_train, edge_index, edge_weight, batch_size=32):
        """Train for one epoch with progress bar and logging"""
        self.model.train()
        total_loss = 0
        total_mae = 0
        num_batches = 0
        
        # Create batches
        num_samples = X_train.size(0)
        indices = torch.randperm(num_samples)
        
        # Progress bar for batches
        batch_iterator = range(0, num_samples, batch_size)
        if self.show_progress:
            batch_iterator = tqdm(batch_iterator, desc="Training Batches", 
                                leave=False, unit="batch", 
                                bar_format='{l_bar}{bar:20}{r_bar}')
        
        for i in batch_iterator:
            batch_indices = indices[i:i + batch_size]
            X_batch = X_train[batch_indices].to(self.device)
            y_batch = y_train[batch_indices].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Handle None edge_index for FC models
            if edge_index is not None:
                edge_index_device = edge_index.to(self.device)
                edge_weight_device = edge_weight.to(self.device) if edge_weight is not None else None
            else:
                edge_index_device = None
                edge_weight_device = None
                
            y_pred = self.model(X_batch, edge_index_device, edge_weight_device)
            
            # Calculate loss
            loss = F.mse_loss(y_pred, y_batch)
            mae = F.l1_loss(y_pred, y_batch)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_mae += mae.item()
            num_batches += 1
            
            # Update progress bar with current metrics
            if self.show_progress and hasattr(batch_iterator, 'set_postfix'):
                batch_iterator.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'MAE': f'{mae.item():.4f}'
                })
        
        avg_loss = total_loss / num_batches
        avg_mae = total_mae / num_batches
        
        return avg_loss, avg_mae
    
    def validate(self, X_val, y_val, edge_index, edge_weight, batch_size=32):
        """Validate the model with progress bar and logging"""
        self.model.eval()
        total_loss = 0
        total_mae = 0
        num_batches = 0
        
        with torch.no_grad():
            num_samples = X_val.size(0)
            
            # Progress bar for validation batches
            batch_iterator = range(0, num_samples, batch_size)
            if self.show_progress:
                batch_iterator = tqdm(batch_iterator, desc="Validation", 
                                    leave=False, unit="batch",
                                    bar_format='{l_bar}{bar:20}{r_bar}')
            
            for i in batch_iterator:
                X_batch = X_val[i:i + batch_size].to(self.device)
                y_batch = y_val[i:i + batch_size].to(self.device)
                
                # Handle None edge_index for FC models
                if edge_index is not None:
                    edge_index_device = edge_index.to(self.device)
                    edge_weight_device = edge_weight.to(self.device) if edge_weight is not None else None
                else:
                    edge_index_device = None
                    edge_weight_device = None
                
                y_pred = self.model(X_batch, edge_index_device, edge_weight_device)
                
                loss = F.mse_loss(y_pred, y_batch)
                mae = F.l1_loss(y_pred, y_batch)
                
                total_loss += loss.item()
                total_mae += mae.item()
                num_batches += 1
                
                # Update progress bar
                if self.show_progress and hasattr(batch_iterator, 'set_postfix'):
                    batch_iterator.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'MAE': f'{mae.item():.4f}'
                    })
        
        avg_loss = total_loss / num_batches
        avg_mae = total_mae / num_batches
        
        return avg_loss, avg_mae
    
    def train(self, X_train, y_train, X_val, y_val, edge_index, edge_weight, 
              epochs=100, batch_size=32, early_stopping_patience=15, verbose=True):
        """Full training loop with progress bars and comprehensive logging"""
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Log training start
        if self.result_manager:
            self.result_manager.logger.info(f"Starting training for {epochs} epochs - Model: {self.model.__class__.__name__}")
        
        print(f"Starting training for {epochs} epochs...")
        print(f"   Model: {self.model.__class__.__name__}")
        print(f"   Device: {self.device}")
        print(f"   Batch size: {batch_size}")
        
        # Progress bar for epochs
        epoch_iterator = range(epochs)
        if self.show_progress:
            epoch_iterator = tqdm(epoch_iterator, desc="Training Progress", 
                                unit="epoch", ncols=120,
                                bar_format='{l_bar}{bar:30}{r_bar}{bar:-30b}')
        
        for epoch in epoch_iterator:
            # Training
            train_loss, train_mae = self.train_epoch(X_train, y_train, edge_index, edge_weight, batch_size)
            
            # Validation
            val_loss, val_mae = self.validate(X_val, y_val, edge_index, edge_weight, batch_size)
            
            # Learning rate scheduling
            self.scheduler.step()
            lr = self.optimizer.param_groups[0]['lr']
            
            # Save metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_maes.append(train_mae)
            self.val_maes.append(val_mae)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                if self.result_manager:
                    model_save_path = os.path.join(self.result_manager.dirs['models'], 'best_model.pth')
                    torch.save(self.model.state_dict(), model_save_path)
                best_indicator = "★"
                
                # Log new best model
                if self.result_manager:
                    self.result_manager.logger.info(f"New best model at epoch {epoch+1} - Val Loss: {val_loss:.4f}")
            else:
                patience_counter += 1
                best_indicator = " "
            
            # Update progress bar with detailed metrics
            if self.show_progress and hasattr(epoch_iterator, 'set_postfix'):
                epoch_iterator.set_postfix({
                    'T_Loss': f'{train_loss:.4f}',
                    'V_Loss': f'{val_loss:.4f}',
                    'T_MAE': f'{train_mae:.4f}',
                    'V_MAE': f'{val_mae:.4f}',
                    'LR': f'{lr:.1e}',
                    'Patience': f'{patience_counter}/{early_stopping_patience}',
                    'Best': best_indicator
                })
            
            # Verbose output every 10 epochs
            if verbose and (epoch + 1) % 10 == 0:
                print(f"\nEpoch {epoch+1:3d}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                      f"Train MAE: {train_mae:.4f}, Val MAE: {val_mae:.4f}, LR: {lr:.6f}")
                if best_indicator == "★":
                    print("               ★ New best model saved!")
            
            if patience_counter >= early_stopping_patience:
                if self.show_progress and hasattr(epoch_iterator, 'set_description'):
                    epoch_iterator.set_description("Early Stopping")
                print(f"\nEarly stopping at epoch {epoch+1}")
                if self.result_manager:
                    self.result_manager.logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        # Load best model
        if self.result_manager:
            model_load_path = os.path.join(self.result_manager.dirs['models'], 'best_model.pth')
            if os.path.exists(model_load_path):
                self.model.load_state_dict(torch.load(model_load_path))
        
        print(f"\nTraining completed! Best validation loss: {best_val_loss:.4f}")
        
        # Log training completion
        if self.result_manager:
            self.result_manager.logger.info(f"Training completed - Best val loss: {best_val_loss:.4f}, Total epochs: {len(self.train_losses)}")

# =============================================================================
# EXPERIMENT FRAMEWORK
# =============================================================================

class ModelExperiment:
    """Framework for comparing different GNN architectures with comprehensive result management"""
    
    def __init__(self, device='cpu', show_progress=True, result_manager=None):
        self.device = device
        self.results = {}
        self.models = {}
        self.show_progress = show_progress
        self.result_manager = result_manager
        
        if self.result_manager:
            self.result_manager.logger.info(f"ModelExperiment initialized with device: {device}")
        
    def add_model(self, name, model_class, **kwargs):
        """Add a model configuration to experiment"""
        self.models[name] = {
            'class': model_class,
            'kwargs': kwargs
        }
        
        if self.result_manager:
            self.result_manager.logger.info(f"Added model {name} to experiment")
        
    def run_experiment(self, X_train, y_train, X_val, y_val, X_test, y_test, 
                      edge_index, edge_weight, epochs=50, batch_size=32):
        """Run experiments for all models with comprehensive result saving"""
        
        print("Starting model comparison experiments...")
        
        if self.result_manager:
            self.result_manager.logger.info(f"Starting model comparison experiment with {len(self.models)} models")
        
        # First, test all models to ensure they work
        print("\nTesting all models before training...")
        valid_models = {}
        
        for model_name, model_config in self.models.items():
            try:
                model_test = model_config['class'](**model_config['kwargs'])
                with torch.no_grad():
                    test_input = X_train[:2]
                    test_output = model_test(test_input, edge_index, edge_weight)
                    expected_shape = y_train[:2].shape
                    print(f"   {model_name}: {test_output.shape} (expected: {expected_shape})")
                    assert test_output.shape == expected_shape, f"Shape mismatch for {model_name}"
                    valid_models[model_name] = model_config
                    
                    if self.result_manager:
                        self.result_manager.logger.info(f"Model {model_name} validation passed")
                        
            except Exception as e:
                print(f"   {model_name}: Failed - {e}")
                print(f"   Removing {model_name} from experiments")
                
                if self.result_manager:
                    self.result_manager.logger.warning(f"Model {model_name} validation failed: {e}")
                continue
        
        self.models = valid_models
        print(f"\nStarting actual training for {len(self.models)} valid models...")
        
        # Progress bar for model experiments
        model_items = list(self.models.items())
        if self.show_progress:
            model_iterator = tqdm(model_items, desc="Model Comparison", 
                                unit="model", ncols=100)
        else:
            model_iterator = model_items
        
        for model_name, model_config in model_iterator:
            if self.show_progress and hasattr(model_iterator, 'set_description'):
                model_iterator.set_description(f"Training {model_name}")
            
            print(f"\n🔬 Training {model_name}...")
            
            try:
                # Initialize model
                model = model_config['class'](**model_config['kwargs'])
                trainer = TrafficGNNTrainer(model, device=self.device, 
                                          show_progress=self.show_progress,
                                          result_manager=self.result_manager)
                
                # Train model
                trainer.train(X_train, y_train, X_val, y_val, edge_index, edge_weight, 
                             epochs=epochs, batch_size=batch_size, verbose=False)
                
                # Evaluate on test set
                test_loss, test_mae = trainer.validate(X_test, y_test, edge_index, edge_weight, batch_size)
                
                # Prepare test results
                test_results = {
                    'test_loss': test_loss,
                    'test_mae': test_mae
                }
                
                # Save model and results using result manager
                if self.result_manager:
                    self.result_manager.save_model_and_results(
                        model_name, trainer, model_config['kwargs'], test_results
                    )
                
                # Store in results for comparison
                self.results[model_name] = {
                    'trainer': trainer,
                    'test_results': test_results,
                    'model_config': model_config['kwargs'],
                    'training_history': {
                        'train_losses': trainer.train_losses,
                        'val_losses': trainer.val_losses,
                        'train_maes': trainer.train_maes,
                        'val_maes': trainer.val_maes
                    }
                }
                
                print(f"   ✅ {model_name}: Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}")
                
            except Exception as e:
                print(f"   ❌ {model_name}: Training failed - {e}")
                print(f"   Skipping {model_name}")
                
                if self.result_manager:
                    self.result_manager.logger.error(f"Model {model_name} training failed: {e}")
                continue
        
        if self.show_progress and hasattr(model_iterator, 'set_description'):
            model_iterator.set_description("Model Comparison Complete")
        
        if self.results:
            print(f"\n🎯 Experiment completed! {len(self.results)} models trained successfully.")
            self.print_ranking()
            
            # Save comprehensive results if result manager available
            if self.result_manager:
                experiment_dir = self.result_manager.save_experiment_summary(self.results)
                print(f"\n📁 All results saved to: {experiment_dir}")
        else:
            print("❌ No models completed successfully!")
            
        return self.results
    
    def print_ranking(self):
        """Print model ranking by performance"""
        if not self.results:
            return
        
        print("\n📊 MODEL RANKING (by Test Loss):")
        print("-" * 50)
        
        # Sort by test loss
        sorted_results = sorted(self.results.items(), key=lambda x: x[1]['test_results']['test_loss'])
        
        for i, (model_name, results) in enumerate(sorted_results, 1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            test_loss = results['test_results']['test_loss']
            test_mae = results['test_results']['test_mae']
            print(f"{medal} {model_name:20s} - Loss: {test_loss:.4f}, MAE: {test_mae:.4f}")
        
        # Log ranking
        if self.result_manager:
            best_model = sorted_results[0][0]
            self.result_manager.logger.info(f"Best performing model: {best_model}")

def auto_detect_dataset_and_build_graph(traffic_data):
    """Automatically detect dataset and build appropriate graph"""
    
    print("=== AUTOMATIC GRAPH CONSTRUCTION ===")
    print("🔍 Detecting dataset and selecting optimal graph construction method...")
    
    # Try to determine dataset type from shape and available files
    num_sensors = traffic_data.shape[1]
    
    # Common sensor counts for known datasets
    if num_sensors == 207:
        dataset_name = "METR-LA"
        adj_file = './data/adj_mx_METR-LA.pkl'
    else:
        dataset_name = "Unknown"
        adj_file = './data/adj_mx_METR-LA.pkl'  # Default fallback
    
    print(f"📊 Dataset detected: {dataset_name} ({num_sensors} sensors)")
    
    # Priority 1: Try to load pre-computed adjacency matrix
    try:
        print(f"🔗 Attempting to load pre-computed adjacency matrix...")
        precomputed_builder = TrafficGraphBuilder(method='precomputed')
        edge_index, edge_weight = precomputed_builder.load_precomputed_adjacency(adj_file, traffic_data)
        
        print(f"✅ SUCCESS: Using pre-computed adjacency matrix for {dataset_name}")
        return edge_index, edge_weight, precomputed_builder, "precomputed"
        
    except Exception as e:
        print(f"❌ Failed to load pre-computed adjacency matrix: {e}")
        print("📋 Falling back to correlation-based graph construction...")
        
        # Priority 2: Fallback to correlation-based method
        try:
            correlation_builder = TrafficGraphBuilder(method='correlation', threshold=0.3)
            edge_index, edge_weight = correlation_builder.build_correlation_graph(traffic_data)
            
            print(f"⚠️  FALLBACK: Using correlation-based graph (threshold=0.3)")
            return edge_index, edge_weight, correlation_builder, "correlation"
            
        except Exception as e2:
            print(f"❌ Correlation-based method also failed: {e2}")
            
            # Priority 3: Last resort - k-NN method
            print("📋 Last resort: Using k-NN graph construction...")
            knn_builder = TrafficGraphBuilder(method='knn')
            edge_index, edge_weight = knn_builder.build_knn_graph(traffic_data, k=8)
            
            print(f"🔧 EMERGENCY FALLBACK: Using k-NN graph (k=8)")
            return edge_index, edge_weight, knn_builder, "knn"

# =============================================================================
# EXPERIMENT FUNCTIONS
# =============================================================================

def experiment_single_model(traffic_data, X_train, y_train, X_val, y_val, X_test, y_test, 
                           edge_index, edge_weight, processor, result_manager, device, batch_size=32):
    """Experiment 1: Single Model Training and Evaluation"""
    
    print("\n" + "="*60)
    print("EXPERIMENT 1: STGCN Model Training with Result Management")
    print("="*60)
    
    # Initialize STGCN model
    num_nodes = traffic_data.shape[1]
    seq_length = processor.seq_length
    pred_length = processor.pred_length
    
    stgcn_model = STGCN(
        num_nodes=num_nodes,
        in_channels=1,
        hidden_channels=64,
        out_channels=1,
        num_layers=3,
        seq_length=seq_length,
        pred_length=pred_length
    )
    
    # Train the model with result manager integration
    stgcn_trainer = TrafficGNNTrainer(stgcn_model, device=device, learning_rate=0.001, 
                                      result_manager=result_manager)
    stgcn_trainer.train(X_train, y_train, X_val, y_val, edge_index, edge_weight, 
                        epochs=50, batch_size=batch_size, verbose=True)
    
    # Evaluate on test set
    test_loss, test_mae = stgcn_trainer.validate(X_test, y_test, edge_index, edge_weight, batch_size=batch_size)
    print(f"\nSTGCN Test Results:")
    print(f"   Test Loss (MSE): {test_loss:.4f}")
    print(f"   Test MAE: {test_mae:.4f}")
    
    # Save model and results comprehensively
    model_config = {
        'num_nodes': num_nodes,
        'hidden_channels': 64,
        'num_layers': 3,
        'seq_length': seq_length,
        'pred_length': pred_length
    }
    
    test_results = {
        'test_loss': test_loss,
        'test_mae': test_mae
    }
    
    result_manager.save_model_and_results('STGCN', stgcn_trainer, model_config, test_results)
    
    # Extract training history from trainer
    training_history = {
        'train_losses': stgcn_trainer.train_losses,
        'val_losses': stgcn_trainer.val_losses,
        'train_maes': stgcn_trainer.train_maes,
        'val_maes': stgcn_trainer.val_maes
    }
    
    return {'STGCN': {
        'trainer': stgcn_trainer, 
        'test_results': test_results,
        'training_history': training_history,
        'model_config': model_config
    }}

def experiment_model_comparison(traffic_data, X_train, y_train, X_val, y_val, X_test, y_test, 
                              edge_index, edge_weight, processor, result_manager, device):
    """Experiment 2: Model Architecture Comparison"""
    
    print("\n" + "="*60)
    print("EXPERIMENT 2: Model Architecture Comparison with Full Export")
    print("="*60)
    
    # Initialize experiment framework with result manager
    experiment = ModelExperiment(device=device, show_progress=True, result_manager=result_manager)
    
    # Get model parameters
    num_nodes = traffic_data.shape[1]
    seq_length = processor.seq_length
    pred_length = processor.pred_length
    
    # Add different models to compare
    experiment.add_model('STGCN', STGCN, 
                        num_nodes=num_nodes, hidden_channels=64, num_layers=3,
                        seq_length=seq_length, pred_length=pred_length)
    
    experiment.add_model('GraphWaveNet', GraphWaveNet,
                        num_nodes=num_nodes, residual_channels=32, layers=1,
                        seq_length=seq_length, pred_length=pred_length)
    
    experiment.add_model('GraphAttention', GraphAttentionNetwork,
                        num_nodes=num_nodes, hidden_channels=64, num_layers=3, heads=4,
                        seq_length=seq_length, pred_length=pred_length)
    
    experiment.add_model('GraphTransformer', GraphTransformer,
                        num_nodes=num_nodes, hidden_channels=64, num_layers=3, heads=4,
                        seq_length=seq_length, pred_length=pred_length)
    
    # Run comparison experiment with comprehensive result saving
    comparison_results = experiment.run_experiment(
        X_train, y_train, X_val, y_val, X_test, y_test, 
        edge_index, edge_weight, epochs=30, batch_size=32
    )
    
    print(f"\n🎯 Model comparison completed!")
    print(f"📁 All models saved to: {result_manager.experiment_dir}")
    
    return comparison_results

def experiment_hyperparameter_tuning(traffic_data, X_train, y_train, X_val, y_val, X_test, y_test, 
                                    edge_index, edge_weight, processor, result_manager, device):
    """Experiment 3: Hyperparameter Tuning"""
    
    print("\n" + "="*60)
    print("EXPERIMENT 3: Hyperparameter Tuning")
    print("="*60)
    
    # Define hyperparameter grid
    param_grid = {
        'hidden_channels': [32, 64, 128],
        'num_layers': [2, 3, 4],
        'learning_rate': [0.001, 0.01, 0.005],
        'batch_size': [16, 32, 64]
    }
    
    # Generate all combinations
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = list(product(*values))
    param_combinations = [dict(zip(keys, combo)) for combo in combinations]
    
    print(f"   Total combinations: {len(param_combinations)}")
    
    # Sample 5 random combinations for demo
    sample_combinations = random.sample(param_combinations, min(5, len(param_combinations)))
    
    # Run experiments
    best_score = float('inf')
    best_params = None
    results = []
    
    num_nodes = traffic_data.shape[1]
    seq_length = processor.seq_length
    pred_length = processor.pred_length
    
    print(f"\n🧪 Running {len(sample_combinations)} hyperparameter experiments...")
    
    for i, params in enumerate(tqdm(sample_combinations, desc="Hyperparameter Search", unit="exp")):
        print(f"\n   🔬 Experiment {i+1}: {params}")
        
        try:
            # Create model with current parameters
            model = STGCN(
                num_nodes=num_nodes,
                hidden_channels=params['hidden_channels'],
                num_layers=params['num_layers'],
                seq_length=seq_length,
                pred_length=pred_length
            )
            
            # Train model
            trainer = TrafficGNNTrainer(model, device=device, learning_rate=params['learning_rate'], 
                                      show_progress=False)
            trainer.train(X_train, y_train, X_val, y_val, edge_index, edge_weight, 
                         epochs=20, batch_size=params['batch_size'], verbose=False)
            
            # Evaluate
            test_loss, test_mae = trainer.validate(X_test, y_test, edge_index, edge_weight, 
                                                  batch_size=params['batch_size'])
            
            results.append({
                'params': params,
                'test_loss': test_loss,
                'test_mae': test_mae
            })
            
            if test_loss < best_score:
                best_score = test_loss
                best_params = params
                best_indicator = "🎯"
            else:
                best_indicator = ""
            
            print(f"      ✅ Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f} {best_indicator}")
            
        except Exception as e:
            print(f"      ❌ Experiment {i+1} failed: {e}")
            continue
    
    print(f"\n🏆 HYPERPARAMETER SEARCH RESULTS:")
    print(f"   Best parameters: {best_params}")
    print(f"   Best test loss: {best_score:.4f}")
    
    if result_manager:
        result_manager.logger.info(f"Hyperparameter tuning completed - Best params: {best_params}, Best loss: {best_score:.4f}")
    
    return best_params, results


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Traffic GNN Training Script')
    parser.add_argument('--experiment', choices=['single', 'comparison', 'hyperparameter', 'all'], 
                       default='single', help='Type of experiment to run')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--data_path', type=str, default='./data/METR-LA.csv', help='Path to traffic data CSV')
    parser.add_argument('--dataset', choices=['METR-LA'], default='auto', 
                       help='Dataset to use (auto-detects from data_path by default)')
    parser.add_argument('--seq_length', type=int, default=12, help='Input sequence length')
    parser.add_argument('--pred_length', type=int, default=3, help='Prediction length')
    
    args = parser.parse_args()
    
    # Setup environment
    OUT_DIR, device = setup_environment()
    
    print(f"\n🚀 Starting Traffic GNN Training Script")
    print(f"   Experiment Type: {args.experiment}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch Size: {args.batch_size}")
    print(f"   Output Directory: {OUT_DIR}")
    
    # Initialize result manager
    result_manager = ComprehensiveResultManager(
        base_dir='traffic_gnn_results',
        experiment_name=f"traffic_gnn_{args.experiment}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        out_dir=OUT_DIR
    )
    
    # Load data
    print("\n📊 Loading traffic data...")
    data_loader = TrafficDataLoader()
    
    try:
        traffic_data = data_loader.load_csv_data(args.data_path)
        print(f"   ✅ Successfully loaded data: {traffic_data.shape}")
    except:
        print(f"   ⚠️  Could not load {args.data_path}, creating sample data...")
        traffic_data = data_loader.create_sample_data(n_sensors=50, n_timesteps=1000)
    
    # Build graph
    print("\n🔗 Building graph...")
    edge_index, edge_weight, graph_builder, method_used = auto_detect_dataset_and_build_graph(traffic_data)
    
    # Process temporal data
    print("\n⏰ Processing temporal sequences...")
    processor = TemporalDataProcessor(seq_length=args.seq_length, pred_length=args.pred_length, stride=1)
    X, y = processor.create_sequences(traffic_data, normalize=True)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = processor.train_test_split(X, y)
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.FloatTensor(y_val)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)
    
    # Save dataset information
    result_manager.save_dataset_info(traffic_data, processor)
    
    # Run experiments based on selection
    if args.experiment == 'single':
        results = experiment_single_model(
            traffic_data, X_train, y_train, X_val, y_val, X_test, y_test,
            edge_index, edge_weight, processor, result_manager, device, batch_size=args.batch_size
        )
    
    elif args.experiment == 'comparison':
        results = experiment_model_comparison(
            traffic_data, X_train, y_train, X_val, y_val, X_test, y_test,
            edge_index, edge_weight, processor, result_manager, device
        )
    
    elif args.experiment == 'hyperparameter':
        best_params, hp_results = experiment_hyperparameter_tuning(
            traffic_data, X_train, y_train, X_val, y_val, X_test, y_test,
            edge_index, edge_weight, processor, result_manager, device
        )
        results = {'hyperparameter_tuning': {'best_params': best_params, 'results': hp_results}}
    
    
    elif args.experiment == 'all':
        print("\n🔬 Running all experiments...")
        
        # Single model experiment
        single_results = experiment_single_model(
            traffic_data, X_train, y_train, X_val, y_val, X_test, y_test,
            edge_index, edge_weight, processor, result_manager, device
        )
        
        # Model comparison
        comparison_results = experiment_model_comparison(
            traffic_data, X_train, y_train, X_val, y_val, X_test, y_test,
            edge_index, edge_weight, processor, result_manager, device
        )
        
        # Hyperparameter tuning
        best_params, hp_results = experiment_hyperparameter_tuning(
            traffic_data, X_train, y_train, X_val, y_val, X_test, y_test,
            edge_index, edge_weight, processor, result_manager, device
        )
        
        
        results = {**single_results, **comparison_results, 
                  'hyperparameter_tuning': {'best_params': best_params, 'results': hp_results}}
    
    # Save final experiment summary
    if 'hyperparameter_tuning' not in results:
        experiment_dir = result_manager.save_experiment_summary(results)
    else:
        experiment_dir = result_manager.experiment_dir
    
    print(f"\n🎉 Experiment completed successfully!")
    print(f"📁 All results saved to: {experiment_dir}")
    print(f"📊 CSV files available in: {result_manager.dirs['csv']}")
    print(f"📈 Models saved in: {result_manager.dirs['models']}")
    print(f"📝 Logs available in: {result_manager.dirs['logs']}")
    
    return experiment_dir

if __name__ == "__main__":
    main()
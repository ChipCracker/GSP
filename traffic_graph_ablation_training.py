#!/usr/bin/env python3
"""
Graph Strategy Ablation Study - Training Script
=============================================

SLURM-compatible script for systematic evaluation of graph construction strategies
using the best-performing GraphTransformer model.

Usage:
    python traffic_graph_ablation_training.py --runs 5 --graphs all
    sbatch slurm_ablation.sh

Author: Generated for Graph Ablation Study
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
import networkx as nx

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch_geometric
from torch_geometric.nn import TransformerConv
from torch_geometric.data import Data

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
    
    print(f"🔧 Graph Ablation Study Setup:")
    print(f"   Output Directory: {OUT_DIR}")
    print(f"   PyTorch Version: {torch.__version__}")
    print(f"   PyTorch Geometric Version: {torch_geometric.__version__}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Device: {device}")
    
    return OUT_DIR, device

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =============================================================================
# DATA LOADING (Reuse from main training script)
# =============================================================================

class TrafficDataLoader:
    """Enhanced data loader for METR-LA traffic dataset"""

    def __init__(self, file_path=None, timezone='US/Pacific'):
        self.data = None
        self.sensor_ids = None
        self.timestamps = None
        self.scaler = StandardScaler()
        self.timezone = timezone

    def load_csv_data(self, file_path):
        """Load traffic data from CSV and localize timestamps"""
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        # Timezone handling
        df.index = df.index.tz_localize(
            self.timezone, ambiguous='shift_forward', nonexistent='shift_forward'
        )
        
        self.data = df
        self.sensor_ids = df.columns.tolist()
        self.timestamps = df.index
        
        print(f"📊 Loaded traffic data: {df.shape}")
        print(f"   Time range: {df.index.min()} to {df.index.max()}")
        print(f"   Sensors: {len(self.sensor_ids)}")
        print(f"   Missing values: {df.isnull().sum().sum()}")
        
        return df

    def create_sample_data(self, num_sensors=207, num_timesteps=10000):
        """Create synthetic sample data for testing"""
        print("⚠️  Creating synthetic sample data for testing")
        
        timestamps = pd.date_range('2012-03-01', periods=num_timesteps, freq='5min')
        sensor_ids = [f'sensor_{i:03d}' for i in range(num_sensors)]
        
        # Generate realistic traffic patterns
        data = np.random.exponential(2.0, (num_timesteps, num_sensors))
        data = np.maximum(0, data + np.sin(np.arange(num_timesteps)[:, np.newaxis] * 2 * np.pi / 288) * 0.5)
        
        df = pd.DataFrame(data, index=timestamps, columns=sensor_ids)
        df.index = df.index.tz_localize(self.timezone)
        
        self.data = df
        self.sensor_ids = sensor_ids
        self.timestamps = timestamps
        
        return df

# =============================================================================
# ENHANCED GRAPH BUILDER
# =============================================================================

class EnhancedGraphBuilder:
    """Enhanced graph builder with multiple construction strategies"""
    
    def __init__(self):
        self.adjacency_matrix = None
        self.edge_index = None
        self.edge_weight = None
        self.graph_properties = {}
    
    def build_knn_graph(self, data, k=8):
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
        
        # Store graph properties
        self._compute_graph_properties(adj_matrix, f'knn_{k}')
        
        print(f"   Graph created: {len(data.columns)} nodes, {len(edge_weights)} edges")
        return self.edge_index, self.edge_weight
    
    def build_correlation_graph(self, data, threshold=0.7):
        """Build correlation-based graph with threshold"""
        print(f"Building correlation graph (threshold={threshold})...")
        
        # Calculate correlation matrix
        corr_matrix = data.corr().abs()
        
        # Create adjacency matrix (threshold correlation)
        adj_matrix = (corr_matrix > threshold).astype(int)
        np.fill_diagonal(adj_matrix.values, 0)  # Remove self-loops
        
        self.adjacency_matrix = adj_matrix
        
        # Convert to edge list format
        edge_indices = np.where(adj_matrix.values == 1)
        self.edge_index = torch.tensor(np.vstack(edge_indices), dtype=torch.long)
        
        # Edge weights are correlation values
        edge_weights = corr_matrix.values[edge_indices]
        self.edge_weight = torch.tensor(edge_weights, dtype=torch.float)
        
        # Store graph properties
        self._compute_graph_properties(adj_matrix.values, f'corr_{threshold}')
        
        print(f"   Graph created: {len(data.columns)} nodes, {len(edge_weights)} edges")
        return self.edge_index, self.edge_weight
    
    def load_adjacency_matrix(self, adj_file='data/adj_mx_METR-LA.pkl'):
        """Load precomputed adjacency matrix"""
        print(f"Loading adjacency matrix from {adj_file}...")
        
        try:
            with open(adj_file, 'rb') as f:
                adj_data = pickle.load(f, encoding='latin1')
            
            sensor_ids, id_to_index, adj_matrix = adj_data
            
            # Convert to edge format with threshold
            threshold = 0.1
            adj_binary = (adj_matrix > threshold).astype(int)
            np.fill_diagonal(adj_binary, 0)
            
            edge_indices = np.where(adj_binary == 1)
            self.edge_index = torch.tensor(np.vstack(edge_indices), dtype=torch.long)
            
            edge_weights = adj_matrix[edge_indices]
            self.edge_weight = torch.tensor(edge_weights, dtype=torch.float)
            
            # Store graph properties
            self._compute_graph_properties(adj_binary, 'adjacency')
            
            print(f"   Graph loaded: {adj_matrix.shape[0]} nodes, {len(edge_weights)} edges")
            return self.edge_index, self.edge_weight
            
        except FileNotFoundError:
            print(f"⚠️  Adjacency file {adj_file} not found, creating dummy graph")
            return self._create_dummy_adjacency_graph()
    
    def _create_dummy_adjacency_graph(self):
        """Create dummy adjacency graph for testing"""
        num_nodes = 207
        # Create a sparse random graph
        edge_prob = 0.05
        adj_matrix = np.random.rand(num_nodes, num_nodes) < edge_prob
        adj_matrix = np.logical_or(adj_matrix, adj_matrix.T)  # Make symmetric
        np.fill_diagonal(adj_matrix, 0)
        
        edge_indices = np.where(adj_matrix)
        self.edge_index = torch.tensor(np.vstack(edge_indices), dtype=torch.long)
        self.edge_weight = torch.rand(len(edge_indices[0]))
        
        self._compute_graph_properties(adj_matrix.astype(int), 'adjacency_dummy')
        
        return self.edge_index, self.edge_weight
    
    def _compute_graph_properties(self, adj_matrix, graph_name):
        """Compute and store graph properties"""
        num_nodes = adj_matrix.shape[0]
        num_edges = np.sum(adj_matrix) // 2  # Undirected graph
        
        # Basic properties
        avg_degree = np.sum(adj_matrix) / num_nodes
        edge_density = num_edges / (num_nodes * (num_nodes - 1) / 2)
        
        # NetworkX for advanced properties
        G = nx.from_numpy_array(adj_matrix)
        clustering = nx.average_clustering(G)
        
        try:
            avg_path_length = nx.average_shortest_path_length(G) if nx.is_connected(G) else float('inf')
        except:
            avg_path_length = float('inf')
        
        self.graph_properties[graph_name] = {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'avg_degree': avg_degree,
            'edge_density': edge_density,
            'clustering_coefficient': clustering,
            'avg_path_length': avg_path_length,
            'is_connected': nx.is_connected(G)
        }
    
    def get_all_graph_strategies(self, data):
        """Generate all graph strategies for ablation study"""
        strategies = {}
        
        # k-NN variants
        for k in [5, 8, 10, 12]:
            edge_index, edge_weight = self.build_knn_graph(data, k)
            strategies[f'knn_{k}'] = {
                'edge_index': edge_index.clone(),
                'edge_weight': edge_weight.clone(),
                'properties': self.graph_properties[f'knn_{k}'].copy()
            }
        
        # Correlation variants
        for threshold in [0.3, 0.5, 0.7, 0.9]:
            edge_index, edge_weight = self.build_correlation_graph(data, threshold)
            strategies[f'corr_{threshold:.1f}'] = {
                'edge_index': edge_index.clone(),
                'edge_weight': edge_weight.clone(),
                'properties': self.graph_properties[f'corr_{threshold}'].copy()
            }
        
        # Adjacency matrix
        edge_index, edge_weight = self.load_adjacency_matrix()
        strategies['adjacency'] = {
            'edge_index': edge_index.clone(),
            'edge_weight': edge_weight.clone(),
            'properties': self.graph_properties['adjacency'].copy() if 'adjacency' in self.graph_properties else self.graph_properties.get('adjacency_dummy', {})
        }
        
        return strategies

# =============================================================================
# GRAPHTRANSFORMER MODEL (Best performing model)
# =============================================================================

class GraphTransformer(nn.Module):
    """Graph Transformer for traffic forecasting - Best performing model"""
    
    def __init__(self, num_nodes, in_channels=1, hidden_channels=64, out_channels=1, 
                 num_layers=3, heads=4, seq_length=12, pred_length=3, dropout=0.1):
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
# TEMPORAL DATA PROCESSOR (Reuse from main script)
# =============================================================================

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
        
        print(f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        
        return (torch.FloatTensor(X_train), torch.FloatTensor(y_train),
                torch.FloatTensor(X_val), torch.FloatTensor(y_val),
                torch.FloatTensor(X_test), torch.FloatTensor(y_test))

# =============================================================================
# TRAINING FRAMEWORK
# =============================================================================

class GraphTransformerTrainer:
    """Training framework specifically for GraphTransformer ablation study"""
    
    def __init__(self, model, device='cpu', learning_rate=0.001, weight_decay=1e-4):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.7)
        
        self.train_losses = []
        self.val_losses = []
        self.train_maes = []
        self.val_maes = []
        
    def train_epoch(self, X_train, y_train, edge_index, edge_weight, batch_size=32):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_mae = 0
        num_batches = 0
        
        # Create batches
        num_samples = X_train.size(0)
        indices = torch.randperm(num_samples)
        
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]
            
            batch_X = X_train[batch_indices].to(self.device)
            batch_y = y_train[batch_indices].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(batch_X, edge_index.to(self.device), edge_weight.to(self.device))
            
            # Calculate loss
            loss = F.mse_loss(predictions, batch_y)
            mae = F.l1_loss(predictions, batch_y)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_mae += mae.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_mae = total_mae / num_batches
        
        return avg_loss, avg_mae
    
    def validate(self, X_val, y_val, edge_index, edge_weight, batch_size=64):
        """Validation step"""
        self.model.eval()
        total_loss = 0
        total_mae = 0
        num_batches = 0
        
        with torch.no_grad():
            for start_idx in range(0, X_val.size(0), batch_size):
                end_idx = min(start_idx + batch_size, X_val.size(0))
                
                batch_X = X_val[start_idx:end_idx].to(self.device)
                batch_y = y_val[start_idx:end_idx].to(self.device)
                
                predictions = self.model(batch_X, edge_index.to(self.device), edge_weight.to(self.device))
                
                loss = F.mse_loss(predictions, batch_y)
                mae = F.l1_loss(predictions, batch_y)
                
                total_loss += loss.item()
                total_mae += mae.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_mae = total_mae / num_batches
        
        return avg_loss, avg_mae
    
    def train(self, X_train, y_train, X_val, y_val, edge_index, edge_weight, 
              epochs=50, patience=15, batch_size=32):
        """Complete training loop with early stopping"""
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            train_loss, train_mae = self.train_epoch(X_train, y_train, edge_index, edge_weight, batch_size)
            
            # Validation
            val_loss, val_mae = self.validate(X_val, y_val, edge_index, edge_weight)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_maes.append(train_mae)
            self.val_maes.append(val_mae)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model state
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            # Print progress
            if epoch % 10 == 0 or patience_counter == 0:
                print(f"Epoch {epoch:3d}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                      f"Train MAE={train_mae:.4f}, Val MAE={val_mae:.4f}")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
            
            self.scheduler.step()
        
        # Restore best model
        self.model.load_state_dict(best_model_state)
        
        return {
            'best_val_loss': best_val_loss,
            'epochs_trained': epoch + 1,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_maes': self.train_maes,
            'val_maes': self.val_maes
        }
    
    def test(self, X_test, y_test, edge_index, edge_weight, batch_size=64):
        """Test the model"""
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for start_idx in range(0, X_test.size(0), batch_size):
                end_idx = min(start_idx + batch_size, X_test.size(0))
                
                batch_X = X_test[start_idx:end_idx].to(self.device)
                batch_y = y_test[start_idx:end_idx].to(self.device)
                
                predictions = self.model(batch_X, edge_index.to(self.device), edge_weight.to(self.device))
                
                all_predictions.append(predictions.cpu())
                all_targets.append(batch_y.cpu())
        
        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)
        
        test_loss = F.mse_loss(predictions, targets).item()
        test_mae = F.l1_loss(predictions, targets).item()
        
        return {
            'test_loss': test_loss,
            'test_mae': test_mae,
            'predictions': predictions.numpy(),
            'targets': targets.numpy()
        }

# =============================================================================
# ABLATION EXPERIMENT FRAMEWORK
# =============================================================================

class AblationExperiment:
    """Framework for systematic graph strategy ablation study"""
    
    def __init__(self, output_dir, device='cpu'):
        self.output_dir = output_dir
        self.device = device
        self.results = {}
        
        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = os.path.join(output_dir, f'graph_ablation_{timestamp}')
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.experiment_dir, 'experiment.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        print(f"🔬 Ablation experiment initialized: {self.experiment_dir}")
    
    def run_experiment(self, data, graph_strategies, num_runs=5, epochs=50):
        """Run complete ablation experiment"""
        
        self.logger.info(f"Starting ablation experiment with {len(graph_strategies)} graph strategies")
        self.logger.info(f"Number of runs per strategy: {num_runs}")
        
        # Prepare data
        processor = TemporalDataProcessor(seq_length=12, pred_length=3)
        X, y = processor.create_sequences(data)
        X_train, y_train, X_val, y_val, X_test, y_test = processor.train_test_split(X, y)
        
        # Store data info
        data_info = {
            'num_sensors': data.shape[1],
            'num_timesteps': data.shape[0],
            'num_sequences': len(X),
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test)
        }
        
        all_results = {}
        
        # Run experiments for each graph strategy
        for strategy_name, strategy_info in graph_strategies.items():
            self.logger.info(f"Testing graph strategy: {strategy_name}")
            
            edge_index = strategy_info['edge_index']
            edge_weight = strategy_info['edge_weight']
            graph_properties = strategy_info['properties']
            
            strategy_results = []
            
            # Multiple runs for statistical significance
            for run_idx in range(num_runs):
                self.logger.info(f"  Run {run_idx + 1}/{num_runs}")
                
                # Set seed for reproducibility
                set_seed(42 + run_idx)
                
                # Create model
                model = GraphTransformer(
                    num_nodes=data.shape[1],
                    hidden_channels=64,
                    num_layers=3,
                    heads=4,
                    seq_length=12,
                    pred_length=3
                )
                
                # Create trainer
                trainer = GraphTransformerTrainer(model, self.device)
                
                # Train model
                training_results = trainer.train(
                    X_train, y_train, X_val, y_val, 
                    edge_index, edge_weight, 
                    epochs=epochs
                )
                
                # Test model
                test_results = trainer.test(X_test, y_test, edge_index, edge_weight)
                
                # Combine results
                run_result = {
                    'run_id': run_idx,
                    'strategy_name': strategy_name,
                    'graph_properties': graph_properties,
                    'training_results': training_results,
                    'test_results': test_results,
                    'seed': 42 + run_idx
                }
                
                strategy_results.append(run_result)
                
                self.logger.info(f"    Test MAE: {test_results['test_mae']:.4f}, Test Loss: {test_results['test_loss']:.4f}")
            
            # Compute strategy statistics
            test_maes = [r['test_results']['test_mae'] for r in strategy_results]
            test_losses = [r['test_results']['test_loss'] for r in strategy_results]
            
            strategy_summary = {
                'strategy_name': strategy_name,
                'graph_properties': graph_properties,
                'runs': strategy_results,
                'statistics': {
                    'mean_test_mae': np.mean(test_maes),
                    'std_test_mae': np.std(test_maes),
                    'mean_test_loss': np.mean(test_losses),
                    'std_test_loss': np.std(test_losses),
                    'num_runs': num_runs
                }
            }
            
            all_results[strategy_name] = strategy_summary
            
            self.logger.info(f"  {strategy_name} summary: MAE={np.mean(test_maes):.4f}±{np.std(test_maes):.4f}")
        
        # Save results
        self.results = {
            'experiment_info': {
                'timestamp': datetime.now().isoformat(),
                'device': str(self.device),
                'num_strategies': len(graph_strategies),
                'num_runs': num_runs,
                'epochs': epochs
            },
            'data_info': data_info,
            'results': all_results
        }
        
        self._save_results()
        self._create_summary()
        
        return self.results
    
    def _save_results(self):
        """Save comprehensive results"""
        
        # Save main results
        results_file = os.path.join(self.experiment_dir, 'ablation_results.json')
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save summary CSV
        summary_data = []
        for strategy_name, strategy_results in self.results['results'].items():
            stats = strategy_results['statistics']
            props = strategy_results['graph_properties']
            
            summary_data.append({
                'strategy': strategy_name,
                'mean_test_mae': stats['mean_test_mae'],
                'std_test_mae': stats['std_test_mae'],
                'mean_test_loss': stats['mean_test_loss'],
                'std_test_loss': stats['std_test_loss'],
                'num_edges': props['num_edges'],
                'avg_degree': props['avg_degree'],
                'edge_density': props['edge_density'],
                'clustering_coefficient': props['clustering_coefficient'],
                'is_connected': props['is_connected']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('mean_test_mae')
        summary_df.to_csv(os.path.join(self.experiment_dir, 'summary.csv'), index=False)
        
        # Save detailed results CSV
        detailed_data = []
        for strategy_name, strategy_results in self.results['results'].items():
            for run in strategy_results['runs']:
                detailed_data.append({
                    'strategy': strategy_name,
                    'run_id': run['run_id'],
                    'seed': run['seed'],
                    'test_mae': run['test_results']['test_mae'],
                    'test_loss': run['test_results']['test_loss'],
                    'epochs_trained': run['training_results']['epochs_trained'],
                    'best_val_loss': run['training_results']['best_val_loss']
                })
        
        detailed_df = pd.DataFrame(detailed_data)
        detailed_df.to_csv(os.path.join(self.experiment_dir, 'detailed_results.csv'), index=False)
        
        self.logger.info(f"Results saved to {self.experiment_dir}")
    
    def _create_summary(self):
        """Create experiment summary"""
        summary = []
        summary.append("GRAPH STRATEGY ABLATION STUDY SUMMARY")
        summary.append("=" * 60)
        summary.append(f"Experiment Directory: {self.experiment_dir}")
        summary.append(f"Timestamp: {self.results['experiment_info']['timestamp']}")
        summary.append(f"Device: {self.results['experiment_info']['device']}")
        summary.append(f"Number of Strategies: {self.results['experiment_info']['num_strategies']}")
        summary.append(f"Runs per Strategy: {self.results['experiment_info']['num_runs']}")
        summary.append("")
        
        # Performance ranking
        summary.append("PERFORMANCE RANKING:")
        summary.append("-" * 40)
        
        # Sort strategies by performance
        sorted_strategies = sorted(
            self.results['results'].items(),
            key=lambda x: x[1]['statistics']['mean_test_mae']
        )
        
        for rank, (strategy_name, strategy_results) in enumerate(sorted_strategies, 1):
            stats = strategy_results['statistics']
            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}."
            summary.append(f"{medal} {strategy_name:15s} - MAE: {stats['mean_test_mae']:.4f}±{stats['std_test_mae']:.4f}")
        
        summary.append("")
        summary.append("GRAPH PROPERTIES ANALYSIS:")
        summary.append("-" * 40)
        
        for strategy_name, strategy_results in sorted_strategies:
            props = strategy_results['graph_properties']
            stats = strategy_results['statistics']
            summary.append(f"{strategy_name}:")
            summary.append(f"  MAE: {stats['mean_test_mae']:.4f}±{stats['std_test_mae']:.4f}")
            summary.append(f"  Edges: {props['num_edges']}, Density: {props['edge_density']:.4f}")
            summary.append(f"  Avg Degree: {props['avg_degree']:.2f}, Clustering: {props['clustering_coefficient']:.4f}")
            summary.append("")
        
        # Save summary
        summary_text = "\n".join(summary)
        with open(os.path.join(self.experiment_dir, 'SUMMARY.txt'), 'w') as f:
            f.write(summary_text)
        
        print("\n" + summary_text)

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Graph Strategy Ablation Study')
    parser.add_argument('--runs', type=int, default=5, help='Number of runs per strategy')
    parser.add_argument('--epochs', type=int, default=50, help='Maximum epochs per run')
    parser.add_argument('--graphs', type=str, default='all', help='Graph strategies to test (all, knn, corr, adj)')
    parser.add_argument('--data', type=str, default='data/METR-LA.csv', help='Path to traffic data CSV')
    
    args = parser.parse_args()
    
    # Setup environment
    OUT_DIR, device = setup_environment()
    
    print(f"🚀 Starting Graph Strategy Ablation Study")
    print(f"   Runs per strategy: {args.runs}")
    print(f"   Max epochs: {args.epochs}")
    print(f"   Graph strategies: {args.graphs}")
    print(f"   Data file: {args.data}")
    
    # Load data
    data_loader = TrafficDataLoader()
    if os.path.exists(args.data):
        data = data_loader.load_csv_data(args.data)
    else:
        print(f"⚠️  Data file {args.data} not found, creating sample data")
        data = data_loader.create_sample_data()
    
    # Build all graph strategies
    graph_builder = EnhancedGraphBuilder()
    
    if args.graphs == 'all':
        graph_strategies = graph_builder.get_all_graph_strategies(data)
    else:
        # Parse specific strategies
        strategy_names = args.graphs.split(',')
        all_strategies = graph_builder.get_all_graph_strategies(data)
        graph_strategies = {name: all_strategies[name] for name in strategy_names if name in all_strategies}
    
    print(f"📊 Testing {len(graph_strategies)} graph strategies:")
    for name in graph_strategies.keys():
        print(f"   - {name}")
    
    # Run ablation experiment
    experiment = AblationExperiment(OUT_DIR, device)
    results = experiment.run_experiment(data, graph_strategies, args.runs, args.epochs)
    
    print(f"✅ Ablation study completed!")
    print(f"📁 Results saved to: {experiment.experiment_dir}")
    print(f"📊 Use the analysis notebook to visualize results")

if __name__ == "__main__":
    main()
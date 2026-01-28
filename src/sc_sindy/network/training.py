"""
Training utilities for Structure Network.

This module provides functions for training the Structure Network
on labeled trajectory data.
"""

import warnings
from typing import Callable, Dict, List, Tuple

import numpy as np

from .feature_extraction import extract_trajectory_features

# Check for PyTorch availability
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    from .structure_network import StructureNetwork

    def train_structure_network(
        train_data: List[Tuple[np.ndarray, np.ndarray]],
        epochs: int = 100,
        batch_size: int = 32,
        lr: float = 0.001,
        hidden_dims: List[int] = None,
        dropout: float = 0.2,
        val_split: float = 0.1,
        early_stopping_patience: int = 10,
        verbose: bool = True,
    ) -> Tuple[StructureNetwork, Dict]:
        """
        Train Structure Network on labeled trajectory data.

        Parameters
        ----------
        train_data : List[Tuple[np.ndarray, np.ndarray]]
            List of (features, structure_mask) tuples where:
            - features: Trajectory features with shape [n_features]
            - structure_mask: Binary mask with shape [n_vars * n_terms] (flattened)
        epochs : int, optional
            Number of training epochs (default: 100).
        batch_size : int, optional
            Batch size (default: 32).
        lr : float, optional
            Learning rate (default: 0.001).
        hidden_dims : List[int], optional
            Hidden layer dimensions (default: [128, 64]).
        dropout : float, optional
            Dropout rate (default: 0.2).
        val_split : float, optional
            Fraction of data for validation (default: 0.1).
        early_stopping_patience : int, optional
            Epochs without improvement before stopping (default: 10).
        verbose : bool, optional
            Whether to print training progress (default: True).

        Returns
        -------
        model : StructureNetwork
            Trained model.
        history : Dict
            Training history with 'train_loss' and 'val_loss'.

        Examples
        --------
        >>> train_data = [(features1, mask1), (features2, mask2), ...]
        >>> model, history = train_structure_network(train_data, epochs=50)
        """
        if len(train_data) == 0:
            raise ValueError("Training data is empty")

        if hidden_dims is None:
            hidden_dims = [128, 64]

        # Prepare data
        features_list, masks_list = zip(*train_data)
        X = torch.FloatTensor(np.array(features_list))
        y = torch.FloatTensor(np.array(masks_list))

        input_dim = X.shape[1]
        output_dim = y.shape[1]

        # Split into train/val
        n_samples = len(X)
        n_val = max(1, int(n_samples * val_split))
        indices = torch.randperm(n_samples)
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]

        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]

        # Create model
        model = StructureNetwork(
            input_dim=input_dim, output_dim=output_dim, hidden_dims=hidden_dims, dropout=dropout
        )

        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCELoss()

        # Create data loader
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Training loop
        history = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, y_val).item()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            if verbose and (epoch + 1) % 20 == 0:
                print(
                    f"Epoch {epoch+1}/{epochs}, "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}"
                )

            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break

        # Restore best model
        if best_state is not None:
            model.load_state_dict(best_state)

        return model, history

    def generate_training_data(
        systems: list,
        n_trajectories_per_system: int = 50,
        t_span: Tuple[float, float] = (0, 50),
        n_points: int = 5000,
        noise_levels: List[float] = None,
        dt: float = None,
        term_names: List[str] = None,
        build_library_fn: Callable = None,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate training data from dynamical systems.

        Parameters
        ----------
        systems : list
            List of DynamicalSystem instances.
        n_trajectories_per_system : int, optional
            Number of trajectories per system (default: 50).
        t_span : Tuple[float, float], optional
            Time span for integration (default: (0, 50)).
        n_points : int, optional
            Number of time points (default: 5000).
        noise_levels : List[float], optional
            Noise levels to sample from (default: [0.0, 0.05, 0.10]).
        dt : float, optional
            Time step. If None, computed from t_span and n_points.
        term_names : List[str], optional
            Library term names. Required if systems have different dimensions.
        build_library_fn : Callable, optional
            Function to build library. If None, uses build_library_2d.

        Returns
        -------
        train_data : List[Tuple[np.ndarray, np.ndarray]]
            List of (features, structure_mask) tuples.
        """
        if noise_levels is None:
            noise_levels = [0.0, 0.05, 0.10]

        t = np.linspace(t_span[0], t_span[1], n_points)
        if dt is None:
            dt = t[1] - t[0]

        train_data = []

        for system in systems:
            # Get true structure
            if term_names is None:
                # Use default library
                from ..core.library import build_library_2d

                dummy_x = np.zeros((10, system.dim))
                _, default_names = build_library_2d(dummy_x, poly_order=3)
                current_names = default_names
            else:
                current_names = term_names

            true_structure = system.get_true_structure(current_names)
            structure_flat = true_structure.flatten().astype(float)

            for _ in range(n_trajectories_per_system):
                # Random initial condition
                x0 = np.random.randn(system.dim) * 2

                # Random noise level
                noise = np.random.choice(noise_levels)

                try:
                    # Generate trajectory
                    x = system.generate_trajectory(x0, t, noise_level=noise)

                    # Check for numerical issues
                    if np.any(np.isnan(x)) or np.any(np.isinf(x)):
                        continue

                    # Trim edges
                    trim = 100
                    x_trim = x[trim:-trim]

                    # Extract features
                    features = extract_trajectory_features(x_trim, dt)

                    # Check for numerical issues in features
                    if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                        continue

                    train_data.append((features, structure_flat))

                except Exception as e:
                    warnings.warn(f"Failed to generate trajectory: {e}")
                    continue

        return train_data

    def evaluate_network(
        model: StructureNetwork,
        test_data: List[Tuple[np.ndarray, np.ndarray]],
        threshold: float = 0.5,
    ) -> Dict:
        """
        Evaluate trained Structure Network.

        Parameters
        ----------
        model : StructureNetwork
            Trained model.
        test_data : List[Tuple[np.ndarray, np.ndarray]]
            Test data as (features, structure_mask) tuples.
        threshold : float, optional
            Classification threshold (default: 0.5).

        Returns
        -------
        metrics : Dict
            Evaluation metrics including accuracy, precision, recall, F1.
        """
        features_list, masks_list = zip(*test_data)
        X = torch.FloatTensor(np.array(features_list))
        y_true = np.array(masks_list)

        model.eval()
        with torch.no_grad():
            y_pred_prob = model(X).numpy()

        y_pred = (y_pred_prob > threshold).astype(float)

        # Compute metrics
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))

        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
        }

else:

    def train_structure_network(*args, **kwargs):
        raise ImportError("PyTorch is required for training. Install with: pip install torch")

    def generate_training_data(*args, **kwargs):
        raise ImportError("PyTorch is required. Install with: pip install torch")

    def evaluate_network(*args, **kwargs):
        raise ImportError("PyTorch is required. Install with: pip install torch")

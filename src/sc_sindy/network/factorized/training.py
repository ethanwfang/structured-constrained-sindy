"""
Training utilities for factorized structure networks.

This module provides functions to train the factorized network on
mixed-dimension dynamical systems (2D, 3D, etc. combined).
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Type

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .trajectory_encoder import extract_per_variable_stats, extract_stats_with_correlations
from .term_representation import get_library_terms, count_library_terms
from .factorized_network import FactorizedStructureNetwork, FactorizedStructureNetworkV2


@dataclass
class TrainingSample:
    """Single training sample for factorized network."""

    stats: np.ndarray  # [n_vars, stats_dim]
    structure: np.ndarray  # [n_vars, n_terms] binary mask
    n_vars: int
    poly_order: int
    corr_matrix: Optional[np.ndarray] = None  # [n_vars, n_vars] pairwise correlations


def generate_training_sample(
    system,
    poly_order: int = 3,
    t_span: Tuple[float, float] = (0, 50),
    n_points: int = 5000,
    noise_level: float = 0.0,
    trim: int = 100,
    include_correlations: bool = False,
    include_spectral: bool = False,
) -> Optional[TrainingSample]:
    """
    Generate a single training sample from a dynamical system.

    Parameters
    ----------
    system : DynamicalSystem
        Instantiated dynamical system.
    poly_order : int, optional
        Maximum polynomial order for library.
    t_span : Tuple[float, float], optional
        Time span for trajectory.
    n_points : int, optional
        Number of time points.
    noise_level : float, optional
        Noise level as fraction of signal std.
    trim : int, optional
        Points to trim from start/end.
    include_correlations : bool, optional
        If True, extract pairwise correlations (default: False).
    include_spectral : bool, optional
        If True, include spectral features (autocorr_time, peak_freq,
        spectral_entropy, spectral_centroid) in statistics. Increases
        stats_dim from 8 to 12 (default: False).

    Returns
    -------
    sample : TrainingSample or None
        Training sample, or None if generation failed.
    """
    try:
        # Generate trajectory
        n_vars = system.dim
        x0 = np.random.randn(n_vars) * 2
        t = np.linspace(t_span[0], t_span[1], n_points)

        trajectory = system.generate_trajectory(x0, t, noise_level=noise_level)

        # Trim transients
        if trim > 0:
            trajectory = trajectory[trim:-trim]

        # Check for valid trajectory
        if np.any(np.isnan(trajectory)) or np.any(np.isinf(trajectory)):
            return None

        # Extract statistics (and optionally correlations)
        if include_correlations:
            stats, corr_matrix = extract_stats_with_correlations(
                trajectory, include_spectral=include_spectral
            )
        else:
            stats = extract_per_variable_stats(
                trajectory, include_spectral=include_spectral
            )
            corr_matrix = None

        # Get true structure
        term_names = get_library_terms(n_vars, poly_order)
        structure = system.get_true_structure(term_names)

        return TrainingSample(
            stats=stats,
            structure=structure,
            n_vars=n_vars,
            poly_order=poly_order,
            corr_matrix=corr_matrix,
        )

    except Exception:
        return None


def generate_mixed_training_data(
    systems_by_dim: Dict[int, List],
    n_trajectories_per_system: int = 50,
    poly_order: int = 3,
    noise_levels: List[float] = None,
    t_span: Tuple[float, float] = (0, 50),
    n_points: int = 5000,
    include_correlations: bool = False,
    include_spectral: bool = False,
) -> List[TrainingSample]:
    """
    Generate training data from systems of mixed dimensions.

    Parameters
    ----------
    systems_by_dim : Dict[int, List]
        Dictionary mapping dimension to list of system classes.
        E.g., {2: [VanDerPol, Duffing], 3: [Lorenz, Rossler]}
    n_trajectories_per_system : int, optional
        Number of trajectories per system (default: 50).
    poly_order : int, optional
        Maximum polynomial order (default: 3).
    noise_levels : List[float], optional
        Noise levels to sample from (default: [0.0, 0.05, 0.10]).
    t_span : Tuple[float, float], optional
        Time span for trajectories.
    n_points : int, optional
        Number of time points.
    include_correlations : bool, optional
        If True, extract pairwise correlations for each sample (default: False).
    include_spectral : bool, optional
        If True, include spectral features in statistics (default: False).
        Increases stats_dim from 8 to 12.

    Returns
    -------
    samples : List[TrainingSample]
        Training samples from all systems.
    """
    if noise_levels is None:
        noise_levels = [0.0, 0.05, 0.10]

    samples = []

    for dim, system_classes in systems_by_dim.items():
        for system_cls in system_classes:
            # Instantiate system
            try:
                system = system_cls()
                assert system.dim == dim
            except Exception as e:
                print(f"Warning: Could not instantiate {system_cls}: {e}")
                continue

            for _ in range(n_trajectories_per_system):
                noise_level = np.random.choice(noise_levels)

                sample = generate_training_sample(
                    system=system,
                    poly_order=poly_order,
                    t_span=t_span,
                    n_points=n_points,
                    noise_level=noise_level,
                    include_correlations=include_correlations,
                    include_spectral=include_spectral,
                )

                if sample is not None:
                    samples.append(sample)

    print(f"Generated {len(samples)} training samples")
    return samples


if TORCH_AVAILABLE:

    class MixedDimensionDataset(Dataset):
        """
        PyTorch Dataset for mixed-dimension training.

        Since samples have different dimensions, we group by (n_vars, poly_order)
        and batch within groups.
        """

        def __init__(self, samples: List[TrainingSample]):
            self.samples = samples
            # Check if any sample has correlations
            self.has_correlations = any(s.corr_matrix is not None for s in samples)

            # Group by (n_vars, poly_order) for batching
            self.groups = {}
            for i, sample in enumerate(samples):
                key = (sample.n_vars, sample.poly_order)
                if key not in self.groups:
                    self.groups[key] = []
                self.groups[key].append(i)

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            sample = self.samples[idx]
            item = {
                "stats": torch.FloatTensor(sample.stats),
                "structure": torch.FloatTensor(sample.structure),
                "n_vars": sample.n_vars,
                "poly_order": sample.poly_order,
            }
            # Include correlations if available
            if sample.corr_matrix is not None:
                item["corr_matrix"] = torch.FloatTensor(sample.corr_matrix)
            return item

    def collate_by_dimension(batch):
        """
        Custom collate function that handles mixed dimensions.

        Groups samples by (n_vars, poly_order) within the batch.
        """
        # Group by dimension
        groups = {}
        for item in batch:
            key = (item["n_vars"], item["poly_order"])
            if key not in groups:
                groups[key] = []
            groups[key].append(item)

        # Stack within groups
        result = {}
        for key, items in groups.items():
            n_vars, poly_order = key
            group_data = {
                "stats": torch.stack([item["stats"] for item in items]),
                "structure": torch.stack([item["structure"] for item in items]),
                "n_vars": n_vars,
                "poly_order": poly_order,
            }
            # Include correlations if any sample has them
            if "corr_matrix" in items[0]:
                group_data["corr_matrix"] = torch.stack(
                    [item["corr_matrix"] for item in items]
                )
            result[key] = group_data

        return result

    def train_factorized_network(
        samples: List[TrainingSample],
        model: Optional[nn.Module] = None,
        latent_dim: int = 64,
        epochs: int = 100,
        batch_size: int = 32,
        lr: float = 0.001,
        val_split: float = 0.1,
        early_stopping_patience: int = 10,
        use_v2: bool = True,
        verbose: bool = True,
        pos_weight: float = 1.0,
        seed: Optional[int] = None,
        use_relative_eq_encoder: bool = True,
        use_correlations: bool = False,
        interaction_type: str = "bilinear",
        use_eq_encoder: bool = True,
        stats_dim: Optional[int] = None,
    ) -> Tuple[nn.Module, Dict]:
        """
        Train a factorized structure network.

        Parameters
        ----------
        samples : List[TrainingSample]
            Training samples (can be mixed dimensions).
        model : nn.Module, optional
            Pre-existing model to continue training. If None, creates new model.
        latent_dim : int, optional
            Latent space dimension (default: 64).
        epochs : int, optional
            Number of training epochs (default: 100).
        batch_size : int, optional
            Batch size (default: 32).
        lr : float, optional
            Learning rate (default: 0.001).
        val_split : float, optional
            Validation split fraction (default: 0.1).
        early_stopping_patience : int, optional
            Early stopping patience (default: 10).
        use_v2 : bool, optional
            Use FactorizedStructureNetworkV2 (more efficient) (default: True).
        verbose : bool, optional
            Print training progress (default: True).
        pos_weight : float, optional
            Weight for positive class in BCE loss (default: 1.0).
            Higher values (e.g., 3.0) optimize for recall over precision.
            This is important for equation discovery where missing terms
            (false negatives) are worse than extra terms (false positives)
            since STLS can zero out false positives.
        seed : int, optional
            Random seed for reproducibility.
        use_relative_eq_encoder : bool, optional
            If True, use dimension-agnostic relative position encoder for
            equation indices (default: True).
        use_correlations : bool, optional
            If True, use pairwise correlations in trajectory encoding.
            Samples must have corr_matrix populated (default: False).
        interaction_type : str, optional
            Type of interaction between embeddings: 'bilinear' or 'additive'
            (default: 'bilinear').
        use_eq_encoder : bool, optional
            If True, use equation encoder. If False, all equations share
            the same representation (default: True).
        stats_dim : int, optional
            Number of statistics per variable. If None, inferred from samples
            (default: None). Use 8 for base statistics, 12 for spectral features.

        Returns
        -------
        model : nn.Module
            Trained model.
        history : Dict
            Training history with 'train_loss' and 'val_loss' lists.
        """
        # Set seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # Infer stats_dim from samples if not provided
        if stats_dim is None and len(samples) > 0:
            stats_dim = samples[0].stats.shape[1]
        elif stats_dim is None:
            stats_dim = 8  # Default

        # Create model if not provided
        if model is None:
            if use_v2:
                model = FactorizedStructureNetworkV2(
                    latent_dim=latent_dim,
                    use_relative_eq_encoder=use_relative_eq_encoder,
                    use_correlations=use_correlations,
                    interaction_type=interaction_type,
                    use_eq_encoder=use_eq_encoder,
                    stats_dim=stats_dim,
                )
            else:
                model = FactorizedStructureNetwork(latent_dim=latent_dim)

        # Split train/val
        n_val = int(len(samples) * val_split)
        indices = np.random.permutation(len(samples))
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]

        train_samples = [samples[i] for i in train_indices]
        val_samples = [samples[i] for i in val_indices]

        # Create datasets and dataloaders
        train_dataset = MixedDimensionDataset(train_samples)
        val_dataset = MixedDimensionDataset(val_samples)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_by_dimension,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_by_dimension,
        )

        # Loss and optimizer
        # Use weighted BCE for recall optimization
        # Higher pos_weight emphasizes not missing true positives (improves recall)
        def weighted_bce_loss(pred, target):
            """Weighted BCE loss that prioritizes recall."""
            # Standard BCE: -[y*log(p) + (1-y)*log(1-p)]
            # Weighted: -[pos_weight*y*log(p) + (1-y)*log(1-p)]
            eps = 1e-7
            pred = torch.clamp(pred, eps, 1 - eps)
            loss = -pos_weight * target * torch.log(pred) - (1 - target) * torch.log(1 - pred)
            return loss.mean()

        criterion = weighted_bce_loss if pos_weight != 1.0 else nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Training loop with timing
        history = {"train_loss": [], "val_loss": [], "epoch_times": []}
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None
        training_start_time = time.time()

        for epoch in range(epochs):
            epoch_start_time = time.time()
            # Training
            model.train()
            train_losses = []

            for batch_groups in train_loader:
                optimizer.zero_grad()
                batch_loss = 0

                for key, batch in batch_groups.items():
                    n_vars, poly_order = key
                    stats = batch["stats"]  # [batch, n_vars, stats_dim]
                    target = batch["structure"]  # [batch, n_vars, n_terms]
                    corr_matrix = batch.get("corr_matrix", None)  # [batch, n_vars, n_vars]

                    # Forward pass (with correlations if available, only for V2)
                    if use_v2 and corr_matrix is not None:
                        probs = model.forward(stats, n_vars, poly_order, corr_matrix)
                    else:
                        probs = model.forward(stats, n_vars, poly_order)

                    # Handle single sample case
                    if probs.dim() == 2:
                        probs = probs.unsqueeze(0)

                    loss = criterion(probs, target)
                    batch_loss += loss

                batch_loss.backward()
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_losses.append(batch_loss.item())

            avg_train_loss = np.mean(train_losses)
            history["train_loss"].append(avg_train_loss)

            # Validation
            model.eval()
            val_losses = []

            with torch.no_grad():
                for batch_groups in val_loader:
                    for key, batch in batch_groups.items():
                        n_vars, poly_order = key
                        stats = batch["stats"]
                        target = batch["structure"]
                        corr_matrix = batch.get("corr_matrix", None)

                        # Forward pass (with correlations if available, only for V2)
                        if use_v2 and corr_matrix is not None:
                            probs = model.forward(stats, n_vars, poly_order, corr_matrix)
                        else:
                            probs = model.forward(stats, n_vars, poly_order)
                        if probs.dim() == 2:
                            probs = probs.unsqueeze(0)

                        loss = criterion(probs, target)
                        val_losses.append(loss.item())

            avg_val_loss = np.mean(val_losses) if val_losses else avg_train_loss
            history["val_loss"].append(avg_val_loss)
            history["epoch_times"].append(time.time() - epoch_start_time)

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_state = model.state_dict().copy()
            else:
                patience_counter += 1

            if verbose and (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"Train: {avg_train_loss:.4f}, Val: {avg_val_loss:.4f}"
                )

            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break

        # Restore best model
        if best_state is not None:
            model.load_state_dict(best_state)

        # Add timing summary to history
        total_training_time = time.time() - training_start_time
        history["timing"] = {
            "total_seconds": total_training_time,
            "epochs_completed": len(history["epoch_times"]),
            "avg_epoch_seconds": np.mean(history["epoch_times"]) if history["epoch_times"] else 0,
        }

        return model, history

    def train_factorized_network_with_systems(
        systems_2d: Optional[List[Type]] = None,
        systems_3d: Optional[List[Type]] = None,
        n_trajectories_per_system: int = 50,
        poly_order: int = 3,
        latent_dim: int = 64,
        epochs: int = 100,
        **kwargs,
    ) -> Tuple[nn.Module, Dict, Dict]:
        """
        High-level training function that takes system classes directly.

        Parameters
        ----------
        systems_2d : List[Type], optional
            List of 2D dynamical system classes.
        systems_3d : List[Type], optional
            List of 3D dynamical system classes.
        n_trajectories_per_system : int, optional
            Trajectories to generate per system.
        poly_order : int, optional
            Maximum polynomial order.
        latent_dim : int, optional
            Network latent dimension.
        epochs : int, optional
            Training epochs.
        **kwargs
            Additional arguments passed to train_factorized_network.

        Returns
        -------
        model : nn.Module
            Trained model.
        history : Dict
            Training history.
        config : Dict
            Configuration including system info.
        """
        systems_by_dim = {}

        if systems_2d:
            systems_by_dim[2] = systems_2d
        if systems_3d:
            systems_by_dim[3] = systems_3d

        if not systems_by_dim:
            raise ValueError("Must provide at least one system list")

        # Generate training data
        samples = generate_mixed_training_data(
            systems_by_dim=systems_by_dim,
            n_trajectories_per_system=n_trajectories_per_system,
            poly_order=poly_order,
        )

        # Train model
        model, history = train_factorized_network(
            samples=samples,
            latent_dim=latent_dim,
            epochs=epochs,
            **kwargs,
        )

        # Build config
        config = {
            "systems_2d": [s.__name__ for s in (systems_2d or [])],
            "systems_3d": [s.__name__ for s in (systems_3d or [])],
            "n_trajectories_per_system": n_trajectories_per_system,
            "poly_order": poly_order,
            "latent_dim": latent_dim,
            "n_train_samples": len(samples),
        }

        return model, history, config

else:
    # Placeholders when PyTorch is not available
    class MixedDimensionDataset:
        """Placeholder when PyTorch is not available."""

        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required. Install with: pip install torch")

    def train_factorized_network(*args, **kwargs):
        raise ImportError("PyTorch is required. Install with: pip install torch")

    def train_factorized_network_with_systems(*args, **kwargs):
        raise ImportError("PyTorch is required. Install with: pip install torch")

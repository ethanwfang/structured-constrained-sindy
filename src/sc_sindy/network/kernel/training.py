"""
Training utilities for kernel-based structure networks.

This module provides functions to train the kernel network on
mixed-dimension dynamical systems.

Training Approach:
------------------
The kernel network is trained end-to-end using binary cross-entropy loss:

    L = BCE(k(z_traj, e_term, e_eq), y_true)

where:
- z_traj: trajectory embedding from TrajectoryKernelEncoder
- e_term: term type embedding from TermTypeEncoder
- e_eq: equation embedding from EquationEncoder
- k(...): learnable kernel function
- y_true: ground truth binary structure

This is different from traditional kernel methods (like SVM) which use
fixed kernels. Here, the kernel is parameterized by neural networks
and learned jointly with the embeddings.

The training is similar to metric learning / Siamese networks:
- Learn representations where compatible (traj, term, eq) tuples
  have high kernel values
- Incompatible tuples have low kernel values
"""

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

from ..factorized.trajectory_encoder import extract_per_variable_stats
from ..factorized.term_representation import get_library_terms
from .kernel_network import KernelStructureNetwork, KernelType


@dataclass
class KernelTrainingSample:
    """Single training sample for kernel network."""

    stats: np.ndarray        # [n_vars, stats_dim]
    structure: np.ndarray    # [n_vars, n_terms] binary mask
    n_vars: int
    poly_order: int
    system_name: str = ""    # For debugging/analysis


def generate_kernel_training_sample(
    system,
    poly_order: int = 3,
    t_span: Tuple[float, float] = (0, 50),
    n_points: int = 5000,
    noise_level: float = 0.0,
    trim: int = 100,
) -> Optional[KernelTrainingSample]:
    """
    Generate a single training sample from a dynamical system.

    Parameters
    ----------
    system : DynamicalSystem
        Instantiated dynamical system.
    poly_order : int
        Maximum polynomial order for library.
    t_span : Tuple[float, float]
        Time span for trajectory.
    n_points : int
        Number of time points.
    noise_level : float
        Noise level as fraction of signal std.
    trim : int
        Points to trim from start/end to remove transients.

    Returns
    -------
    sample : KernelTrainingSample or None
        Training sample, or None if generation failed.
    """
    try:
        # Generate trajectory
        n_vars = system.dim
        x0 = np.random.randn(n_vars) * 2
        t = np.linspace(t_span[0], t_span[1], n_points)

        trajectory = system.generate_trajectory(x0, t, noise_level=noise_level)

        # Trim transients
        if trim > 0 and len(trajectory) > 2 * trim:
            trajectory = trajectory[trim:-trim]

        # Check for valid trajectory
        if np.any(np.isnan(trajectory)) or np.any(np.isinf(trajectory)):
            return None

        # Check trajectory has reasonable values
        if np.max(np.abs(trajectory)) > 1e6:
            return None

        # Extract statistics
        stats = extract_per_variable_stats(trajectory)

        # Get true structure
        term_names = get_library_terms(n_vars, poly_order)
        structure = system.get_true_structure(term_names)

        return KernelTrainingSample(
            stats=stats,
            structure=structure,
            n_vars=n_vars,
            poly_order=poly_order,
            system_name=system.__class__.__name__,
        )

    except Exception as e:
        return None


def generate_kernel_training_data(
    systems_by_dim: Dict[int, List],
    n_trajectories_per_system: int = 50,
    poly_order: int = 3,
    noise_levels: Optional[List[float]] = None,
    t_span: Tuple[float, float] = (0, 50),
    n_points: int = 5000,
    verbose: bool = True,
) -> List[KernelTrainingSample]:
    """
    Generate training data from systems of mixed dimensions.

    Parameters
    ----------
    systems_by_dim : Dict[int, List]
        Dictionary mapping dimension to list of system classes.
    n_trajectories_per_system : int
        Number of trajectories per system.
    poly_order : int
        Maximum polynomial order.
    noise_levels : List[float], optional
        Noise levels to sample from.
    t_span : Tuple[float, float]
        Time span for trajectories.
    n_points : int
        Number of time points.
    verbose : bool
        Print progress.

    Returns
    -------
    samples : List[KernelTrainingSample]
        Training samples from all systems.
    """
    if noise_levels is None:
        noise_levels = [0.0, 0.05, 0.10]

    samples = []
    total_systems = sum(len(v) for v in systems_by_dim.values())
    processed = 0

    for dim, system_classes in systems_by_dim.items():
        for system_cls in system_classes:
            processed += 1
            # Instantiate system
            try:
                system = system_cls()
                assert system.dim == dim, f"Expected dim {dim}, got {system.dim}"
            except Exception as e:
                if verbose:
                    print(f"  Warning: Could not instantiate {system_cls.__name__}: {e}")
                continue

            n_success = 0
            for _ in range(n_trajectories_per_system):
                noise_level = np.random.choice(noise_levels)

                sample = generate_kernel_training_sample(
                    system=system,
                    poly_order=poly_order,
                    t_span=t_span,
                    n_points=n_points,
                    noise_level=noise_level,
                )

                if sample is not None:
                    samples.append(sample)
                    n_success += 1

            if verbose:
                print(f"  [{processed}/{total_systems}] {system_cls.__name__}: "
                      f"{n_success}/{n_trajectories_per_system} samples")

    if verbose:
        print(f"\nGenerated {len(samples)} training samples total")

    return samples


if TORCH_AVAILABLE:

    class KernelTrainingDataset(Dataset):
        """PyTorch Dataset for kernel network training."""

        def __init__(self, samples: List[KernelTrainingSample]):
            self.samples = samples

            # Group by (n_vars, poly_order) for efficient batching
            self.groups: Dict[Tuple[int, int], List[int]] = {}
            for i, sample in enumerate(samples):
                key = (sample.n_vars, sample.poly_order)
                if key not in self.groups:
                    self.groups[key] = []
                self.groups[key].append(i)

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            sample = self.samples[idx]
            return {
                "stats": torch.FloatTensor(sample.stats),
                "structure": torch.FloatTensor(sample.structure),
                "n_vars": sample.n_vars,
                "poly_order": sample.poly_order,
            }

    def collate_kernel_batch(batch: List[Dict]) -> Dict[Tuple[int, int], Dict]:
        """
        Custom collate function that groups samples by dimension.

        This is necessary because samples of different dimensions have
        different tensor shapes and cannot be directly stacked.
        """
        groups: Dict[Tuple[int, int], List[Dict]] = {}

        for item in batch:
            key = (item["n_vars"], item["poly_order"])
            if key not in groups:
                groups[key] = []
            groups[key].append(item)

        result = {}
        for key, items in groups.items():
            n_vars, poly_order = key
            result[key] = {
                "stats": torch.stack([item["stats"] for item in items]),
                "structure": torch.stack([item["structure"] for item in items]),
                "n_vars": n_vars,
                "poly_order": poly_order,
            }

        return result

    def train_kernel_network(
        samples: List[KernelTrainingSample],
        model: Optional[KernelStructureNetwork] = None,
        embed_dim: int = 64,
        kernel_type: KernelType = KernelType.NEURAL,
        epochs: int = 100,
        batch_size: int = 32,
        lr: float = 0.001,
        weight_decay: float = 1e-5,
        val_split: float = 0.1,
        early_stopping_patience: int = 15,
        verbose: bool = True,
    ) -> Tuple[KernelStructureNetwork, Dict]:
        """
        Train a kernel structure network.

        Parameters
        ----------
        samples : List[KernelTrainingSample]
            Training samples (can be mixed dimensions).
        model : KernelStructureNetwork, optional
            Pre-existing model to continue training. If None, creates new.
        embed_dim : int
            Embedding dimension.
        kernel_type : KernelType
            Type of kernel function.
        epochs : int
            Number of training epochs.
        batch_size : int
            Batch size.
        lr : float
            Learning rate.
        weight_decay : float
            L2 regularization.
        val_split : float
            Validation split fraction.
        early_stopping_patience : int
            Early stopping patience.
        verbose : bool
            Print training progress.

        Returns
        -------
        model : KernelStructureNetwork
            Trained model.
        history : Dict
            Training history with 'train_loss', 'val_loss', etc.
        """
        # Create model if not provided
        if model is None:
            model = KernelStructureNetwork(
                embed_dim=embed_dim,
                kernel_type=kernel_type,
            )

        # Split train/val
        n_val = max(1, int(len(samples) * val_split))
        indices = np.random.permutation(len(samples))
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]

        train_samples = [samples[i] for i in train_indices]
        val_samples = [samples[i] for i in val_indices]

        if verbose:
            print(f"Training samples: {len(train_samples)}")
            print(f"Validation samples: {len(val_samples)}")

        # Create datasets and dataloaders
        train_dataset = KernelTrainingDataset(train_samples)
        val_dataset = KernelTrainingDataset(val_samples)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_kernel_batch,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_kernel_batch,
        )

        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
        )

        # Training loop
        history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rates": [],
        }
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(epochs):
            # Clear term embedding cache each epoch (embeddings are learned)
            model.clear_cache()

            # Training phase
            model.train()
            train_losses = []

            for batch_groups in train_loader:
                optimizer.zero_grad()
                losses = []

                for key, batch in batch_groups.items():
                    n_vars, poly_order = key
                    stats = batch["stats"]      # [batch, n_vars, stats_dim]
                    target = batch["structure"] # [batch, n_vars, n_terms]

                    # Forward pass through kernel network
                    probs = model.forward(stats, n_vars, poly_order)

                    # Handle single sample case
                    if probs.dim() == 2:
                        probs = probs.unsqueeze(0)

                    # Compute loss
                    loss = criterion(probs, target)
                    losses.append(loss)

                # Sum all losses and backward
                batch_loss = sum(losses)
                batch_loss.backward()

                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                train_losses.append(batch_loss.item())

            avg_train_loss = np.mean(train_losses)
            history["train_loss"].append(avg_train_loss)

            # Validation phase
            model.eval()
            val_losses = []

            with torch.no_grad():
                for batch_groups in val_loader:
                    for key, batch in batch_groups.items():
                        n_vars, poly_order = key
                        stats = batch["stats"]
                        target = batch["structure"]

                        probs = model.forward(stats, n_vars, poly_order)
                        if probs.dim() == 2:
                            probs = probs.unsqueeze(0)

                        loss = criterion(probs, target)
                        val_losses.append(loss.item())

            avg_val_loss = np.mean(val_losses) if val_losses else avg_train_loss
            history["val_loss"].append(avg_val_loss)
            history["learning_rates"].append(optimizer.param_groups[0]["lr"])

            # Update learning rate
            scheduler.step(avg_val_loss)

            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1

            # Logging
            if verbose and (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1:3d}/{epochs} - "
                    f"Train: {avg_train_loss:.4f}, Val: {avg_val_loss:.4f}, "
                    f"LR: {optimizer.param_groups[0]['lr']:.6f}"
                )

            # Early stopping
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break

        # Restore best model
        if best_state is not None:
            model.load_state_dict(best_state)

        if verbose:
            print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")

        return model, history

    def evaluate_kernel_network(
        model: KernelStructureNetwork,
        systems_by_dim: Dict[int, List],
        poly_order: int = 3,
        n_trajectories: int = 20,
        threshold: float = 0.5,
        verbose: bool = True,
    ) -> Dict:
        """
        Evaluate kernel network on test systems.

        Parameters
        ----------
        model : KernelStructureNetwork
            Trained model.
        systems_by_dim : Dict[int, List]
            Test systems organized by dimension.
        poly_order : int
            Polynomial order.
        n_trajectories : int
            Number of test trajectories per system.
        threshold : float
            Classification threshold.
        verbose : bool
            Print results.

        Returns
        -------
        results : Dict
            Results organized by dimension and system.
        """
        model.eval()
        results = {}

        for dim, system_classes in systems_by_dim.items():
            if not system_classes:
                continue

            dim_results = []

            for system_cls in system_classes:
                try:
                    system = system_cls()
                except Exception as e:
                    if verbose:
                        print(f"  Warning: Could not instantiate {system_cls.__name__}: {e}")
                    continue

                system_metrics = []

                for _ in range(n_trajectories):
                    sample = generate_kernel_training_sample(
                        system=system,
                        poly_order=poly_order,
                        t_span=(0, 50),
                        n_points=5000,
                        noise_level=0.05,
                    )

                    if sample is None:
                        continue

                    # Get predictions
                    with torch.no_grad():
                        stats_tensor = torch.FloatTensor(sample.stats)
                        probs = model.forward(stats_tensor, dim, poly_order)
                        predictions = probs.numpy()

                    # Compute metrics
                    pred_binary = predictions > threshold
                    true_binary = sample.structure.astype(bool)

                    tp = np.sum(pred_binary & true_binary)
                    fp = np.sum(pred_binary & ~true_binary)
                    fn = np.sum(~pred_binary & true_binary)

                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

                    system_metrics.append({
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                    })

                if system_metrics:
                    avg_metrics = {
                        "precision": np.mean([m["precision"] for m in system_metrics]),
                        "recall": np.mean([m["recall"] for m in system_metrics]),
                        "f1": np.mean([m["f1"] for m in system_metrics]),
                        "n_samples": len(system_metrics),
                    }
                    dim_results.append((system_cls.__name__, avg_metrics))

                    if verbose:
                        print(f"  {system_cls.__name__}: "
                              f"F1={avg_metrics['f1']:.3f} "
                              f"(P={avg_metrics['precision']:.3f}, "
                              f"R={avg_metrics['recall']:.3f})")

            results[dim] = dim_results

        return results

else:
    # Placeholders when PyTorch is not available
    class KernelTrainingDataset:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required. Install with: pip install torch")

    def train_kernel_network(*args, **kwargs):
        raise ImportError("PyTorch is required. Install with: pip install torch")

    def evaluate_kernel_network(*args, **kwargs):
        raise ImportError("PyTorch is required. Install with: pip install torch")

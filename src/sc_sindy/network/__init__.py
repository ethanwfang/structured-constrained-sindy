"""
Neural network components for Structure-Constrained SINDy.

This module provides the Structure Network and related utilities
for learning equation structure from trajectory data.
"""

from .feature_extraction import (
    compute_feature_dimension,
    extract_features_batch,
    extract_trajectory_features,
    get_feature_names,
    normalize_features,
)
from .structure_network import (
    MultiHeadStructureNetwork,
    StructureNetwork,
    create_oracle_network_probs,
)
from .training import (
    evaluate_network,
    generate_training_data,
    train_structure_network,
)

# Check PyTorch availability
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

__all__ = [
    # Feature extraction
    "extract_trajectory_features",
    "extract_features_batch",
    "get_feature_names",
    "compute_feature_dimension",
    "normalize_features",
    # Structure Network
    "StructureNetwork",
    "MultiHeadStructureNetwork",
    "create_oracle_network_probs",
    # Training
    "train_structure_network",
    "generate_training_data",
    "evaluate_network",
    # Availability flag
    "TORCH_AVAILABLE",
]

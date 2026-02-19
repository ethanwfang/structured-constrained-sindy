"""
Factorized Structure Networks for Dimension-Agnostic Structure Prediction.

This module provides neural network architectures that can predict equation
structure for dynamical systems of any dimension, by factorizing the prediction
into trajectory encoding and term embedding components.

Key Components
--------------
FactorizedStructureNetwork, FactorizedStructureNetworkV2
    Main network architectures combining trajectory encoding with term prediction.

TermEmbedder
    Embeds polynomial terms into a latent space using structural representations.

TrajectoryEncoder, StatisticsEncoder, GRUEncoder, HybridEncoder
    Encode trajectories of any dimension into fixed-size latent vectors.

FactorizedPredictor
    High-level interface for prediction using trained models.

Example Usage
-------------
>>> from sc_sindy.network.factorized import (
...     FactorizedStructureNetworkV2,
...     train_factorized_network_with_systems,
...     FactorizedPredictor,
... )
>>> from sc_sindy.systems.oscillators import VanDerPolOscillator
>>> from sc_sindy.systems.chaotic import Lorenz
>>>
>>> # Train on mixed 2D and 3D systems
>>> model, history, config = train_factorized_network_with_systems(
...     systems_2d=[VanDerPolOscillator],
...     systems_3d=[Lorenz],
...     epochs=50,
... )
>>>
>>> # Create predictor
>>> predictor = FactorizedPredictor(model)
>>>
>>> # Predict on any dimension!
>>> import numpy as np
>>> x_2d = np.random.randn(1000, 2)
>>> probs_2d = predictor.predict(x_2d, poly_order=3)  # [2, 10]
>>>
>>> x_3d = np.random.randn(1000, 3)
>>> probs_3d = predictor.predict(x_3d, poly_order=2)  # [3, 10]
>>>
>>> x_4d = np.random.randn(1000, 4)  # Zero-shot!
>>> probs_4d = predictor.predict(x_4d, poly_order=2)  # [4, 15]
"""

# Term representation utilities
from .term_representation import (
    term_name_to_powers,
    powers_to_term_name,
    get_library_terms,
    get_library_powers,
    powers_to_tensor_index,
    count_library_terms,
    get_term_total_order,
    PowerList,
    PowerTuple,
)

# Term embedder
from .term_embedder import TermEmbedder, PositionalTermEmbedder

# Trajectory encoders
from .trajectory_encoder import (
    TrajectoryEncoder,
    StatisticsEncoder,
    GRUEncoder,
    HybridEncoder,
    extract_per_variable_stats,
)

# Main networks
from .factorized_network import (
    FactorizedStructureNetwork,
    FactorizedStructureNetworkV2,
)

# Training
from .training import (
    TrainingSample,
    generate_training_sample,
    generate_mixed_training_data,
    train_factorized_network,
    train_factorized_network_with_systems,
)

# Inference
from .inference import (
    FactorizedPredictor,
    predict_structure_for_sindy,
)


__all__ = [
    # Term representation
    "term_name_to_powers",
    "powers_to_term_name",
    "get_library_terms",
    "get_library_powers",
    "powers_to_tensor_index",
    "count_library_terms",
    "get_term_total_order",
    "PowerList",
    "PowerTuple",
    # Term embedder
    "TermEmbedder",
    "PositionalTermEmbedder",
    # Trajectory encoders
    "TrajectoryEncoder",
    "StatisticsEncoder",
    "GRUEncoder",
    "HybridEncoder",
    "extract_per_variable_stats",
    # Networks
    "FactorizedStructureNetwork",
    "FactorizedStructureNetworkV2",
    # Training
    "TrainingSample",
    "generate_training_sample",
    "generate_mixed_training_data",
    "train_factorized_network",
    "train_factorized_network_with_systems",
    # Inference
    "FactorizedPredictor",
    "predict_structure_for_sindy",
]

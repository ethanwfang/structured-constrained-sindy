"""
Kernel-based structure network for dimension-agnostic equation discovery.

This module implements a kernel approach to structure prediction where the
network learns a similarity function k(trajectory, term, equation) that
predicts whether a term should be active in an equation.

Key Components:
---------------
- KernelStructureNetwork: Main model class
- KernelType: Enum of available kernel types (linear, polynomial, rbf, neural, bilinear)
- train_kernel_network: Training function
- evaluate_kernel_network: Evaluation function

Example Usage:
--------------
>>> from sc_sindy.network.kernel import (
...     KernelStructureNetwork,
...     KernelType,
...     train_kernel_network,
...     generate_kernel_training_data,
... )
>>> from sc_sindy.evaluation.splits_factorized import get_factorized_train_systems
>>>
>>> # Generate training data
>>> train_systems = get_factorized_train_systems()
>>> samples = generate_kernel_training_data(train_systems, n_trajectories_per_system=30)
>>>
>>> # Train model with neural kernel
>>> model, history = train_kernel_network(
...     samples,
...     embed_dim=64,
...     kernel_type=KernelType.NEURAL,
...     epochs=50,
... )
>>>
>>> # Predict on new trajectory
>>> X = np.random.randn(1000, 3)  # 3D trajectory
>>> probs = model.predict(X, poly_order=3)  # [3, 20] probabilities
"""

from .kernel_network import (
    KernelStructureNetwork,
    KernelType,
    KernelFunction,
    TrajectoryKernelEncoder,
    TermTypeEncoder,
    EquationEncoder,
    TermTypeFeatures,
    extract_term_type_features,
)

from .training import (
    KernelTrainingSample,
    generate_kernel_training_sample,
    generate_kernel_training_data,
    train_kernel_network,
    evaluate_kernel_network,
)

__all__ = [
    # Main model
    "KernelStructureNetwork",
    "KernelType",
    # Components
    "KernelFunction",
    "TrajectoryKernelEncoder",
    "TermTypeEncoder",
    "EquationEncoder",
    # Term features
    "TermTypeFeatures",
    "extract_term_type_features",
    # Training
    "KernelTrainingSample",
    "generate_kernel_training_sample",
    "generate_kernel_training_data",
    "train_kernel_network",
    "evaluate_kernel_network",
]

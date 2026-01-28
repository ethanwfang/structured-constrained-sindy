"""
Structure Network for predicting equation structure.

This module provides the neural network model that predicts which
library terms should be active in the governing equations.
"""

from typing import List

import numpy as np

# Check for PyTorch availability
try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class StructureNetwork(nn.Module):
        """
        Neural network for predicting equation structure from trajectory features.

        The network takes trajectory features as input and outputs probabilities
        for each library term being active in each equation.

        Parameters
        ----------
        input_dim : int
            Dimension of input features.
        output_dim : int
            Number of output predictions (n_vars * n_terms for flattened output,
            or n_terms for per-variable prediction).
        hidden_dims : List[int], optional
            Dimensions of hidden layers (default: [128, 64]).
        dropout : float, optional
            Dropout rate (default: 0.2).

        Examples
        --------
        >>> model = StructureNetwork(input_dim=20, output_dim=20)
        >>> features = torch.randn(32, 20)  # batch of 32
        >>> probs = model(features)  # shape: [32, 20]
        """

        def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_dims: List[int] = None,
            dropout: float = 0.2,
        ):
            super().__init__()

            if hidden_dims is None:
                hidden_dims = [128, 64]

            layers = []
            prev_dim = input_dim

            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                prev_dim = hidden_dim

            layers.append(nn.Linear(prev_dim, output_dim))
            layers.append(nn.Sigmoid())

            self.network = nn.Sequential(*layers)
            self.input_dim = input_dim
            self.output_dim = output_dim

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass.

            Parameters
            ----------
            x : torch.Tensor
                Input features with shape [batch_size, input_dim].

            Returns
            -------
            probs : torch.Tensor
                Predicted probabilities with shape [batch_size, output_dim].
            """
            return self.network(x)

        def predict(self, features: np.ndarray) -> np.ndarray:
            """
            Make predictions from numpy array.

            Parameters
            ----------
            features : np.ndarray
                Input features with shape [n_samples, input_dim] or [input_dim].

            Returns
            -------
            probs : np.ndarray
                Predicted probabilities.
            """
            self.eval()
            with torch.no_grad():
                if features.ndim == 1:
                    features = features.reshape(1, -1)
                x = torch.FloatTensor(features)
                probs = self.forward(x)
                return probs.numpy()

        def save(self, path: str):
            """Save model to file."""
            torch.save(
                {
                    "state_dict": self.state_dict(),
                    "input_dim": self.input_dim,
                    "output_dim": self.output_dim,
                },
                path,
            )

        @classmethod
        def load(cls, path: str, **kwargs) -> "StructureNetwork":
            """Load model from file."""
            checkpoint = torch.load(path, weights_only=False)
            model = cls(
                input_dim=checkpoint["input_dim"], output_dim=checkpoint["output_dim"], **kwargs
            )
            model.load_state_dict(checkpoint["state_dict"])
            return model

    class MultiHeadStructureNetwork(nn.Module):
        """
        Multi-head Structure Network with separate heads for each equation.

        This variant has a shared feature extractor but separate prediction
        heads for each state variable's equation.

        Parameters
        ----------
        input_dim : int
            Dimension of input features.
        n_vars : int
            Number of state variables (equations).
        n_terms : int
            Number of library terms.
        hidden_dims : List[int], optional
            Dimensions of shared hidden layers.
        head_dims : List[int], optional
            Dimensions of per-head hidden layers.
        dropout : float, optional
            Dropout rate.
        """

        def __init__(
            self,
            input_dim: int,
            n_vars: int,
            n_terms: int,
            hidden_dims: List[int] = None,
            head_dims: List[int] = None,
            dropout: float = 0.2,
        ):
            super().__init__()

            if hidden_dims is None:
                hidden_dims = [128, 64]
            if head_dims is None:
                head_dims = [32]

            self.n_vars = n_vars
            self.n_terms = n_terms

            # Shared feature extractor
            shared_layers = []
            prev_dim = input_dim
            for hidden_dim in hidden_dims:
                shared_layers.append(nn.Linear(prev_dim, hidden_dim))
                shared_layers.append(nn.ReLU())
                shared_layers.append(nn.Dropout(dropout))
                prev_dim = hidden_dim
            self.shared = nn.Sequential(*shared_layers)

            # Per-variable heads
            self.heads = nn.ModuleList()
            for _ in range(n_vars):
                head_layers = []
                head_prev_dim = prev_dim
                for head_dim in head_dims:
                    head_layers.append(nn.Linear(head_prev_dim, head_dim))
                    head_layers.append(nn.ReLU())
                    head_layers.append(nn.Dropout(dropout))
                    head_prev_dim = head_dim
                head_layers.append(nn.Linear(head_prev_dim, n_terms))
                head_layers.append(nn.Sigmoid())
                self.heads.append(nn.Sequential(*head_layers))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass.

            Returns tensor with shape [batch_size, n_vars, n_terms].
            """
            shared_features = self.shared(x)
            outputs = [head(shared_features) for head in self.heads]
            return torch.stack(outputs, dim=1)

        def predict(self, features: np.ndarray) -> np.ndarray:
            """Make predictions from numpy array."""
            self.eval()
            with torch.no_grad():
                if features.ndim == 1:
                    features = features.reshape(1, -1)
                x = torch.FloatTensor(features)
                probs = self.forward(x)
                return probs.numpy()

else:
    # Placeholder when PyTorch is not available
    class StructureNetwork:
        """Placeholder for StructureNetwork when PyTorch is not available."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch is required for StructureNetwork. " "Install with: pip install torch"
            )

    class MultiHeadStructureNetwork:
        """Placeholder when PyTorch is not available."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch is required for MultiHeadStructureNetwork. "
                "Install with: pip install torch"
            )


def create_oracle_network_probs(
    true_structure: np.ndarray, confidence: float = 0.9, noise: float = 0.05
) -> np.ndarray:
    """
    Create oracle network probabilities from true structure.

    Useful for testing and ablation studies where ground truth is known.

    Parameters
    ----------
    true_structure : np.ndarray
        Boolean mask with shape [n_vars, n_terms].
    confidence : float, optional
        Probability for true active terms (default: 0.9).
    noise : float, optional
        Base probability for inactive terms (default: 0.05).

    Returns
    -------
    probs : np.ndarray
        Network probability matrix with shape [n_vars, n_terms].
    """
    probs = true_structure.astype(float) * confidence
    probs[~true_structure] = noise
    return probs

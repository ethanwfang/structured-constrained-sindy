"""
Kernel-Based Structure Network for dimension-agnostic equation discovery.

This module implements a kernel approach where the network learns a similarity
function between trajectory embeddings and term embeddings. The key insight is
that instead of predicting structure directly, we learn:

    k(z_traj, e_term, e_eq) → P(term active in equation | trajectory)

The "kernel" is parameterized by neural networks, making it learnable while
maintaining the theoretical properties of kernel methods.

Training:
---------
Unlike traditional SVMs that use the kernel trick with fixed kernels, we use
a "neural kernel" approach where the kernel function is learned end-to-end:

1. Forward pass: Compute k(z_traj, e_term, e_eq) for all (term, equation) pairs
2. Loss: Binary cross-entropy against ground truth structure
3. Backward pass: Update all parameters (encoders + kernel params)

This is similar to metric learning / Siamese networks, where we learn an
embedding space where similar items are close together.
"""

from typing import Dict, List, Optional, Tuple, Literal
from dataclasses import dataclass
from enum import Enum

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Import shared utilities from factorized module
from ..factorized.trajectory_encoder import extract_per_variable_stats
from ..factorized.term_representation import get_library_powers, PowerList


class KernelType(Enum):
    """Types of kernel functions available."""
    LINEAR = "linear"           # Simple dot product
    POLYNOMIAL = "polynomial"   # (x·y + c)^d
    RBF = "rbf"                 # exp(-γ||x-y||²)
    NEURAL = "neural"           # Learned nonlinear kernel
    BILINEAR = "bilinear"       # x^T W y (learnable W)


@dataclass
class TermTypeFeatures:
    """
    Abstract features for a term type (not position-specific).

    This allows the kernel to reason about term SEMANTICS rather than
    positions in a fixed library.
    """
    total_degree: int           # Sum of all powers (e.g., xy = 2, x²y = 3)
    n_variables: int            # Number of distinct variables involved
    max_power: int              # Highest power in the term
    has_self_interaction: bool  # Any power > 1 (e.g., x², x³)
    is_bilinear: bool           # Exactly 2 variables, each with power 1
    is_pure_power: bool         # Only one variable (e.g., x³)

    def to_vector(self) -> np.ndarray:
        """Convert to feature vector."""
        return np.array([
            self.total_degree,
            self.n_variables,
            self.max_power,
            float(self.has_self_interaction),
            float(self.is_bilinear),
            float(self.is_pure_power),
        ], dtype=np.float32)


def extract_term_type_features(powers: PowerList) -> TermTypeFeatures:
    """
    Extract abstract type features from a term's power representation.

    Parameters
    ----------
    powers : PowerList
        List of (var_idx, power) tuples.

    Returns
    -------
    features : TermTypeFeatures
        Abstract features describing the term type.
    """
    if len(powers) == 0:
        # Constant term
        return TermTypeFeatures(
            total_degree=0,
            n_variables=0,
            max_power=0,
            has_self_interaction=False,
            is_bilinear=False,
            is_pure_power=False,
        )

    total_degree = sum(p for _, p in powers)
    n_variables = len(powers)
    max_power = max(p for _, p in powers)
    has_self_interaction = any(p > 1 for _, p in powers)
    is_bilinear = (n_variables == 2 and all(p == 1 for _, p in powers))
    is_pure_power = (n_variables == 1)

    return TermTypeFeatures(
        total_degree=total_degree,
        n_variables=n_variables,
        max_power=max_power,
        has_self_interaction=has_self_interaction,
        is_bilinear=is_bilinear,
        is_pure_power=is_pure_power,
    )


if TORCH_AVAILABLE:

    class TrajectoryKernelEncoder(nn.Module):
        """
        Encode trajectories into a kernel-compatible representation.

        This encoder maps trajectories of any dimension to a fixed-size
        representation suitable for kernel computations.

        The encoding is designed to capture:
        1. Per-variable statistics (mean, std, etc.)
        2. Cross-variable relationships (correlations, phase differences)
        3. Global dynamics (overall energy, complexity)

        Parameters
        ----------
        embed_dim : int
            Output embedding dimension.
        stats_dim : int
            Number of statistics per variable.
        hidden_dim : int
            Hidden layer dimension.
        """

        def __init__(
            self,
            embed_dim: int = 64,
            stats_dim: int = 8,
            hidden_dim: int = 128,
        ):
            super().__init__()

            self.embed_dim = embed_dim
            self.stats_dim = stats_dim

            # Per-variable encoder (shared across all variables)
            self.var_encoder = nn.Sequential(
                nn.Linear(stats_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )

            # Cross-variable attention for capturing relationships
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=4,
                batch_first=True,
                dropout=0.1,
            )

            # Final projection to embedding space
            self.projector = nn.Sequential(
                nn.Linear(hidden_dim, embed_dim),
                nn.LayerNorm(embed_dim),
            )

        def forward(self, stats: torch.Tensor) -> torch.Tensor:
            """
            Encode trajectory statistics.

            Parameters
            ----------
            stats : torch.Tensor
                Per-variable statistics [batch, n_vars, stats_dim] or [n_vars, stats_dim].

            Returns
            -------
            embedding : torch.Tensor
                Trajectory embedding [batch, embed_dim] or [embed_dim].
            """
            single = stats.dim() == 2
            if single:
                stats = stats.unsqueeze(0)

            batch_size, n_vars, _ = stats.shape

            # Encode each variable
            stats_flat = stats.view(-1, self.stats_dim)
            var_embeds = self.var_encoder(stats_flat)
            var_embeds = var_embeds.view(batch_size, n_vars, -1)

            # Cross-variable attention
            attn_out, _ = self.cross_attention(var_embeds, var_embeds, var_embeds)

            # Mean pool and project
            pooled = attn_out.mean(dim=1)
            embedding = self.projector(pooled)

            if single:
                embedding = embedding.squeeze(0)

            return embedding

        def encode_trajectory(self, X: np.ndarray) -> torch.Tensor:
            """Encode raw trajectory numpy array."""
            stats = extract_per_variable_stats(X)
            stats_tensor = torch.FloatTensor(stats)
            return self.forward(stats_tensor)


    class TermTypeEncoder(nn.Module):
        """
        Encode term types into kernel-compatible representations.

        Unlike position-based term embedders, this encodes the SEMANTIC
        properties of terms (degree, structure, etc.) which are
        dimension-agnostic.

        Parameters
        ----------
        embed_dim : int
            Output embedding dimension.
        n_type_features : int
            Number of term type features.
        """

        def __init__(
            self,
            embed_dim: int = 64,
            n_type_features: int = 6,
            hidden_dim: int = 64,
        ):
            super().__init__()

            self.embed_dim = embed_dim
            self.n_type_features = n_type_features

            # Encode type features
            self.type_encoder = nn.Sequential(
                nn.Linear(n_type_features, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, embed_dim),
                nn.LayerNorm(embed_dim),
            )

            # Also learn embeddings for common patterns
            # These can capture patterns not in the hand-crafted features
            self.degree_embed = nn.Embedding(10, embed_dim // 4)  # degrees 0-9
            self.n_vars_embed = nn.Embedding(10, embed_dim // 4)  # 0-9 variables

            # Combine hand-crafted and learned features
            self.combiner = nn.Linear(embed_dim + embed_dim // 2, embed_dim)

        def forward(self, type_features: torch.Tensor, degrees: torch.Tensor, n_vars: torch.Tensor) -> torch.Tensor:
            """
            Encode term type features.

            Parameters
            ----------
            type_features : torch.Tensor
                Type feature vectors [n_terms, n_type_features].
            degrees : torch.Tensor
                Total degrees [n_terms].
            n_vars : torch.Tensor
                Number of variables per term [n_terms].

            Returns
            -------
            embeddings : torch.Tensor
                Term type embeddings [n_terms, embed_dim].
            """
            # Hand-crafted features
            type_embed = self.type_encoder(type_features)

            # Learned embeddings
            degree_embed = self.degree_embed(degrees.clamp(0, 9))
            n_vars_embed = self.n_vars_embed(n_vars.clamp(0, 9))

            # Combine
            combined = torch.cat([type_embed, degree_embed, n_vars_embed], dim=-1)
            return self.combiner(combined)

        def embed_terms(self, powers_list: List[PowerList]) -> torch.Tensor:
            """Embed a list of terms by their power representations."""
            type_features = []
            degrees = []
            n_vars_list = []

            for powers in powers_list:
                feat = extract_term_type_features(powers)
                type_features.append(feat.to_vector())
                degrees.append(feat.total_degree)
                n_vars_list.append(feat.n_variables)

            type_features = torch.FloatTensor(np.stack(type_features))
            degrees = torch.LongTensor(degrees)
            n_vars = torch.LongTensor(n_vars_list)

            return self.forward(type_features, degrees, n_vars)


    class EquationEncoder(nn.Module):
        """
        Encode equation indices into embeddings.

        For dimension-agnostic operation, we use relative position encoding
        rather than absolute indices.

        Parameters
        ----------
        embed_dim : int
            Output embedding dimension.
        max_eqs : int
            Maximum number of equations supported.
        """

        def __init__(self, embed_dim: int = 64, max_eqs: int = 10):
            super().__init__()

            self.embed_dim = embed_dim
            self.max_eqs = max_eqs

            # Learnable equation embeddings
            self.eq_embed = nn.Embedding(max_eqs, embed_dim)

            # Layer norm for stability
            self.norm = nn.LayerNorm(embed_dim)

        def forward(self, n_eqs: int) -> torch.Tensor:
            """
            Get embeddings for all equations.

            Parameters
            ----------
            n_eqs : int
                Number of equations (= number of state variables).

            Returns
            -------
            embeddings : torch.Tensor
                Equation embeddings [n_eqs, embed_dim].
            """
            indices = torch.arange(min(n_eqs, self.max_eqs))
            embeds = self.eq_embed(indices)
            return self.norm(embeds)


    class KernelFunction(nn.Module):
        """
        Learnable kernel function for computing compatibility scores.

        This is the core of the kernel approach: given trajectory embedding,
        term embedding, and equation embedding, compute a "kernel value"
        that represents compatibility.

        Multiple kernel types are supported:
        - linear: Simple dot product
        - polynomial: (x·y + c)^d
        - rbf: exp(-γ||x-y||²)
        - neural: Learned nonlinear function
        - bilinear: x^T W y

        Parameters
        ----------
        embed_dim : int
            Input embedding dimension.
        kernel_type : KernelType
            Type of kernel function.
        """

        def __init__(
            self,
            embed_dim: int = 64,
            kernel_type: KernelType = KernelType.NEURAL,
            hidden_dim: int = 64,
        ):
            super().__init__()

            self.embed_dim = embed_dim
            self.kernel_type = kernel_type

            if kernel_type == KernelType.LINEAR:
                # Just projection layers to align spaces
                self.traj_proj = nn.Linear(embed_dim, embed_dim)
                self.term_proj = nn.Linear(embed_dim, embed_dim)
                self.eq_proj = nn.Linear(embed_dim, embed_dim)

            elif kernel_type == KernelType.POLYNOMIAL:
                self.traj_proj = nn.Linear(embed_dim, embed_dim)
                self.term_proj = nn.Linear(embed_dim, embed_dim)
                self.eq_proj = nn.Linear(embed_dim, embed_dim)
                self.c = nn.Parameter(torch.tensor(1.0))
                self.degree = 2

            elif kernel_type == KernelType.RBF:
                self.traj_proj = nn.Linear(embed_dim, embed_dim)
                self.term_proj = nn.Linear(embed_dim, embed_dim)
                self.eq_proj = nn.Linear(embed_dim, embed_dim)
                self.log_gamma = nn.Parameter(torch.tensor(0.0))

            elif kernel_type == KernelType.BILINEAR:
                # Bilinear form: z^T W e
                self.W_term = nn.Parameter(torch.randn(embed_dim, embed_dim) * 0.01)
                self.W_eq = nn.Parameter(torch.randn(embed_dim, embed_dim) * 0.01)

            elif kernel_type == KernelType.NEURAL:
                # Learned nonlinear kernel
                self.kernel_net = nn.Sequential(
                    nn.Linear(embed_dim * 3, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim, 1),
                )

        def forward(
            self,
            z_traj: torch.Tensor,
            e_term: torch.Tensor,
            e_eq: torch.Tensor,
        ) -> torch.Tensor:
            """
            Compute kernel values.

            Parameters
            ----------
            z_traj : torch.Tensor
                Trajectory embedding [batch, embed_dim].
            e_term : torch.Tensor
                Term embeddings [n_terms, embed_dim].
            e_eq : torch.Tensor
                Equation embeddings [n_eqs, embed_dim].

            Returns
            -------
            kernel_values : torch.Tensor
                Compatibility scores [batch, n_eqs, n_terms].
            """
            batch_size = z_traj.shape[0]
            n_terms = e_term.shape[0]
            n_eqs = e_eq.shape[0]

            if self.kernel_type == KernelType.LINEAR:
                # Project to shared space
                z = self.traj_proj(z_traj)  # [batch, dim]
                t = self.term_proj(e_term)   # [n_terms, dim]
                e = self.eq_proj(e_eq)       # [n_eqs, dim]

                # Compute: sum_d z_d * t_d * e_d
                # Expand: [batch, 1, 1, dim] * [1, 1, n_terms, dim] * [1, n_eqs, 1, dim]
                z = z[:, None, None, :]
                t = t[None, None, :, :]
                e = e[None, :, None, :]

                # Element-wise product and sum over embedding dim
                values = (z * t * e).sum(dim=-1)  # [batch, n_eqs, n_terms]

            elif self.kernel_type == KernelType.POLYNOMIAL:
                z = self.traj_proj(z_traj)
                t = self.term_proj(e_term)
                e = self.eq_proj(e_eq)

                z = z[:, None, None, :]
                t = t[None, None, :, :]
                e = e[None, :, None, :]

                dot = (z * t * e).sum(dim=-1)
                values = (dot + self.c) ** self.degree

            elif self.kernel_type == KernelType.RBF:
                z = self.traj_proj(z_traj)  # [batch, dim]
                t = self.term_proj(e_term)   # [n_terms, dim]
                e = self.eq_proj(e_eq)       # [n_eqs, dim]

                # Compute combined embedding for term+eq
                # Then compute RBF distance to trajectory
                # [batch, n_eqs, n_terms, dim]
                combined = t[None, None, :, :] + e[None, :, None, :]
                z_expanded = z[:, None, None, :]

                sq_dist = ((z_expanded - combined) ** 2).sum(dim=-1)
                gamma = torch.exp(self.log_gamma)
                values = torch.exp(-gamma * sq_dist)

            elif self.kernel_type == KernelType.BILINEAR:
                # z^T W_term t * z^T W_eq e
                zt = torch.matmul(z_traj, self.W_term)  # [batch, dim]
                ze = torch.matmul(z_traj, self.W_eq)    # [batch, dim]

                # [batch, n_terms] and [batch, n_eqs]
                score_t = torch.matmul(zt, e_term.T)
                score_e = torch.matmul(ze, e_eq.T)

                # Combine: [batch, n_eqs, n_terms]
                values = score_e[:, :, None] * score_t[:, None, :]

            elif self.kernel_type == KernelType.NEURAL:
                # Concatenate all embeddings and pass through network
                # [batch, 1, 1, dim], [1, 1, n_terms, dim], [1, n_eqs, 1, dim]
                z = z_traj[:, None, None, :].expand(-1, n_eqs, n_terms, -1)
                t = e_term[None, None, :, :].expand(batch_size, n_eqs, -1, -1)
                e = e_eq[None, :, None, :].expand(batch_size, -1, n_terms, -1)

                # Concatenate: [batch, n_eqs, n_terms, dim*3]
                concat = torch.cat([z, t, e], dim=-1)

                # Apply kernel network
                values = self.kernel_net(concat).squeeze(-1)

            return values


    class KernelStructureNetwork(nn.Module):
        """
        Kernel-based structure network for dimension-agnostic prediction.

        This network learns a kernel function k(trajectory, term, equation)
        that predicts whether a term should be active in a given equation.

        The kernel approach provides:
        1. Dimension-agnostic operation (single model for all dimensions)
        2. Interpretable similarity-based predictions
        3. Theoretical connection to SVMs and kernel methods
        4. End-to-end trainable with BCE loss

        Parameters
        ----------
        embed_dim : int
            Embedding dimension for all components.
        kernel_type : KernelType
            Type of kernel function to use.
        stats_dim : int
            Number of statistics per variable.

        Examples
        --------
        >>> model = KernelStructureNetwork(embed_dim=64)
        >>> # Works on any dimension
        >>> X_2d = np.random.randn(1000, 2)
        >>> probs_2d = model.predict(X_2d, poly_order=3)  # [2, 10]
        >>> X_3d = np.random.randn(1000, 3)
        >>> probs_3d = model.predict(X_3d, poly_order=3)  # [3, 20]
        """

        def __init__(
            self,
            embed_dim: int = 64,
            kernel_type: KernelType = KernelType.NEURAL,
            stats_dim: int = 8,
            hidden_dim: int = 128,
        ):
            super().__init__()

            self.embed_dim = embed_dim
            self.kernel_type = kernel_type

            # Encoders
            self.traj_encoder = TrajectoryKernelEncoder(
                embed_dim=embed_dim,
                stats_dim=stats_dim,
                hidden_dim=hidden_dim,
            )
            self.term_encoder = TermTypeEncoder(
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
            )
            self.eq_encoder = EquationEncoder(
                embed_dim=embed_dim,
            )

            # Kernel function
            self.kernel = KernelFunction(
                embed_dim=embed_dim,
                kernel_type=kernel_type,
                hidden_dim=hidden_dim,
            )

            # Cache for term embeddings (since they don't change for a given library)
            self._term_cache: Dict[Tuple[int, int], torch.Tensor] = {}

        def _get_term_embeddings(self, n_vars: int, poly_order: int) -> torch.Tensor:
            """Get or compute term embeddings for a library.

            Note: During training, we always recompute to allow gradient flow.
            During eval, we can cache for efficiency.
            """
            key = (n_vars, poly_order)

            # Always recompute during training (needed for gradient flow)
            # Use cache only during eval for efficiency
            if self.training or key not in self._term_cache:
                powers_list = get_library_powers(n_vars, poly_order)
                embeddings = self.term_encoder.embed_terms(powers_list)

                # Only cache during eval mode
                if not self.training:
                    self._term_cache[key] = embeddings
                return embeddings

            return self._term_cache[key]

        def clear_cache(self):
            """Clear term embedding cache."""
            self._term_cache.clear()

        def forward(
            self,
            stats: torch.Tensor,
            n_vars: int,
            poly_order: int,
        ) -> torch.Tensor:
            """
            Forward pass: predict structure probabilities.

            Parameters
            ----------
            stats : torch.Tensor
                Per-variable statistics [batch, n_vars, stats_dim] or [n_vars, stats_dim].
            n_vars : int
                Number of state variables.
            poly_order : int
                Maximum polynomial order.

            Returns
            -------
            probs : torch.Tensor
                Structure probabilities [batch, n_vars, n_terms] or [n_vars, n_terms].
            """
            single = stats.dim() == 2
            if single:
                stats = stats.unsqueeze(0)

            # Encode trajectory
            z_traj = self.traj_encoder(stats)  # [batch, embed_dim]

            # Get term embeddings
            e_term = self._get_term_embeddings(n_vars, poly_order)  # [n_terms, embed_dim]

            # Get equation embeddings
            e_eq = self.eq_encoder(n_vars)  # [n_vars, embed_dim]

            # Compute kernel values
            kernel_values = self.kernel(z_traj, e_term, e_eq)  # [batch, n_vars, n_terms]

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(kernel_values)

            if single:
                probs = probs.squeeze(0)

            return probs

        def predict(self, X: np.ndarray, poly_order: int = 3) -> np.ndarray:
            """
            Predict structure from raw trajectory.

            Parameters
            ----------
            X : np.ndarray
                Trajectory with shape [T, n_vars].
            poly_order : int
                Maximum polynomial order.

            Returns
            -------
            probs : np.ndarray
                Structure probabilities [n_vars, n_terms].
            """
            self.eval()
            with torch.no_grad():
                stats = extract_per_variable_stats(X)
                stats_tensor = torch.FloatTensor(stats)
                n_vars = X.shape[1]
                probs = self.forward(stats_tensor, n_vars, poly_order)
                return probs.numpy()

        def save(self, path: str):
            """Save model to file."""
            torch.save({
                'state_dict': self.state_dict(),
                'embed_dim': self.embed_dim,
                'kernel_type': self.kernel_type.value,
            }, path)

        @classmethod
        def load(cls, path: str) -> 'KernelStructureNetwork':
            """Load model from file."""
            checkpoint = torch.load(path, weights_only=False)
            model = cls(
                embed_dim=checkpoint['embed_dim'],
                kernel_type=KernelType(checkpoint['kernel_type']),
            )
            model.load_state_dict(checkpoint['state_dict'])
            return model


else:
    # Placeholders when PyTorch is not available
    class KernelStructureNetwork:
        """Placeholder when PyTorch is not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch is required for KernelStructureNetwork. "
                "Install with: pip install torch"
            )

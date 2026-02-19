"""
Trajectory Encoder for factorized structure networks.

This module provides neural network components to encode dynamical system
trajectories into a fixed-size latent space, regardless of the input dimension.
"""

from typing import Optional

import numpy as np
from scipy.stats import kurtosis, skew

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def extract_per_variable_stats(x: np.ndarray) -> np.ndarray:
    """
    Extract statistical features from each variable of a trajectory.

    Parameters
    ----------
    x : np.ndarray
        Trajectory with shape [T, n_vars].

    Returns
    -------
    stats : np.ndarray
        Statistics with shape [n_vars, n_stats] where n_stats=8.
    """
    n_vars = x.shape[1]
    stats_per_var = 8

    stats = np.zeros((n_vars, stats_per_var))

    for i in range(n_vars):
        xi = x[:, i]
        stats[i, 0] = np.mean(xi)
        stats[i, 1] = np.std(xi)
        stats[i, 2] = skew(xi)
        stats[i, 3] = kurtosis(xi)
        stats[i, 4] = np.mean(xi**2)  # energy
        stats[i, 5] = np.max(xi) - np.min(xi)  # range
        stats[i, 6] = np.median(xi)
        stats[i, 7] = np.mean(np.abs(np.diff(xi)))  # avg derivative magnitude

    # Handle NaN/Inf
    stats = np.nan_to_num(stats, nan=0.0, posinf=1e6, neginf=-1e6)

    return stats


def extract_pairwise_correlations(x: np.ndarray) -> np.ndarray:
    """
    Extract pairwise correlation matrix from trajectory.

    This captures cross-variable interactions that per-variable statistics miss.
    For example, in predator-prey systems, strong negative correlation between
    predator and prey populations is a key feature.

    Parameters
    ----------
    x : np.ndarray
        Trajectory with shape [T, n_vars].

    Returns
    -------
    corr_matrix : np.ndarray
        Pairwise correlation matrix with shape [n_vars, n_vars].
        Values are in [-1, 1].
    """
    # Compute correlation matrix
    corr_matrix = np.corrcoef(x.T)

    # Handle NaN (can occur if a variable has zero variance)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    return corr_matrix


def extract_stats_with_correlations(x: np.ndarray) -> tuple:
    """
    Extract both per-variable stats and pairwise correlations.

    Parameters
    ----------
    x : np.ndarray
        Trajectory with shape [T, n_vars].

    Returns
    -------
    stats : np.ndarray
        Per-variable statistics with shape [n_vars, 8].
    corr_matrix : np.ndarray
        Pairwise correlation matrix with shape [n_vars, n_vars].
    """
    stats = extract_per_variable_stats(x)
    corr_matrix = extract_pairwise_correlations(x)
    return stats, corr_matrix


if TORCH_AVAILABLE:

    class StatisticsEncoder(nn.Module):
        """
        Encode trajectory using per-variable statistics.

        This is a simple but effective approach that:
        1. Computes fixed statistics for each variable
        2. Processes each variable's stats through a shared MLP
        3. Aggregates across variables (mean pooling)
        4. Projects to final latent dimension

        Works for any input dimension because it processes variables independently.

        Parameters
        ----------
        latent_dim : int
            Output latent dimension.
        stats_dim : int, optional
            Number of statistics per variable (default: 8).
        hidden_dim : int, optional
            Hidden layer dimension (default: 64).
        """

        def __init__(
            self,
            latent_dim: int = 64,
            stats_dim: int = 8,
            hidden_dim: int = 64,
        ):
            super().__init__()

            self.latent_dim = latent_dim
            self.stats_dim = stats_dim

            # Input normalization layer
            self.input_norm = nn.LayerNorm(stats_dim)

            # Shared MLP for processing each variable's statistics
            self.var_encoder = nn.Sequential(
                nn.Linear(stats_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )

            # Final projection after aggregation
            self.projector = nn.Sequential(
                nn.Linear(hidden_dim, latent_dim),
                nn.ReLU(),
                nn.Linear(latent_dim, latent_dim),
            )

        def forward(self, stats: torch.Tensor) -> torch.Tensor:
            """
            Encode pre-computed statistics.

            Parameters
            ----------
            stats : torch.Tensor
                Per-variable statistics with shape [batch, n_vars, stats_dim]
                or [n_vars, stats_dim] for single sample.

            Returns
            -------
            latent : torch.Tensor
                Latent encoding with shape [batch, latent_dim] or [latent_dim].
            """
            single_sample = stats.dim() == 2
            if single_sample:
                stats = stats.unsqueeze(0)

            batch_size, n_vars, _ = stats.shape

            # Normalize input statistics to handle large values
            # Reshape to [batch * n_vars, stats_dim]
            stats_flat = stats.view(-1, self.stats_dim)
            stats_normalized = self.input_norm(stats_flat)

            # Process each variable through shared encoder
            var_embeds = self.var_encoder(stats_normalized)

            # Reshape back to [batch, n_vars, hidden_dim]
            var_embeds = var_embeds.view(batch_size, n_vars, -1)

            # Aggregate across variables (mean pooling)
            aggregated = var_embeds.mean(dim=1)  # [batch, hidden_dim]

            # Project to latent space
            latent = self.projector(aggregated)

            if single_sample:
                latent = latent.squeeze(0)

            return latent

        def encode_trajectory(self, x: np.ndarray) -> torch.Tensor:
            """
            Encode a trajectory from numpy array.

            Parameters
            ----------
            x : np.ndarray
                Trajectory with shape [T, n_vars].

            Returns
            -------
            latent : torch.Tensor
                Latent encoding with shape [latent_dim].
            """
            stats = extract_per_variable_stats(x)
            stats_tensor = torch.FloatTensor(stats)
            return self.forward(stats_tensor)

    class StatisticsEncoderWithCorrelations(nn.Module):
        """
        Encode trajectory using per-variable statistics AND pairwise correlations.

        This extends StatisticsEncoder by incorporating cross-variable information
        through the correlation matrix. This helps capture interactions like:
        - Predator-prey negative correlations
        - Coupled oscillator phase relationships
        - Chaotic system variable dependencies

        Architecture:
        1. Process per-variable stats through shared MLP (like StatisticsEncoder)
        2. Process correlation row through correlation encoder
        3. Combine via attention: correlations modulate how variables interact
        4. Aggregate and project to latent space

        Parameters
        ----------
        latent_dim : int
            Output latent dimension.
        stats_dim : int, optional
            Number of statistics per variable (default: 8).
        hidden_dim : int, optional
            Hidden layer dimension (default: 64).
        use_correlation_attention : bool, optional
            If True, use correlations as attention weights. If False, concatenate
            correlation features (default: True).
        """

        def __init__(
            self,
            latent_dim: int = 64,
            stats_dim: int = 8,
            hidden_dim: int = 64,
            use_correlation_attention: bool = True,
        ):
            super().__init__()

            self.latent_dim = latent_dim
            self.stats_dim = stats_dim
            self.hidden_dim = hidden_dim
            self.use_correlation_attention = use_correlation_attention

            # Input normalization layer
            self.input_norm = nn.LayerNorm(stats_dim)

            # Shared MLP for processing each variable's statistics
            self.var_encoder = nn.Sequential(
                nn.Linear(stats_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )

            if use_correlation_attention:
                # Use correlations to weight variable interactions
                # Process correlation values to get attention logits
                self.corr_attention = nn.Sequential(
                    nn.Linear(1, 8),
                    nn.ReLU(),
                    nn.Linear(8, 1),
                )
            else:
                # Encode correlation row as additional per-variable features
                # Use a small network that processes pairs and aggregates
                self.corr_pair_encoder = nn.Sequential(
                    nn.Linear(2, 16),  # [corr_ij, self_flag]
                    nn.ReLU(),
                    nn.Linear(16, 8),
                )
                # Project combined features
                self.combined_proj = nn.Linear(hidden_dim + 8, hidden_dim)

            # Final projection after aggregation
            self.projector = nn.Sequential(
                nn.Linear(hidden_dim, latent_dim),
                nn.ReLU(),
                nn.Linear(latent_dim, latent_dim),
            )

        def forward(
            self,
            stats: torch.Tensor,
            corr_matrix: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """
            Encode pre-computed statistics and correlation matrix.

            Parameters
            ----------
            stats : torch.Tensor
                Per-variable statistics with shape [batch, n_vars, stats_dim]
                or [n_vars, stats_dim] for single sample.
            corr_matrix : torch.Tensor, optional
                Pairwise correlation matrix with shape [batch, n_vars, n_vars]
                or [n_vars, n_vars]. If None, uses identity (no cross-correlation).

            Returns
            -------
            latent : torch.Tensor
                Latent encoding with shape [batch, latent_dim] or [latent_dim].
            """
            single_sample = stats.dim() == 2
            if single_sample:
                stats = stats.unsqueeze(0)
                if corr_matrix is not None:
                    corr_matrix = corr_matrix.unsqueeze(0)

            batch_size, n_vars, _ = stats.shape

            # Default correlation matrix is identity
            if corr_matrix is None:
                corr_matrix = torch.eye(n_vars, device=stats.device)
                corr_matrix = corr_matrix.unsqueeze(0).expand(batch_size, -1, -1)

            # Normalize input statistics
            stats_flat = stats.view(-1, self.stats_dim)
            stats_normalized = self.input_norm(stats_flat)

            # Process each variable through shared encoder
            var_embeds = self.var_encoder(stats_normalized)
            var_embeds = var_embeds.view(batch_size, n_vars, self.hidden_dim)

            if self.use_correlation_attention:
                # Use correlations as attention weights for aggregation
                # Process correlation matrix to get attention logits
                # Shape: [batch, n_vars, n_vars] -> attention weights
                corr_flat = corr_matrix.unsqueeze(-1)  # [batch, n_vars, n_vars, 1]
                attn_logits = self.corr_attention(corr_flat).squeeze(-1)  # [batch, n_vars, n_vars]

                # Softmax along last dim to get attention weights
                attn_weights = torch.softmax(attn_logits, dim=-1)  # [batch, n_vars, n_vars]

                # Weighted aggregation: each variable attends to others based on correlation
                # var_embeds: [batch, n_vars, hidden]
                # attn_weights: [batch, n_vars, n_vars]
                # Result: [batch, n_vars, hidden]
                attended = torch.bmm(attn_weights, var_embeds)

                # Mean over variables for final aggregation
                aggregated = attended.mean(dim=1)  # [batch, hidden]
            else:
                # Concatenate correlation features to each variable
                # For each variable i, aggregate its correlations with all other variables
                corr_features_list = []
                for i in range(n_vars):
                    # Get correlations for variable i
                    corr_row = corr_matrix[:, i, :]  # [batch, n_vars]

                    # Create input pairs: [corr_ij, is_self]
                    is_self = torch.zeros(batch_size, n_vars, device=stats.device)
                    is_self[:, i] = 1.0

                    pair_input = torch.stack([corr_row, is_self], dim=-1)  # [batch, n_vars, 2]

                    # Encode pairs and mean-pool
                    pair_encoded = self.corr_pair_encoder(pair_input)  # [batch, n_vars, 8]
                    corr_feature = pair_encoded.mean(dim=1)  # [batch, 8]
                    corr_features_list.append(corr_feature)

                corr_features = torch.stack(corr_features_list, dim=1)  # [batch, n_vars, 8]

                # Combine with var embeddings
                combined = torch.cat([var_embeds, corr_features], dim=-1)  # [batch, n_vars, hidden+8]
                combined = self.combined_proj(combined)  # [batch, n_vars, hidden]

                # Mean pooling
                aggregated = combined.mean(dim=1)  # [batch, hidden]

            # Project to latent space
            latent = self.projector(aggregated)

            if single_sample:
                latent = latent.squeeze(0)

            return latent

        def encode_trajectory(self, x: np.ndarray) -> torch.Tensor:
            """
            Encode a trajectory from numpy array, extracting correlations.

            Parameters
            ----------
            x : np.ndarray
                Trajectory with shape [T, n_vars].

            Returns
            -------
            latent : torch.Tensor
                Latent encoding with shape [latent_dim].
            """
            stats, corr_matrix = extract_stats_with_correlations(x)
            stats_tensor = torch.FloatTensor(stats)
            corr_tensor = torch.FloatTensor(corr_matrix)
            return self.forward(stats_tensor, corr_tensor)

    class GRUEncoder(nn.Module):
        """
        Encode trajectory using per-variable GRU and aggregation.

        This approach:
        1. Processes each variable's time series through a shared GRU
        2. Takes the final hidden state as the variable embedding
        3. Aggregates variable embeddings via attention or mean pooling
        4. Projects to final latent dimension

        Captures temporal dynamics but is slower than StatisticsEncoder.

        Parameters
        ----------
        latent_dim : int
            Output latent dimension.
        hidden_dim : int, optional
            GRU hidden dimension (default: 64).
        n_layers : int, optional
            Number of GRU layers (default: 2).
        use_attention : bool, optional
            Use attention for aggregation instead of mean pooling (default: False).
        """

        def __init__(
            self,
            latent_dim: int = 64,
            hidden_dim: int = 64,
            n_layers: int = 2,
            use_attention: bool = False,
        ):
            super().__init__()

            self.latent_dim = latent_dim
            self.hidden_dim = hidden_dim
            self.use_attention = use_attention

            # Shared GRU for processing each variable's time series
            self.gru = nn.GRU(
                input_size=1,  # Single variable at a time
                hidden_size=hidden_dim,
                num_layers=n_layers,
                batch_first=True,
                dropout=0.1 if n_layers > 1 else 0,
            )

            # Optional attention for aggregation
            if use_attention:
                self.attention = nn.MultiheadAttention(
                    embed_dim=hidden_dim, num_heads=4, batch_first=True
                )

            # Final projection
            self.projector = nn.Sequential(
                nn.Linear(hidden_dim, latent_dim),
                nn.ReLU(),
                nn.Linear(latent_dim, latent_dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Encode trajectory tensor.

            Parameters
            ----------
            x : torch.Tensor
                Trajectory with shape [batch, T, n_vars] or [T, n_vars].

            Returns
            -------
            latent : torch.Tensor
                Latent encoding with shape [batch, latent_dim] or [latent_dim].
            """
            single_sample = x.dim() == 2
            if single_sample:
                x = x.unsqueeze(0)

            batch_size, T, n_vars = x.shape

            # Process each variable through shared GRU
            var_embeds = []
            for i in range(n_vars):
                xi = x[:, :, i : i + 1]  # [batch, T, 1]
                _, h_n = self.gru(xi)  # h_n: [n_layers, batch, hidden]
                var_embed = h_n[-1]  # Last layer: [batch, hidden]
                var_embeds.append(var_embed)

            # Stack: [batch, n_vars, hidden]
            var_embeds = torch.stack(var_embeds, dim=1)

            # Aggregate across variables
            if self.use_attention:
                # Self-attention over variables
                attn_out, _ = self.attention(var_embeds, var_embeds, var_embeds)
                aggregated = attn_out.mean(dim=1)  # [batch, hidden]
            else:
                # Mean pooling
                aggregated = var_embeds.mean(dim=1)  # [batch, hidden]

            # Project to latent space
            latent = self.projector(aggregated)

            if single_sample:
                latent = latent.squeeze(0)

            return latent

        def encode_trajectory(self, x: np.ndarray) -> torch.Tensor:
            """
            Encode a trajectory from numpy array.

            Parameters
            ----------
            x : np.ndarray
                Trajectory with shape [T, n_vars].

            Returns
            -------
            latent : torch.Tensor
                Latent encoding with shape [latent_dim].
            """
            x_tensor = torch.FloatTensor(x)
            return self.forward(x_tensor)

    class HybridEncoder(nn.Module):
        """
        Hybrid encoder combining statistics and GRU.

        This approach:
        1. Computes statistics for fast global features
        2. Runs GRU on subsampled trajectory for temporal patterns
        3. Concatenates and projects both

        Balances speed and expressiveness.

        Parameters
        ----------
        latent_dim : int
            Output latent dimension.
        stats_dim : int, optional
            Number of statistics per variable (default: 8).
        hidden_dim : int, optional
            Hidden layer dimension (default: 64).
        gru_subsample : int, optional
            Subsample trajectory by this factor for GRU (default: 10).
        """

        def __init__(
            self,
            latent_dim: int = 64,
            stats_dim: int = 8,
            hidden_dim: int = 64,
            gru_subsample: int = 10,
        ):
            super().__init__()

            self.latent_dim = latent_dim
            self.stats_dim = stats_dim
            self.gru_subsample = gru_subsample

            # Statistics branch
            self.stats_encoder = StatisticsEncoder(
                latent_dim=hidden_dim, stats_dim=stats_dim, hidden_dim=hidden_dim
            )

            # GRU branch
            self.gru_encoder = GRUEncoder(
                latent_dim=hidden_dim, hidden_dim=hidden_dim // 2, n_layers=1
            )

            # Final projection combining both
            self.projector = nn.Sequential(
                nn.Linear(hidden_dim * 2, latent_dim),
                nn.ReLU(),
                nn.Linear(latent_dim, latent_dim),
            )

        def forward(
            self, stats: torch.Tensor, x_subsampled: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
            """
            Encode using both statistics and trajectory.

            Parameters
            ----------
            stats : torch.Tensor
                Per-variable statistics with shape [batch, n_vars, stats_dim]
                or [n_vars, stats_dim].
            x_subsampled : torch.Tensor, optional
                Subsampled trajectory for GRU. If None, zeros are used.

            Returns
            -------
            latent : torch.Tensor
                Latent encoding with shape [batch, latent_dim] or [latent_dim].
            """
            single_sample = stats.dim() == 2
            if single_sample:
                stats = stats.unsqueeze(0)
                if x_subsampled is not None:
                    x_subsampled = x_subsampled.unsqueeze(0)

            # Statistics branch
            stats_embed = self.stats_encoder(stats)  # [batch, hidden]

            # GRU branch
            if x_subsampled is not None:
                gru_embed = self.gru_encoder(x_subsampled)  # [batch, hidden]
            else:
                gru_embed = torch.zeros_like(stats_embed)

            # Combine
            combined = torch.cat([stats_embed, gru_embed], dim=-1)
            latent = self.projector(combined)

            if single_sample:
                latent = latent.squeeze(0)

            return latent

        def encode_trajectory(self, x: np.ndarray) -> torch.Tensor:
            """
            Encode a trajectory from numpy array.

            Parameters
            ----------
            x : np.ndarray
                Trajectory with shape [T, n_vars].

            Returns
            -------
            latent : torch.Tensor
                Latent encoding with shape [latent_dim].
            """
            # Extract statistics
            stats = extract_per_variable_stats(x)
            stats_tensor = torch.FloatTensor(stats)

            # Subsample trajectory for GRU
            x_sub = x[:: self.gru_subsample]
            x_sub_tensor = torch.FloatTensor(x_sub)

            return self.forward(stats_tensor, x_sub_tensor)

    # Alias for the recommended encoder
    TrajectoryEncoder = StatisticsEncoder

else:
    # Placeholders when PyTorch is not available
    class StatisticsEncoder:
        """Placeholder when PyTorch is not available."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch is required for StatisticsEncoder. Install with: pip install torch"
            )

    class GRUEncoder:
        """Placeholder when PyTorch is not available."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch is required for GRUEncoder. Install with: pip install torch"
            )

    class HybridEncoder:
        """Placeholder when PyTorch is not available."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch is required for HybridEncoder. Install with: pip install torch"
            )

    TrajectoryEncoder = StatisticsEncoder

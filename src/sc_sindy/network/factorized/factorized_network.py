"""
Factorized Structure Network for dimension-agnostic structure prediction.

This module provides the main network architecture that combines trajectory
encoding with term embedding to predict equation structure regardless of
the input dimension.
"""

from typing import List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .term_representation import PowerList, get_library_powers, count_library_terms
from .term_embedder import TermEmbedder
from .trajectory_encoder import (
    StatisticsEncoder,
    StatisticsEncoderWithCorrelations,
    GRUEncoder,
    HybridEncoder,
    extract_per_variable_stats,
    extract_stats_with_correlations,
)


if TORCH_AVAILABLE:

    class RelativeEquationEncoder(nn.Module):
        """
        Dimension-agnostic equation encoder using relative position.

        Instead of using a fixed embedding table (which requires pre-specifying
        max_vars), this encoder uses relative position features that naturally
        generalize to any number of variables.

        Features:
        - rel_pos = eq_idx / n_vars: normalized position (0 to 1)
        - n_vars_norm = n_vars / 10: normalized dimension count

        This allows the network to learn patterns like "first equation" or
        "last equation" that transfer across dimensions.

        Parameters
        ----------
        embed_dim : int, optional
            Output embedding dimension (default: 64).
        hidden_dim : int, optional
            Hidden layer dimension (default: 32).

        Examples
        --------
        >>> encoder = RelativeEquationEncoder(embed_dim=64)
        >>> # Encode first equation in 3D system
        >>> e0 = encoder(eq_idx=0, n_vars=3)  # shape: [64]
        >>> # Encode first equation in 5D system (same relative position)
        >>> e0_5d = encoder(eq_idx=0, n_vars=5)  # similar to e0
        """

        def __init__(self, embed_dim: int = 64, hidden_dim: int = 32):
            super().__init__()
            self.embed_dim = embed_dim
            self.hidden_dim = hidden_dim

            # MLP: [relative_pos, n_vars_normalized, is_first, is_last] -> embedding
            # Include binary features for first/last equation for stronger signal
            self.mlp = nn.Sequential(
                nn.Linear(4, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, embed_dim),
            )

        def forward(
            self, eq_idx: int, n_vars: int, device: torch.device = None
        ) -> torch.Tensor:
            """
            Encode equation index using relative position.

            Parameters
            ----------
            eq_idx : int
                Equation index (0-indexed).
            n_vars : int
                Total number of variables/equations.
            device : torch.device, optional
                Device for the output tensor.

            Returns
            -------
            embedding : torch.Tensor
                Equation embedding with shape [embed_dim].
            """
            # Compute relative position features
            rel_pos = eq_idx / max(n_vars - 1, 1)  # 0 to 1 (avoid div by zero)
            n_vars_norm = n_vars / 10.0  # normalize by typical max
            is_first = 1.0 if eq_idx == 0 else 0.0
            is_last = 1.0 if eq_idx == n_vars - 1 else 0.0

            features = torch.tensor(
                [rel_pos, n_vars_norm, is_first, is_last],
                dtype=torch.float32,
                device=device,
            )

            return self.mlp(features)

        def forward_batch(self, n_vars: int, device: torch.device = None) -> torch.Tensor:
            """
            Encode all equation indices for a given dimension.

            Parameters
            ----------
            n_vars : int
                Total number of variables/equations.
            device : torch.device, optional
                Device for the output tensor.

            Returns
            -------
            embeddings : torch.Tensor
                Equation embeddings with shape [n_vars, embed_dim].
            """
            embeddings = []
            for eq_idx in range(n_vars):
                embeddings.append(self.forward(eq_idx, n_vars, device))
            return torch.stack(embeddings)

    class FactorizedStructureNetwork(nn.Module):
        """
        Factorized network for predicting equation structure.

        This architecture separates trajectory encoding from term prediction,
        allowing the same model to work across different dimensions.

        Architecture:
        1. TrajectoryEncoder: maps trajectory → latent vector z_traj
        2. TermEmbedder: maps each library term → embedding e_term
        3. MatchingMLP: predicts p(term active | trajectory) from (z_traj, e_term)

        The prediction for each equation is made by concatenating:
        - Trajectory latent z_traj
        - Term embedding e_term
        - Equation index embedding (to distinguish dx/dt from dy/dt, etc.)

        Parameters
        ----------
        latent_dim : int, optional
            Latent space dimension (default: 64).
        encoder_type : str, optional
            Type of trajectory encoder: 'statistics', 'gru', or 'hybrid'
            (default: 'statistics').
        max_vars : int, optional
            Maximum number of state variables supported (default: 10).
        max_power : int, optional
            Maximum polynomial power supported (default: 5).

        Examples
        --------
        >>> model = FactorizedStructureNetwork(latent_dim=64)
        >>> # 2D trajectory
        >>> x_2d = torch.randn(1000, 2)
        >>> probs_2d = model.predict_structure(x_2d, n_vars=2, poly_order=3)
        >>> print(probs_2d.shape)  # [2, 10]
        >>> # 3D trajectory (same model!)
        >>> x_3d = torch.randn(1000, 3)
        >>> probs_3d = model.predict_structure(x_3d, n_vars=3, poly_order=2)
        >>> print(probs_3d.shape)  # [3, 10]
        """

        def __init__(
            self,
            latent_dim: int = 64,
            encoder_type: str = "statistics",
            max_vars: int = 10,
            max_power: int = 5,
        ):
            super().__init__()

            self.latent_dim = latent_dim
            self.encoder_type = encoder_type
            self.max_vars = max_vars
            self.max_power = max_power

            # Trajectory encoder
            if encoder_type == "statistics":
                self.trajectory_encoder = StatisticsEncoder(latent_dim=latent_dim)
            elif encoder_type == "gru":
                self.trajectory_encoder = GRUEncoder(latent_dim=latent_dim)
            elif encoder_type == "hybrid":
                self.trajectory_encoder = HybridEncoder(latent_dim=latent_dim)
            else:
                raise ValueError(f"Unknown encoder type: {encoder_type}")

            # Term embedder
            self.term_embedder = TermEmbedder(
                latent_dim=latent_dim, max_vars=max_vars, max_power=max_power
            )

            # Equation index embeddings (to distinguish which equation we're predicting)
            self.eq_embed = nn.Embedding(max_vars, latent_dim // 2)

            # Matching MLP: takes (z_traj, e_term, eq_embed) → probability
            # Input: latent_dim (traj) + latent_dim (term) + latent_dim//2 (eq)
            match_input_dim = latent_dim * 2 + latent_dim // 2
            self.matching_mlp = nn.Sequential(
                nn.Linear(match_input_dim, latent_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(latent_dim, latent_dim // 2),
                nn.ReLU(),
                nn.Linear(latent_dim // 2, 1),
                nn.Sigmoid(),
            )

        def encode_trajectory_from_stats(self, stats: torch.Tensor) -> torch.Tensor:
            """
            Encode trajectory from pre-computed statistics.

            Parameters
            ----------
            stats : torch.Tensor
                Per-variable statistics with shape [n_vars, stats_dim].

            Returns
            -------
            z_traj : torch.Tensor
                Trajectory latent with shape [latent_dim].
            """
            return self.trajectory_encoder(stats)

        def encode_trajectory(self, x: torch.Tensor) -> torch.Tensor:
            """
            Encode trajectory tensor.

            Parameters
            ----------
            x : torch.Tensor
                Trajectory with shape [T, n_vars].

            Returns
            -------
            z_traj : torch.Tensor
                Trajectory latent with shape [latent_dim].
            """
            if self.encoder_type == "statistics":
                # Extract statistics and encode
                x_np = x.detach().cpu().numpy()
                stats = extract_per_variable_stats(x_np)
                stats_tensor = torch.FloatTensor(stats).to(x.device)
                return self.trajectory_encoder(stats_tensor)
            else:
                # Directly encode trajectory
                return self.trajectory_encoder(x)

        def predict_term_probability(
            self,
            z_traj: torch.Tensor,
            e_term: torch.Tensor,
            eq_idx: int,
        ) -> torch.Tensor:
            """
            Predict probability that a term is active in a given equation.

            Parameters
            ----------
            z_traj : torch.Tensor
                Trajectory latent with shape [latent_dim].
            e_term : torch.Tensor
                Term embedding with shape [latent_dim].
            eq_idx : int
                Equation index (0 for dx/dt, 1 for dy/dt, etc.).

            Returns
            -------
            prob : torch.Tensor
                Probability scalar.
            """
            eq_idx_tensor = torch.tensor(eq_idx, device=z_traj.device)
            eq_e = self.eq_embed(eq_idx_tensor)

            # Concatenate all components
            combined = torch.cat([z_traj, e_term, eq_e])

            return self.matching_mlp(combined).squeeze()

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
                Per-variable statistics with shape [n_vars, stats_dim].
            n_vars : int
                Number of state variables.
            poly_order : int
                Maximum polynomial order.

            Returns
            -------
            probs : torch.Tensor
                Structure probabilities with shape [n_vars, n_terms].
            """
            # Encode trajectory
            z_traj = self.encode_trajectory_from_stats(stats)

            # Embed all library terms
            term_embeds = self.term_embedder(n_vars, poly_order)  # [n_terms, latent_dim]
            n_terms = term_embeds.shape[0]

            # Predict probability for each (equation, term) pair
            probs = torch.zeros(n_vars, n_terms, device=stats.device)

            for eq_idx in range(n_vars):
                eq_e = self.eq_embed(
                    torch.tensor(eq_idx, device=stats.device)
                )  # [latent_dim//2]

                for term_idx in range(n_terms):
                    e_term = term_embeds[term_idx]  # [latent_dim]
                    combined = torch.cat([z_traj, e_term, eq_e])
                    probs[eq_idx, term_idx] = self.matching_mlp(combined).squeeze()

            return probs

        def forward_batched(
            self,
            stats: torch.Tensor,
            n_vars: int,
            poly_order: int,
        ) -> torch.Tensor:
            """
            Batched forward pass for efficiency.

            Parameters
            ----------
            stats : torch.Tensor
                Per-variable statistics with shape [batch, n_vars, stats_dim].
            n_vars : int
                Number of state variables.
            poly_order : int
                Maximum polynomial order.

            Returns
            -------
            probs : torch.Tensor
                Structure probabilities with shape [batch, n_vars, n_terms].
            """
            batch_size = stats.shape[0]
            n_terms = count_library_terms(n_vars, poly_order)

            # Encode trajectories
            z_traj = self.trajectory_encoder(stats)  # [batch, latent_dim]

            # Embed all library terms (same for all batch items)
            term_embeds = self.term_embedder(n_vars, poly_order)  # [n_terms, latent_dim]

            # Build all (z_traj, e_term, eq_e) combinations
            probs = torch.zeros(batch_size, n_vars, n_terms, device=stats.device)

            for eq_idx in range(n_vars):
                eq_e = self.eq_embed(torch.tensor(eq_idx, device=stats.device))
                eq_e = eq_e.unsqueeze(0).expand(batch_size, -1)  # [batch, latent_dim//2]

                for term_idx in range(n_terms):
                    e_term = term_embeds[term_idx].unsqueeze(0).expand(
                        batch_size, -1
                    )  # [batch, latent_dim]
                    combined = torch.cat(
                        [z_traj, e_term, eq_e], dim=-1
                    )  # [batch, input_dim]
                    probs[:, eq_idx, term_idx] = self.matching_mlp(combined).squeeze(-1)

            return probs

        def predict_structure(
            self,
            x: np.ndarray,
            n_vars: Optional[int] = None,
            poly_order: int = 3,
        ) -> np.ndarray:
            """
            Predict structure from numpy trajectory.

            Parameters
            ----------
            x : np.ndarray
                Trajectory with shape [T, n_vars].
            n_vars : int, optional
                Number of variables. If None, inferred from x.
            poly_order : int, optional
                Maximum polynomial order (default: 3).

            Returns
            -------
            probs : np.ndarray
                Structure probabilities with shape [n_vars, n_terms].
            """
            if n_vars is None:
                n_vars = x.shape[1]

            self.eval()
            with torch.no_grad():
                stats = extract_per_variable_stats(x)
                stats_tensor = torch.FloatTensor(stats)
                probs = self.forward(stats_tensor, n_vars, poly_order)
                return probs.numpy()

        def save(self, path: str):
            """Save model to file."""
            torch.save(
                {
                    "state_dict": self.state_dict(),
                    "latent_dim": self.latent_dim,
                    "encoder_type": self.encoder_type,
                    "max_vars": self.max_vars,
                    "max_power": self.max_power,
                },
                path,
            )

        @classmethod
        def load(cls, path: str) -> "FactorizedStructureNetwork":
            """Load model from file."""
            checkpoint = torch.load(path, weights_only=False)
            model = cls(
                latent_dim=checkpoint["latent_dim"],
                encoder_type=checkpoint["encoder_type"],
                max_vars=checkpoint["max_vars"],
                max_power=checkpoint["max_power"],
            )
            model.load_state_dict(checkpoint["state_dict"])
            return model

    class FactorizedStructureNetworkV2(nn.Module):
        """
        Alternative factorized network with more efficient batching.

        This version vectorizes the matching computation across all terms
        and equations simultaneously, which is faster for training.

        Parameters
        ----------
        latent_dim : int, optional
            Latent space dimension (default: 64).
        max_vars : int, optional
            Maximum number of state variables (default: 10).
        max_power : int, optional
            Maximum polynomial power (default: 5).
        max_terms : int, optional
            Maximum number of library terms to pre-allocate (default: 50).
        use_relative_eq_encoder : bool, optional
            If True, use RelativeEquationEncoder (dimension-agnostic) instead
            of embedding table (default: True).
        use_correlations : bool, optional
            If True, use StatisticsEncoderWithCorrelations to capture
            cross-variable interactions (default: False).
        """

        def __init__(
            self,
            latent_dim: int = 64,
            max_vars: int = 10,
            max_power: int = 5,
            max_terms: int = 50,
            use_relative_eq_encoder: bool = True,
            use_correlations: bool = False,
        ):
            super().__init__()

            self.latent_dim = latent_dim
            self.max_vars = max_vars
            self.max_power = max_power
            self.max_terms = max_terms
            self.use_relative_eq_encoder = use_relative_eq_encoder
            self.use_correlations = use_correlations

            # Trajectory encoder: with or without correlations
            if use_correlations:
                self.trajectory_encoder = StatisticsEncoderWithCorrelations(
                    latent_dim=latent_dim, use_correlation_attention=True
                )
            else:
                self.trajectory_encoder = StatisticsEncoder(latent_dim=latent_dim)

            # Term embedder
            self.term_embedder = TermEmbedder(
                latent_dim=latent_dim, max_vars=max_vars, max_power=max_power
            )

            # Equation encoder: relative position (new) or embedding table (legacy)
            if use_relative_eq_encoder:
                self.eq_encoder = RelativeEquationEncoder(embed_dim=latent_dim)
                self.eq_embed = None  # Not used
            else:
                self.eq_embed = nn.Embedding(max_vars, latent_dim)
                self.eq_encoder = None  # Not used

            # Efficient matching network using bilinear attention
            self.traj_proj = nn.Linear(latent_dim, latent_dim)
            self.term_proj = nn.Linear(latent_dim, latent_dim)
            self.eq_proj = nn.Linear(latent_dim, latent_dim)

            # Final classifier
            self.classifier = nn.Sequential(
                nn.Linear(latent_dim, latent_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(latent_dim // 2, 1),
                nn.Sigmoid(),
            )

        def forward(
            self,
            stats: torch.Tensor,
            n_vars: int,
            poly_order: int,
            corr_matrix: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """
            Vectorized forward pass.

            Parameters
            ----------
            stats : torch.Tensor
                Per-variable statistics with shape [batch, n_vars_in, stats_dim]
                or [n_vars_in, stats_dim] for single sample.
            n_vars : int
                Number of state variables in the dynamical system.
            poly_order : int
                Maximum polynomial order.
            corr_matrix : torch.Tensor, optional
                Pairwise correlation matrix with shape [batch, n_vars, n_vars]
                or [n_vars, n_vars]. Only used if use_correlations=True.

            Returns
            -------
            probs : torch.Tensor
                Structure probabilities with shape [batch, n_vars, n_terms]
                or [n_vars, n_terms] for single sample.
            """
            single_sample = stats.dim() == 2
            if single_sample:
                stats = stats.unsqueeze(0)
                if corr_matrix is not None:
                    corr_matrix = corr_matrix.unsqueeze(0)

            # Handle NaN/Inf in input stats with tighter bounds
            stats = torch.nan_to_num(stats, nan=0.0, posinf=100.0, neginf=-100.0)
            stats = torch.clamp(stats, min=-100.0, max=100.0)

            batch_size = stats.shape[0]
            n_terms = count_library_terms(n_vars, poly_order)

            # Encode trajectories: [batch, latent_dim]
            if self.use_correlations:
                z_traj = self.trajectory_encoder(stats, corr_matrix)
            else:
                z_traj = self.trajectory_encoder(stats)
            z_traj = self.traj_proj(z_traj)  # [batch, latent_dim]

            # Embed terms: [n_terms, latent_dim]
            term_embeds = self.term_embedder(n_vars, poly_order)
            term_embeds = self.term_proj(term_embeds)  # [n_terms, latent_dim]

            # Embed equations: [n_vars, latent_dim]
            if self.use_relative_eq_encoder:
                # Use relative position encoder (dimension-agnostic)
                eq_embeds = self.eq_encoder.forward_batch(n_vars, device=stats.device)
            else:
                # Use embedding table (legacy)
                eq_indices = torch.arange(n_vars, device=stats.device)
                eq_embeds = self.eq_embed(eq_indices)
            eq_embeds = self.eq_proj(eq_embeds)  # [n_vars, latent_dim]

            # Normalize embeddings before interaction to prevent explosion
            z_traj = z_traj / (z_traj.norm(dim=-1, keepdim=True) + 1e-8)
            term_embeds = term_embeds / (term_embeds.norm(dim=-1, keepdim=True) + 1e-8)
            eq_embeds = eq_embeds / (eq_embeds.norm(dim=-1, keepdim=True) + 1e-8)

            # Compute interaction for all (batch, eq, term) combinations
            # z_traj: [batch, 1, 1, latent_dim]
            # term_embeds: [1, 1, n_terms, latent_dim]
            # eq_embeds: [1, n_vars, 1, latent_dim]
            z_traj_exp = z_traj.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, latent]
            term_exp = term_embeds.unsqueeze(0).unsqueeze(0)  # [1, 1, n_terms, latent]
            eq_exp = eq_embeds.unsqueeze(0).unsqueeze(2)  # [1, n_vars, 1, latent]

            # Element-wise product (bilinear interaction)
            interaction = z_traj_exp * term_exp * eq_exp  # [batch, n_vars, n_terms, latent]

            # Classify
            probs = self.classifier(interaction).squeeze(-1)  # [batch, n_vars, n_terms]

            # Clamp to valid probability range to handle numerical issues
            probs = torch.clamp(probs, min=1e-7, max=1 - 1e-7)

            if single_sample:
                probs = probs.squeeze(0)

            return probs

        def predict_structure(
            self,
            x: np.ndarray,
            n_vars: Optional[int] = None,
            poly_order: int = 3,
        ) -> np.ndarray:
            """Predict structure from numpy trajectory."""
            if n_vars is None:
                n_vars = x.shape[1]

            self.eval()
            with torch.no_grad():
                if self.use_correlations:
                    stats, corr = extract_stats_with_correlations(x)
                    stats_tensor = torch.FloatTensor(stats)
                    corr_tensor = torch.FloatTensor(corr)
                    probs = self.forward(stats_tensor, n_vars, poly_order, corr_tensor)
                else:
                    stats = extract_per_variable_stats(x)
                    stats_tensor = torch.FloatTensor(stats)
                    probs = self.forward(stats_tensor, n_vars, poly_order)
                return probs.numpy()

        def predict_with_uncertainty(
            self,
            x: np.ndarray,
            n_vars: Optional[int] = None,
            poly_order: int = 3,
            n_samples: int = 20,
        ) -> Tuple[np.ndarray, np.ndarray]:
            """
            Predict structure with uncertainty via MC Dropout.

            This method runs multiple forward passes with dropout enabled to
            estimate prediction uncertainty. Higher uncertainty indicates
            the model is less confident about a term's inclusion.

            Parameters
            ----------
            x : np.ndarray
                Trajectory with shape [T, n_vars].
            n_vars : int, optional
                Number of variables. If None, inferred from x.
            poly_order : int, optional
                Maximum polynomial order (default: 3).
            n_samples : int, optional
                Number of MC samples (default: 20). More samples give
                better uncertainty estimates but are slower.

            Returns
            -------
            mean_probs : np.ndarray
                Mean predicted probabilities with shape [n_vars, n_terms].
            uncertainty : np.ndarray
                Standard deviation of predictions with shape [n_vars, n_terms].
                Higher values indicate more uncertainty.

            Examples
            --------
            >>> model = FactorizedStructureNetworkV2()
            >>> x = np.random.randn(1000, 3)
            >>> mean_probs, uncertainty = model.predict_with_uncertainty(x)
            >>> # High uncertainty regions (>0.1) indicate unreliable predictions
            >>> unreliable = uncertainty > 0.1
            """
            if n_vars is None:
                n_vars = x.shape[1]

            # Prepare inputs
            if self.use_correlations:
                stats, corr = extract_stats_with_correlations(x)
                stats_tensor = torch.FloatTensor(stats)
                corr_tensor = torch.FloatTensor(corr)
            else:
                stats = extract_per_variable_stats(x)
                stats_tensor = torch.FloatTensor(stats)
                corr_tensor = None

            # Enable dropout for MC sampling
            self.train()

            # Collect predictions
            predictions = []
            with torch.no_grad():
                for _ in range(n_samples):
                    if self.use_correlations:
                        probs = self.forward(stats_tensor, n_vars, poly_order, corr_tensor)
                    else:
                        probs = self.forward(stats_tensor, n_vars, poly_order)
                    predictions.append(probs.numpy())

            # Restore eval mode
            self.eval()

            # Compute statistics
            predictions = np.stack(predictions, axis=0)  # [n_samples, n_vars, n_terms]
            mean_probs = predictions.mean(axis=0)
            uncertainty = predictions.std(axis=0)

            return mean_probs, uncertainty

        def save(self, path: str):
            """Save model to file."""
            torch.save(
                {
                    "state_dict": self.state_dict(),
                    "latent_dim": self.latent_dim,
                    "max_vars": self.max_vars,
                    "max_power": self.max_power,
                    "max_terms": self.max_terms,
                    "use_relative_eq_encoder": self.use_relative_eq_encoder,
                    "use_correlations": self.use_correlations,
                },
                path,
            )

        @classmethod
        def load(cls, path: str) -> "FactorizedStructureNetworkV2":
            """Load model from file.

            Handles both the native save format and training script format.
            """
            checkpoint = torch.load(path, weights_only=False)

            # Handle training script format (model_state_dict key)
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
                latent_dim = checkpoint.get("latent_dim", 64)
                # Use defaults for other params since training script doesn't save them
                model = cls(
                    latent_dim=latent_dim,
                    use_relative_eq_encoder=True,  # Default for V2
                    use_correlations=False,
                )
                model.load_state_dict(state_dict)
                return model

            # Handle native save format (state_dict key)
            model = cls(
                latent_dim=checkpoint["latent_dim"],
                max_vars=checkpoint.get("max_vars", 10),
                max_power=checkpoint.get("max_power", 5),
                max_terms=checkpoint.get("max_terms", 50),
                use_relative_eq_encoder=checkpoint.get("use_relative_eq_encoder", False),
                use_correlations=checkpoint.get("use_correlations", False),
            )
            model.load_state_dict(checkpoint["state_dict"])
            return model

else:
    # Placeholders when PyTorch is not available
    class FactorizedStructureNetwork:
        """Placeholder when PyTorch is not available."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch is required for FactorizedStructureNetwork. "
                "Install with: pip install torch"
            )

    class FactorizedStructureNetworkV2:
        """Placeholder when PyTorch is not available."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch is required for FactorizedStructureNetworkV2. "
                "Install with: pip install torch"
            )

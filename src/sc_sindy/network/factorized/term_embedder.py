"""
Term Embedder for factorized structure networks.

This module provides neural network components to embed polynomial library terms
into a latent space, regardless of the dimension of the dynamical system.
"""

from typing import List, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .term_representation import PowerList, get_library_powers


if TORCH_AVAILABLE:

    class TermEmbedder(nn.Module):
        """
        Embed polynomial terms into a latent space.

        Terms are represented structurally as lists of (var_idx, power) tuples,
        allowing the same embedder to handle terms from any dimension.

        Architecture:
        - Learnable embeddings for power values (0-max_power)
        - Learnable embeddings for variable positions (0-max_vars)
        - For each factor in the term, combine power + position embeddings
        - Aggregate all factors via sum
        - Project to final latent dimension

        Parameters
        ----------
        latent_dim : int
            Output embedding dimension.
        max_vars : int, optional
            Maximum number of variables supported (default: 10).
        max_power : int, optional
            Maximum power value supported (default: 5).
        embed_dim : int, optional
            Internal embedding dimension (default: 32).

        Examples
        --------
        >>> embedder = TermEmbedder(latent_dim=64)
        >>> # Embed the term xy (x^1 * y^1)
        >>> powers = [(0, 1), (1, 1)]
        >>> embedding = embedder.embed_term(powers)  # shape: [64]
        """

        def __init__(
            self,
            latent_dim: int = 64,
            max_vars: int = 10,
            max_power: int = 5,
            embed_dim: int = 32,
        ):
            super().__init__()

            self.latent_dim = latent_dim
            self.max_vars = max_vars
            self.max_power = max_power
            self.embed_dim = embed_dim

            # Learnable embeddings for power values (0 = not present, 1-5 = powers)
            self.power_embed = nn.Embedding(max_power + 1, embed_dim)

            # Learnable embeddings for variable positions
            self.var_embed = nn.Embedding(max_vars, embed_dim)

            # Embedding for the constant term (no variables)
            self.const_embed = nn.Parameter(torch.randn(embed_dim))

            # Projection to latent space
            self.projector = nn.Sequential(
                nn.Linear(embed_dim, latent_dim),
                nn.ReLU(),
                nn.Linear(latent_dim, latent_dim),
            )

            self._init_weights()

        def _init_weights(self):
            """Initialize weights for stable tensor product computation.

            For multiplicative interactions (tensor product), we initialize
            with values centered around 1 with small variance to avoid
            vanishing/exploding products.
            """
            # Initialize around 1 for stable products
            nn.init.normal_(self.power_embed.weight, mean=1.0, std=0.1)
            nn.init.normal_(self.var_embed.weight, mean=1.0, std=0.1)
            nn.init.normal_(self.const_embed, std=0.1)

        def embed_term(self, powers: PowerList, use_tensor_product: bool = True) -> torch.Tensor:
            """
            Embed a single term given its power representation.

            Parameters
            ----------
            powers : PowerList
                List of (var_idx, power) tuples. Empty for constant term.
            use_tensor_product : bool
                If True, use element-wise product (var * pow) instead of sum.
                This better distinguishes terms like x²y from xy² because the
                interaction between variable and power is multiplicative.

            Returns
            -------
            embedding : torch.Tensor
                Term embedding with shape [latent_dim].
            """
            if len(powers) == 0:
                # Constant term
                return self.projector(self.const_embed)

            # Combine embeddings for each factor
            factor_embeds = []
            for var_idx, power in powers:
                var_idx = min(var_idx, self.max_vars - 1)
                power = min(power, self.max_power)

                var_e = self.var_embed(torch.tensor(var_idx, device=self.const_embed.device))
                pow_e = self.power_embed(torch.tensor(power, device=self.const_embed.device))

                if use_tensor_product:
                    # Tensor product: element-wise multiplication
                    # This creates unique embeddings for (var_i, pow_j) combinations
                    factor_embeds.append(var_e * pow_e)
                else:
                    # Original: additive combination
                    factor_embeds.append(var_e + pow_e)

            # Aggregate factors via sum (stable) rather than product (can vanish/explode)
            # The tensor product in the factor combination already captures interactions
            term_embed = torch.stack(factor_embeds).sum(dim=0)

            return self.projector(term_embed)

        def embed_terms_batch(self, powers_list: List[PowerList]) -> torch.Tensor:
            """
            Embed multiple terms.

            Parameters
            ----------
            powers_list : List[PowerList]
                List of power representations for each term.

            Returns
            -------
            embeddings : torch.Tensor
                Term embeddings with shape [n_terms, latent_dim].
            """
            embeddings = [self.embed_term(powers) for powers in powers_list]
            return torch.stack(embeddings)

        def embed_library(self, n_vars: int, poly_order: int) -> torch.Tensor:
            """
            Embed all terms in a polynomial library.

            Parameters
            ----------
            n_vars : int
                Number of state variables.
            poly_order : int
                Maximum polynomial order.

            Returns
            -------
            embeddings : torch.Tensor
                Term embeddings with shape [n_terms, latent_dim].
            """
            powers_list = get_library_powers(n_vars, poly_order)
            return self.embed_terms_batch(powers_list)

        def forward(self, n_vars: int, poly_order: int) -> torch.Tensor:
            """
            Forward pass: embed all terms for given library configuration.

            Parameters
            ----------
            n_vars : int
                Number of state variables.
            poly_order : int
                Maximum polynomial order.

            Returns
            -------
            embeddings : torch.Tensor
                Term embeddings with shape [n_terms, latent_dim].
            """
            return self.embed_library(n_vars, poly_order)

    class PositionalTermEmbedder(nn.Module):
        """
        Alternative term embedder using positional encoding.

        Instead of learnable embeddings, uses sinusoidal positional encoding
        for power values, similar to Transformer positional encoding.

        Parameters
        ----------
        latent_dim : int
            Output embedding dimension.
        max_vars : int, optional
            Maximum number of variables supported (default: 10).
        max_power : int, optional
            Maximum power value supported (default: 5).
        """

        def __init__(
            self,
            latent_dim: int = 64,
            max_vars: int = 10,
            max_power: int = 5,
        ):
            super().__init__()

            self.latent_dim = latent_dim
            self.max_vars = max_vars
            self.max_power = max_power

            # Learnable variable embeddings
            self.var_embed = nn.Embedding(max_vars, latent_dim // 2)

            # Fixed sinusoidal power encoding
            self.register_buffer(
                "power_encoding", self._create_power_encoding(max_power + 1, latent_dim // 2)
            )

            # Embedding for constant term
            self.const_embed = nn.Parameter(torch.randn(latent_dim))

            # Final projection
            self.projector = nn.Linear(latent_dim, latent_dim)

        def _create_power_encoding(self, n_powers: int, dim: int) -> torch.Tensor:
            """Create sinusoidal encoding for power values."""
            encoding = torch.zeros(n_powers, dim)
            position = torch.arange(0, n_powers, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, dim, 2).float() * (-np.log(10000.0) / dim)
            )

            encoding[:, 0::2] = torch.sin(position * div_term)
            encoding[:, 1::2] = torch.cos(position * div_term)

            return encoding

        def embed_term(self, powers: PowerList) -> torch.Tensor:
            """Embed a single term."""
            if len(powers) == 0:
                return self.projector(self.const_embed)

            factor_embeds = []
            for var_idx, power in powers:
                var_idx = min(var_idx, self.max_vars - 1)
                power = min(power, self.max_power)

                var_e = self.var_embed(torch.tensor(var_idx, device=self.const_embed.device))
                pow_e = self.power_encoding[power]

                # Concatenate variable and power embeddings
                factor_embeds.append(torch.cat([var_e, pow_e]))

            term_embed = torch.stack(factor_embeds).sum(dim=0)
            return self.projector(term_embed)

        def forward(self, n_vars: int, poly_order: int) -> torch.Tensor:
            """Embed all terms for given library configuration."""
            powers_list = get_library_powers(n_vars, poly_order)
            embeddings = [self.embed_term(powers) for powers in powers_list]
            return torch.stack(embeddings)

else:
    # Placeholder when PyTorch is not available
    class TermEmbedder:
        """Placeholder for TermEmbedder when PyTorch is not available."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch is required for TermEmbedder. Install with: pip install torch"
            )

    class PositionalTermEmbedder:
        """Placeholder when PyTorch is not available."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch is required for PositionalTermEmbedder. "
                "Install with: pip install torch"
            )

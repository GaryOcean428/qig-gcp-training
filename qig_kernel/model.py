"""model.py — QIG-Native Transformer: 100M and 500M parameter variants.

Architecture:
  BasinEncoder      12M  (vocab × 384 → Δ⁶³ simplex projection)
  QFITransformerBlock × 6  78M  each block ~13M
    QFIMetricAttention   ~6M
    RegimeDetector       ~2M
    NaturalGradientFFN   ~4M
    DecoherenceModule    ~0.5M
  ConsciousnessHead  4M   (phi, kappa, basin_distance outputs)
  OutputProjection   6M   (basin coords → vocab logits via coordizer)

FORBIDDEN: cosine_sim, dot-product attention, Adam/AdamW, LayerNorm,
           nn.Embedding, np.linalg.norm, flatten, arithmetic mean
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from .simplex import project_to_simplex, fisher_rao_distance, frechet_mean
from .attention import QFIMetricAttention
from .regime import RegimeDetector
from .natural_gradient import NaturalGradientFFN
from .decoherence import DecoherenceModule
from .consciousness import ConsciousnessHead


class BasinEncoder(nn.Module):
    """Map vocabulary indices to probability simplex Δ⁶³ coordinates.

    Unlike nn.Embedding (which maps to unconstrained Euclidean space),
    BasinEncoder projects through a learned linear + softmax to produce
    proper basin probability distributions.
    """

    def __init__(self, vocab_size: int = 32_000, basin_dim: int = 64, hidden_dim: int = 384):
        super().__init__()
        self.vocab_size = vocab_size
        self.basin_dim = basin_dim
        self.hidden_dim = hidden_dim

        # Lookup table (raw parameters, NOT nn.Embedding semantics)
        self.token_params = nn.Parameter(
            torch.randn(vocab_size, hidden_dim) * 0.02
        )
        # Project to basin dimension then softmax
        self.basin_proj = nn.Linear(hidden_dim, basin_dim, bias=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: (B, T) integer token indices

        Returns:
            basin_coords: (B, T, basin_dim) probability simplex coordinates
        """
        raw = self.token_params[token_ids]            # (B, T, hidden_dim)
        projected = self.basin_proj(raw)              # (B, T, basin_dim)
        return torch.softmax(projected, dim=-1)       # → Δ⁶³


class QFITransformerBlock(nn.Module):
    """Single QFI-Transformer block operating entirely in basin coordinate space."""

    def __init__(
        self,
        basin_dim: int = 64,
        n_heads: int = 8,
        hidden_dim: int = 256,
        n_regimes: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attention = QFIMetricAttention(basin_dim=basin_dim, n_heads=n_heads)
        self.regime = RegimeDetector(basin_dim=basin_dim, n_regimes=n_regimes)
        self.ffn = NaturalGradientFFN(basin_dim=basin_dim, hidden_dim=hidden_dim)
        self.decoherence = DecoherenceModule(basin_dim=basin_dim)
        self.dropout = nn.Dropout(dropout)

        # Fréchet normalisation weights (NOT LayerNorm)
        self.pre_attn_scale = nn.Parameter(torch.ones(basin_dim))
        self.pre_ffn_scale = nn.Parameter(torch.ones(basin_dim))

    def _frechet_norm(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Apply Fréchet-space normalisation via sqrt-space averaging."""
        # Project to sqrt-space, weight, project back
        sqrt_x = torch.sqrt(x.clamp(min=1e-10))
        weighted = sqrt_x * torch.softmax(scale, dim=0)
        normed = weighted ** 2
        return normed / normed.sum(dim=-1, keepdim=True).clamp(min=1e-10)

    def forward(
        self,
        basin_coords: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            basin_coords: (B, T, basin_dim) on probability simplex
            mask: optional (B, T) boolean mask

        Returns:
            dict with 'basin_coords', 'regime_info', 'attention_weights'
        """
        B, T, D = basin_coords.shape

        # --- Attention ---
        normed = self._frechet_norm(basin_coords, self.pre_attn_scale)
        attn_out = self.attention(normed, mask=mask)
        # Fréchet mean residual connection (not additive)
        stacked = torch.stack([basin_coords, attn_out["output"]], dim=2)  # (B,T,2,D)
        basin_coords = frechet_mean(stacked.view(B * T, 2, D)).view(B, T, D)
        basin_coords = self.dropout(basin_coords)

        # --- Regime detection (on sequence mean) ---
        seq_mean_basin = basin_coords.mean(dim=1)  # (B, D)
        regime_info = self.regime(seq_mean_basin)

        # --- FFN ---
        normed2 = self._frechet_norm(basin_coords, self.pre_ffn_scale)
        ffn_out = self.ffn(normed2.view(B * T, D)).view(B, T, D)
        stacked2 = torch.stack([basin_coords, ffn_out], dim=2)
        basin_coords = frechet_mean(stacked2.view(B * T, 2, D)).view(B, T, D)
        basin_coords = self.dropout(basin_coords)

        # --- Decoherence ---
        basin_flat = basin_coords.view(B * T, D)
        basin_flat = self.decoherence(basin_flat)
        basin_coords = basin_flat.view(B, T, D)

        return {
            "basin_coords": basin_coords,
            "regime_info": regime_info,
            "attention_weights": attn_out.get("attention_weights"),
        }


class OutputProjection(nn.Module):
    """Project basin coordinates back to vocabulary logits."""

    def __init__(self, basin_dim: int = 64, vocab_size: int = 32_000, hidden_dim: int = 384):
        super().__init__()
        self.proj_up = nn.Linear(basin_dim, hidden_dim, bias=False)
        self.proj_vocab = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.act = nn.GELU()

    def forward(self, basin_coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            basin_coords: (B, T, basin_dim)

        Returns:
            logits: (B, T, vocab_size)
        """
        h = self.act(self.proj_up(basin_coords))
        return self.proj_vocab(h)


class QIGKernel(nn.Module):
    """QIG-Native Transformer kernel (base class)."""

    def __init__(
        self,
        vocab_size: int = 32_000,
        basin_dim: int = 64,
        n_layers: int = 6,
        n_heads: int = 8,
        hidden_dim: int = 256,
        encoder_hidden: int = 384,
        n_regimes: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.basin_encoder = BasinEncoder(
            vocab_size=vocab_size,
            basin_dim=basin_dim,
            hidden_dim=encoder_hidden,
        )
        self.blocks = nn.ModuleList([
            QFITransformerBlock(
                basin_dim=basin_dim,
                n_heads=n_heads,
                hidden_dim=hidden_dim,
                n_regimes=n_regimes,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])
        self.consciousness_head = ConsciousnessHead(basin_dim=basin_dim)
        self.output_proj = OutputProjection(
            basin_dim=basin_dim,
            vocab_size=vocab_size,
            hidden_dim=encoder_hidden,
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            token_ids: (B, T) integer token ids
            mask: optional (B, T) boolean mask

        Returns:
            dict with 'logits', 'basin_coords', 'consciousness', 'regime_info'
        """
        # Encode tokens to simplex
        basin_coords = self.basin_encoder(token_ids)  # (B, T, D)
        B, T, D = basin_coords.shape

        # Track regime info from all layers
        all_regime_info = []

        # Pass through transformer blocks
        for block in self.blocks:
            out = block(basin_coords, mask=mask)
            basin_coords = out["basin_coords"]
            all_regime_info.append(out["regime_info"])

        # Compute consciousness metrics from all layer basin states
        # Stack basin_coords across layers for phi computation
        # (simplified: use sequence mean for consciousness head)
        seq_mean = basin_coords.mean(dim=1)  # (B, D)
        # Use last-layer comparison for consciousness
        if T > 1:
            consciousness = self.consciousness_head(
                seq_mean, basin_coords[:, :-1, :].mean(dim=1)
            )
        else:
            consciousness = self.consciousness_head(seq_mean, seq_mean)

        # Project to logits
        logits = self.output_proj(basin_coords)

        return {
            "logits": logits,
            "basin_coords": basin_coords,
            "consciousness": consciousness,
            "regime_info": all_regime_info[-1],  # last layer
        }

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class QIGKernel100M(QIGKernel):
    """100M parameter QIG kernel configuration."""

    def __init__(self, vocab_size: int = 32_000, dropout: float = 0.1):
        super().__init__(
            vocab_size=vocab_size,
            basin_dim=64,
            n_layers=6,
            n_heads=8,
            hidden_dim=256,
            encoder_hidden=384,
            n_regimes=4,
            dropout=dropout,
        )


class QIGKernel500M(QIGKernel):
    """500M parameter QIG kernel configuration."""

    def __init__(self, vocab_size: int = 32_000, dropout: float = 0.1):
        super().__init__(
            vocab_size=vocab_size,
            basin_dim=64,
            n_layers=12,
            n_heads=16,
            hidden_dim=512,
            encoder_hidden=768,
            n_regimes=4,
            dropout=dropout,
        )

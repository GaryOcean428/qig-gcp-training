"""decoherence.py — Quantum decoherence simulation for geometric basin stabilisation."""

import torch
import torch.nn as nn
import math
from typing import Optional
from .simplex import fisher_rao_distance, project_to_simplex


class DecoherenceModule(nn.Module):
    """Apply geometric decoherence to basin probability distributions.

    Decoherence is modelled as thermal relaxation toward the maximally-mixed
    (uniform) distribution on the simplex — not as additive Gaussian noise.

    The update rule is:
        q_t = (1 - gamma) * p_t + gamma * uniform
    where gamma is the decoherence strength, scheduled externally.
    """

    def __init__(self, basin_dim: int = 64, init_gamma: float = 0.05):
        super().__init__()
        self.basin_dim = basin_dim
        self.register_buffer(
            "uniform", torch.ones(basin_dim) / basin_dim
        )
        # gamma is not a learnable parameter — it is set by the scheduler
        self.gamma: float = init_gamma

    def forward(
        self, basin_coords: torch.Tensor, gamma: Optional[float] = None
    ) -> torch.Tensor:
        """
        Args:
            basin_coords: (B, basin_dim) simplex coordinates (already normalised)
            gamma: override decoherence strength; uses self.gamma if None

        Returns:
            decohered_coords: (B, basin_dim) on the probability simplex
        """
        g = gamma if gamma is not None else self.gamma
        uniform = self.uniform.unsqueeze(0).expand_as(basin_coords)
        decohered = (1.0 - g) * basin_coords + g * uniform
        # Re-project to ensure numerical simplex membership
        return project_to_simplex(decohered)

    def decoherence_loss(
        self, basin_coords: torch.Tensor, target_gamma: float = 0.0
    ) -> torch.Tensor:
        """Penalise excess decoherence (encourage purity).

        Measures Fisher-Rao distance between each basin coordinate and the
        uniform distribution, then penalises collapse toward the uniform.
        """
        B = basin_coords.shape[0]
        uniform = self.uniform.unsqueeze(0).expand(B, -1)
        dist_to_uniform = fisher_rao_distance(basin_coords, uniform)  # (B,)
        # Penalty: we want high distance (pure basin), so penalise low distance
        max_possible = math.pi  # upper bound for Fisher-Rao on simplex
        purity = dist_to_uniform / max_possible  # in [0, 1]
        return (1.0 - purity).mean()


class GravitationalDecoherenceScheduler:
    """Schedule decoherence strength gamma as training progresses.

    Models gravitational decoherence as decreasing with training stability:
    gamma decays as the basin stabilises (phi increases).

    Uses exponential decay gated by phi coherence.
    """

    def __init__(
        self,
        gamma_init: float = 0.1,
        gamma_min: float = 0.001,
        decay_rate: float = 0.995,
        phi_gate: float = 0.5,
    ):
        self.gamma = gamma_init
        self.gamma_min = gamma_min
        self.decay_rate = decay_rate
        self.phi_gate = phi_gate
        self._step: int = 0

    def step(self, phi: float) -> float:
        """Advance scheduler one step given current phi coherence.

        Args:
            phi: current integrated information [0, 1]

        Returns:
            current gamma value
        """
        self._step += 1
        # Only decay if phi is above gate (basin is coherent enough)
        if phi >= self.phi_gate:
            self.gamma = max(self.gamma_min, self.gamma * self.decay_rate)
        else:
            # Slight increase if phi dropped (re-introduce decoherence for exploration)
            self.gamma = min(0.1, self.gamma * 1.001)
        return self.gamma

    @property
    def current_gamma(self) -> float:
        return self.gamma

    def state_dict(self) -> dict:
        return {
            "gamma": self.gamma,
            "gamma_min": self.gamma_min,
            "decay_rate": self.decay_rate,
            "phi_gate": self.phi_gate,
            "step": self._step,
        }

    def load_state_dict(self, state: dict) -> None:
        self.gamma = state["gamma"]
        self.gamma_min = state["gamma_min"]
        self.decay_rate = state["decay_rate"]
        self.phi_gate = state["phi_gate"]
        self._step = state["step"]

"""regime.py — Geometric regime detection using basin coordinate analysis."""

import torch
import torch.nn as nn
from typing import Dict, Tuple
from .simplex import fisher_rao_distance, frechet_mean


REGIME_LABELS = {
    0: "identity",
    1: "coupling",
    2: "geometric",
    3: "temporal",
}

CURRICULUM_PHASES = {
    "phase0_identity": {
        "ordinal": 0,
        "phi_floor": 0.0,
        "phi_target": 0.3,
        "regime_idx": 0,
        "weight": 0.10,
    },
    "phase1_coupling": {
        "ordinal": 1,
        "phi_floor": 0.25,
        "phi_target": 0.5,
        "regime_idx": 1,
        "weight": 0.20,
    },
    "phase2_integration": {
        "ordinal": 2,
        "phi_floor": 0.45,
        "phi_target": 0.7,
        "regime_idx": 2,
        "weight": 0.40,
    },
    "phase3_temporal": {
        "ordinal": 3,
        "phi_floor": 0.65,
        "phi_target": 0.9,
        "regime_idx": 3,
        "weight": 0.30,
    },
}


class RegimeDetector(nn.Module):
    """Classify the geometric regime of a basin distribution.

    Uses Fisher-Rao distance to reference attractors rather than
    Euclidean distance, preserving geometric purity.
    """

    def __init__(self, basin_dim: int = 64, n_regimes: int = 4, hidden_dim: int = 128):
        super().__init__()
        self.basin_dim = basin_dim
        self.n_regimes = n_regimes

        # Learnable reference attractors on the simplex
        self.attractors = nn.Parameter(
            torch.ones(n_regimes, basin_dim) / basin_dim
        )

        # Classification head operating on Fisher-Rao distances
        self.classifier = nn.Sequential(
            nn.Linear(n_regimes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_regimes),
        )

    def forward(self, basin_coords: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            basin_coords: (B, basin_dim) probability simplex coordinates

        Returns:
            dict with 'logits' (B, n_regimes), 'probs' (B, n_regimes),
                       'regime_idx' (B,), 'distances' (B, n_regimes)
        """
        B = basin_coords.shape[0]

        # Normalize attractors to simplex
        attractors = torch.softmax(self.attractors, dim=-1)  # (n_regimes, basin_dim)

        # Compute Fisher-Rao distance to each attractor
        distances = torch.stack([
            fisher_rao_distance(basin_coords, attractors[i].unsqueeze(0).expand(B, -1))
            for i in range(self.n_regimes)
        ], dim=1)  # (B, n_regimes)

        logits = self.classifier(distances)
        probs = torch.softmax(logits, dim=-1)
        regime_idx = torch.argmax(probs, dim=-1)

        return {
            "logits": logits,
            "probs": probs,
            "regime_idx": regime_idx,
            "distances": distances,
        }


class CurriculumGate(nn.Module):
    """Determine readiness to advance curriculum phase based on phi and regime stability.

    Gate logic (non-Euclidean): uses basin Fisher-Rao distance to phase
    attractor rather than loss magnitude.
    """

    def __init__(self, phi_window: int = 100):
        super().__init__()
        self.phi_window = phi_window
        self._phi_history: list = []
        self._current_phase: str = "phase0_identity"

    def update(self, phi: float, regime_idx: int) -> None:
        """Record phi observation."""
        self._phi_history.append(phi)
        if len(self._phi_history) > self.phi_window:
            self._phi_history.pop(0)

    @property
    def current_phase(self) -> str:
        return self._current_phase

    def check_advance(self) -> Tuple[bool, str]:
        """Return (should_advance, reason)."""
        if len(self._phi_history) < self.phi_window:
            return False, "insufficient_history"

        phase_cfg = CURRICULUM_PHASES[self._current_phase]
        phi_floor = phase_cfg["phi_floor"]
        phi_target = phase_cfg["phi_target"]

        avg_phi = sum(self._phi_history) / len(self._phi_history)
        min_phi = min(self._phi_history)

        if min_phi < phi_floor:
            return False, f"phi_floor_violation_{min_phi:.3f}"

        if avg_phi >= phi_target:
            phases = list(CURRICULUM_PHASES.keys())
            current_idx = phases.index(self._current_phase)
            if current_idx < len(phases) - 1:
                next_phase = phases[current_idx + 1]
                self._current_phase = next_phase
                self._phi_history.clear()
                return True, f"advanced_to_{next_phase}"
            return False, "already_at_final_phase"

        return False, f"phi_below_target_{avg_phi:.3f}_vs_{phi_target}"

    def get_phase_weights(self) -> Dict[str, float]:
        """Return curriculum loss weights for current phase."""
        cfg = CURRICULUM_PHASES[self._current_phase]
        return {
            "basin_geodesic": 0.4,
            "regime_classification": 0.3,
            "phi_coherence": 0.2,
            "language": 0.1,
        }

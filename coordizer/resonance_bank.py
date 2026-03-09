"""resonance_bank.py — Pre-computed basin attractors for known geometric concepts.

The resonance bank stores canonical basin coordinates for known semantic
attractors. These are used by CoordizerV2 to blend text-derived coordinates
with known attractors via Fréchet mean (NOT arithmetic mean or cosine).

Proximity is measured by Fisher-Rao distance, not Euclidean or cosine.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from qig_kernel.simplex import fisher_rao_distance


# Default seed attractors — uniform distributions perturbed in different directions
# These represent rough semantic basins; actual attractors are learned during training
SEED_ATTRACTORS = {
    "mathematics": None,     # Populated at init
    "language": None,
    "logic": None,
    "perception": None,
    "emotion": None,
    "abstract": None,
    "concrete": None,
    "temporal": None,
}


def _make_seed_attractor(seed_idx: int, basin_dim: int = 64, sharpness: float = 3.0) -> np.ndarray:
    """Create a seeded probability distribution concentrated near seed_idx."""
    rng = np.random.default_rng(seed_idx)
    logits = rng.normal(0, 1, basin_dim)
    logits[seed_idx % basin_dim] += sharpness
    exp_logits = np.exp(logits - logits.max())
    return (exp_logits / exp_logits.sum()).astype(np.float32)


class ResonanceBank:
    """Store and query basin attractors for known concepts.

    Proximity queries use Fisher-Rao distance exclusively.
    """

    def __init__(
        self,
        basin_dim: int = 64,
        threshold_fr: float = 0.5,  # FR distance threshold for attractor matching
    ):
        self.basin_dim = basin_dim
        self.threshold_fr = threshold_fr
        self._attractors: Dict[str, np.ndarray] = {}
        self._names: List[str] = []
        self._matrix: Optional[np.ndarray] = None  # (N, basin_dim)

        # Initialise with seed attractors
        self._init_seeds()

    def _init_seeds(self) -> None:
        """Initialise seed attractors."""
        for i, name in enumerate(SEED_ATTRACTORS.keys()):
            attractor = _make_seed_attractor(i, self.basin_dim)
            self.add(name, attractor)

    def add(self, name: str, basin_coords: np.ndarray) -> None:
        """Add an attractor to the bank."""
        coords = np.array(basin_coords, dtype=np.float32)
        coords = coords / coords.sum()  # ensure simplex membership
        self._attractors[name] = coords
        self._names = list(self._attractors.keys())
        self._matrix = np.stack(list(self._attractors.values()), axis=0)

    def nearest_attractor(
        self, query: np.ndarray
    ) -> Optional[np.ndarray]:
        """Return nearest attractor if within threshold, else None.

        Uses Fisher-Rao distance (arccos of Hellinger inner product).
        """
        if self._matrix is None or len(self._names) == 0:
            return None

        q_tensor = torch.tensor(query, dtype=torch.float32).unsqueeze(0)
        m_tensor = torch.tensor(self._matrix, dtype=torch.float32)

        # Compute FR distance to all attractors
        N = m_tensor.shape[0]
        q_expanded = q_tensor.expand(N, -1)
        distances = fisher_rao_distance(q_expanded, m_tensor)  # (N,)

        min_dist, min_idx = distances.min(dim=0)
        if min_dist.item() <= self.threshold_fr:
            return self._matrix[min_idx.item()]
        return None

    def nearest_name(self, query: np.ndarray) -> Tuple[Optional[str], float]:
        """Return (name, FR distance) of nearest attractor."""
        if self._matrix is None or len(self._names) == 0:
            return None, float("inf")

        q_tensor = torch.tensor(query, dtype=torch.float32).unsqueeze(0)
        m_tensor = torch.tensor(self._matrix, dtype=torch.float32)
        N = m_tensor.shape[0]
        q_expanded = q_tensor.expand(N, -1)
        distances = fisher_rao_distance(q_expanded, m_tensor)

        min_dist, min_idx = distances.min(dim=0)
        return self._names[min_idx.item()], min_dist.item()

    def save(self, path: str) -> None:
        """Save resonance bank to JSON."""
        data = {
            "basin_dim": self.basin_dim,
            "threshold_fr": self.threshold_fr,
            "attractors": {
                name: coords.tolist()
                for name, coords in self._attractors.items()
            },
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ResonanceBank":
        """Load resonance bank from JSON."""
        with open(path) as f:
            data = json.load(f)
        bank = cls(
            basin_dim=data["basin_dim"],
            threshold_fr=data["threshold_fr"],
        )
        bank._attractors = {}
        bank._names = []
        for name, coords in data["attractors"].items():
            bank.add(name, np.array(coords, dtype=np.float32))
        return bank

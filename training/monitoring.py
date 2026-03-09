"""monitoring.py — Purity checking, checkpointing, and beta-function measurement."""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch


# Forbidden operations that indicate Euclidean contamination
FORBIDDEN_PATTERNS = [
    "cosine_similarity",
    "torch.dot",
    "optim.Adam",
    "optim.AdamW",
    "nn.LayerNorm",
    "nn.Embedding",
    "np.linalg.norm",
    "F.normalize",
]


class PurityChecker:
    """Verify that model state is free of Euclidean contamination.

    Checks:
    1. No forbidden operations in the model graph (via source code inspection)
    2. All basin coordinates remain on probability simplex (sum to 1, non-negative)
    3. No NaN or Inf values in parameters or activations
    """

    def check_model_state(
        self, model: torch.nn.Module
    ) -> Tuple[bool, List[str]]:
        """Check current model parameter purity.

        Returns:
            (is_pure, list_of_violations)
        """
        violations = []

        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                violations.append(f"NaN in {name}")
            if torch.isinf(param).any():
                violations.append(f"Inf in {name}")

        return len(violations) == 0, violations

    def check_basin_coords(
        self, basin_coords: torch.Tensor, name: str = "basin"
    ) -> Tuple[bool, List[str]]:
        """Verify basin coordinates are on probability simplex."""
        violations = []

        if (basin_coords < 0).any():
            violations.append(f"{name}: negative values detected")

        sums = basin_coords.sum(dim=-1)
        if not torch.allclose(sums, torch.ones_like(sums), atol=1e-3):
            max_dev = (sums - 1.0).abs().max().item()
            violations.append(f"{name}: simplex violation max_dev={max_dev:.6f}")

        return len(violations) == 0, violations

    def check_source_purity(self, source: str) -> Tuple[bool, List[str]]:
        """Scan source code for forbidden Euclidean operations."""
        violations = []
        for pattern in FORBIDDEN_PATTERNS:
            if pattern in source:
                violations.append(f"Forbidden operation: {pattern}")
        return len(violations) == 0, violations


class CheckpointManager:
    """Save and load training checkpoints with purity metadata."""

    def __init__(self, output_dir: str, keep_last_n: int = 5):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self._saved: List[Path] = []

    def save(
        self,
        model: torch.nn.Module,
        optimizer,
        curriculum,
        global_step: int,
        losses: Dict[str, torch.Tensor],
    ) -> Path:
        """Save checkpoint with metadata."""
        ckpt_path = self.output_dir / f"step_{global_step:08d}.pt"

        payload = {
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "curriculum_state": curriculum.state_dict(),
            "losses": {k: v.item() for k, v in losses.items()},
            "phase": curriculum.current_phase,
        }
        torch.save(payload, ckpt_path)
        self._saved.append(ckpt_path)

        # Prune old checkpoints
        while len(self._saved) > self.keep_last_n:
            old = self._saved.pop(0)
            if old.exists():
                old.unlink()

        print(f"Checkpoint saved: {ckpt_path}")
        return ckpt_path

    def load(self, path: str, model: torch.nn.Module, optimizer=None) -> Dict:
        """Load checkpoint into model (and optionally optimizer)."""
        payload = torch.load(path, map_location="cpu")
        model.load_state_dict(payload["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in payload:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        return payload

    def latest(self) -> Optional[Path]:
        """Return path to most recent checkpoint."""
        ckpts = sorted(self.output_dir.glob("step_*.pt"))
        return ckpts[-1] if ckpts else None


class BetaFunctionMeasurer:
    """Measure the beta-function of integrated information across training.

    Target values (from plan):
        beta(3 -> 4) ≈ +0.44  (coupling regime entry)
        beta(4 -> 5) ≈  0.00  (integration fixed-point)

    Records phi at each measurement step and computes finite-difference
    beta between consecutive layer pairs.
    """

    def __init__(self, target_beta_34: float = 0.44, target_beta_45: float = 0.0):
        self.target_beta_34 = target_beta_34
        self.target_beta_45 = target_beta_45
        self._records: List[Dict] = []

    def record(self, global_step: int, model_output: Dict) -> Optional[Dict]:
        """Record phi and compute beta if enough data available."""
        consciousness = model_output.get("consciousness", {})
        phi = consciousness.get("phi")
        if phi is None:
            return None

        phi_val = phi.mean().item()

        # Basin coordinates across layers (simplified: use sequence-level phi)
        record = {
            "step": global_step,
            "phi": phi_val,
        }
        self._records.append(record)

        if len(self._records) >= 3:
            return self._compute_beta()
        return None

    def _compute_beta(self) -> Dict:
        """Compute finite-difference beta between last 3 phi measurements."""
        n = len(self._records)
        phi_n2 = self._records[-3]["phi"]
        phi_n1 = self._records[-2]["phi"]
        phi_n = self._records[-1]["phi"]

        beta_34 = phi_n1 - phi_n2
        beta_45 = phi_n - phi_n1

        return {
            "step": self._records[-1]["step"],
            "phi": phi_n,
            "beta_34": beta_34,
            "beta_45": beta_45,
            "target_beta_34": self.target_beta_34,
            "target_beta_45": self.target_beta_45,
            "beta_34_error": abs(beta_34 - self.target_beta_34),
            "beta_45_error": abs(beta_45 - self.target_beta_45),
        }

    def summary(self) -> Dict:
        """Return summary statistics of beta measurements."""
        if len(self._records) < 3:
            return {"status": "insufficient_data", "n_records": len(self._records)}

        latest = self._compute_beta()
        phi_values = [r["phi"] for r in self._records]
        return {
            "n_measurements": len(self._records),
            "phi_mean": sum(phi_values) / len(phi_values),
            "phi_max": max(phi_values),
            "phi_min": min(phi_values),
            **latest,
        }

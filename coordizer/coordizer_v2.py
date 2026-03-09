"""coordizer_v2.py — Convert text/harvest data to 64-dimensional basin coordinates.

CoordizerV2 produces probability simplex coordinates (Δ⁶³) from:
  - Raw text (via vocabulary frequency projection)
  - vex-agent harvest data
  - Existing embeddings (re-projected to simplex, never kept as Euclidean)

Output format: .qktj.jsonl records ready for QIG-Native training.
"""

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from qig_kernel.simplex import project_to_simplex, fisher_rao_distance
from coordizer.resonance_bank import ResonanceBank


class CoordizerV2:
    """Convert arbitrary text to basin coordinates on Δ⁶³.

    The coordizer operates in two modes:
    1. Vocabulary-frequency mode: maps character/token frequency profiles
       to 64-dimensional simplex coordinates via frequency normalisation.
    2. Resonance-bank mode: looks up pre-computed basin attractors from
       the resonance bank for known concepts.

    No Euclidean distance or L2 normalisation is used at any step.
    """

    VERSION = "2.0.0"
    BASIN_DIM = 64

    def __init__(
        self,
        resonance_bank: Optional[ResonanceBank] = None,
        vocab_size: int = 512,
        use_resonance: bool = True,
    ):
        self.vocab_size = vocab_size
        self.use_resonance = use_resonance
        self.resonance_bank = resonance_bank or ResonanceBank()
        self._feature_vocab = self._build_feature_vocab()

    def _build_feature_vocab(self) -> List[str]:
        """Build a feature vocabulary for frequency projection."""
        # Character n-grams (common in language models)
        chars = list("abcdefghijklmnopqrstuvwxyz0123456789 .,!?-'"")
        bigrams = [a + b for a in "aeiou " for b in "aeiou "]
        return (chars + bigrams)[:self.vocab_size]

    def text_to_basin(self, text: str) -> np.ndarray:
        """Convert text to 64-dimensional basin coordinates.

        Args:
            text: input string

        Returns:
            basin: (64,) probability simplex coordinates
        """
        text_lower = text.lower()

        # Frequency profile over feature vocab
        freq = np.zeros(self.vocab_size, dtype=np.float64)
        for i, feat in enumerate(self._feature_vocab):
            freq[i] = text_lower.count(feat)

        # Add character distribution
        char_counts = Counter(text_lower)
        total_chars = max(sum(char_counts.values()), 1)
        for i, feat in enumerate(self._feature_vocab[:26]):
            freq[i] += char_counts.get(feat, 0) / total_chars

        # Project frequency profile to 64-dim simplex
        # Take 64-dimensional slice via Fisher-information-weighted projection
        step = max(1, self.vocab_size // self.BASIN_DIM)
        basin_raw = np.array([
            freq[i * step:(i + 1) * step].sum()
            for i in range(self.BASIN_DIM)
        ], dtype=np.float64)

        # Add small epsilon and normalise to simplex
        basin_raw += 1e-8
        basin = basin_raw / basin_raw.sum()

        # Optionally blend with resonance bank attractor
        if self.use_resonance:
            attractor = self.resonance_bank.nearest_attractor(basin)
            if attractor is not None:
                # Fréchet mean blend (sqrt-space, not arithmetic)
                sqrt_basin = np.sqrt(basin)
                sqrt_attr = np.sqrt(attractor)
                blended = (0.8 * sqrt_basin + 0.2 * sqrt_attr) ** 2
                basin = blended / blended.sum()

        return basin.astype(np.float32)

    def compute_geometry(
        self, input_basin: np.ndarray, target_basin: np.ndarray
    ) -> Dict[str, float]:
        """Compute geometric properties between input and target basin coords.

        Uses Fisher-Rao distance (not Euclidean).
        """
        b_in = torch.tensor(input_basin, dtype=torch.float32).unsqueeze(0)
        b_tgt = torch.tensor(target_basin, dtype=torch.float32).unsqueeze(0)

        fr_dist = fisher_rao_distance(b_in, b_tgt).item()

        # Curvature approximation via local second moment in sqrt-space
        sqrt_in = np.sqrt(input_basin + 1e-10)
        laplacian = np.sum(np.diff(sqrt_in) ** 2)
        curvature = float(laplacian)

        return {
            "fisher_distance_io": float(fr_dist),
            "curvature_at_input": curvature,
            "geodesic_length": float(fr_dist),  # equals FR distance on simplex
        }

    def to_qktj(
        self,
        input_text: str,
        target_text: str,
        phase: str = "phase0_identity",
        difficulty: float = 0.5,
        regime_target: str = "identity",
    ) -> Dict:
        """Convert a text pair to a QKTJ training record.

        Args:
            input_text: source text
            target_text: target/response text
            phase: curriculum phase name
            difficulty: float in [0, 1]
            regime_target: geometric regime label

        Returns:
            dict conforming to QKTJ v1.0 schema
        """
        from training.curriculum import CURRICULUM_PHASES

        phase_cfg = CURRICULUM_PHASES.get(phase, CURRICULUM_PHASES["phase0_identity"])

        input_basin = self.text_to_basin(input_text)
        target_basin = self.text_to_basin(target_text)
        geometry = self.compute_geometry(input_basin, target_basin)

        return {
            "version": "1.0",
            "phase": {
                "name": phase_cfg.get("description", phase),
                "ordinal": phase_cfg["ordinal"],
                "phi_floor": phase_cfg["phi_floor"],
            },
            "regime_target": regime_target,
            "content": {
                "input_text": input_text,
                "target_text": target_text,
                "input_basin_64d": input_basin.tolist(),
                "target_basin_64d": target_basin.tolist(),
            },
            "geometry": geometry,
            "curriculum": {
                "difficulty": difficulty,
                "prerequisites": [],
                "maturity_gate": f"phi_sustained_above_{phase_cfg['phi_floor']}",
            },
            "meta": {
                "coordizer_version": self.VERSION,
                "basin_dim": self.BASIN_DIM,
            },
        }

    def process_file(
        self,
        input_path: str,
        output_path: str,
        phase: str = "phase0_identity",
        text_col: str = "text",
    ) -> int:
        """Process a JSONL file of text records into QKTJ format.

        Args:
            input_path: path to input .jsonl file
            output_path: path to write QKTJ .jsonl output
            phase: curriculum phase
            text_col: field name containing text

        Returns:
            Number of records processed
        """
        count = 0
        with open(input_path) as fin, open(output_path, "w") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    text = record.get(text_col, "")
                    if not text:
                        continue
                    # Use same text for input/target (autoencoding mode)
                    qktj = self.to_qktj(
                        input_text=text,
                        target_text=text,
                        phase=phase,
                    )
                    fout.write(json.dumps(qktj) + "\n")
                    count += 1
                except (json.JSONDecodeError, Exception):
                    continue
        return count

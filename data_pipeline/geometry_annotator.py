"""geometry_annotator.py — Enrich QKTJ records with geometric metadata.

Takes existing QKTJ records and recomputes / enhances the geometry field
with more precise Fisher-Rao distances, curvature estimates, and
basin stability metrics.

Usage:
    python data_pipeline/geometry_annotator.py \
        --input data/qktj/phase0_identity/text.qktj.jsonl \
        --output data/qktj/phase0_identity/text.annotated.qktj.jsonl
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from qig_kernel.simplex import fisher_rao_distance, project_to_simplex


def compute_basin_curvature(basin: np.ndarray) -> float:
    """Estimate information curvature via second derivative in sqrt-space.

    The Fisher information metric induces curvature in the simplex.
    We approximate it by the Laplacian of the sqrt-space representation.
    """
    sqrt_basin = np.sqrt(basin + 1e-10)
    # Second derivative approximation (finite differences)
    d2 = np.diff(sqrt_basin, n=2)
    curvature = float(np.sum(d2 ** 2))
    return curvature


def compute_basin_stability(
    basin: np.ndarray, n_perturbations: int = 10, eps: float = 0.01
) -> float:
    """Estimate basin stability by measuring sensitivity to small perturbations.

    Perturbs the basin distribution and measures how far it moves in
    Fisher-Rao distance. More stable basins show smaller displacement.

    Returns:
        stability: float in [0, 1] where 1 = maximally stable
    """
    rng = np.random.default_rng(42)
    b_tensor = torch.tensor(basin, dtype=torch.float32).unsqueeze(0)

    displacements = []
    for _ in range(n_perturbations):
        # Small perturbation in sqrt-space
        noise = rng.normal(0, eps, basin.shape).astype(np.float32)
        perturbed = basin + noise
        perturbed = np.clip(perturbed, 1e-10, None)
        perturbed = perturbed / perturbed.sum()

        p_tensor = torch.tensor(perturbed, dtype=torch.float32).unsqueeze(0)
        dist = fisher_rao_distance(b_tensor, p_tensor).item()
        displacements.append(dist)

    mean_displacement = np.mean(displacements)
    # Convert to stability score (low displacement = high stability)
    max_expected_disp = eps * math.sqrt(basin.shape[0])
    stability = 1.0 - min(1.0, mean_displacement / max(max_expected_disp, 1e-6))
    return float(stability)


def annotate_record(record: Dict) -> Dict:
    """Enrich a single QKTJ record with enhanced geometry."""
    content = record.get("content", {})
    input_basin = np.array(content.get("input_basin_64d", [1.0/64]*64), dtype=np.float32)
    target_basin = np.array(content.get("target_basin_64d", [1.0/64]*64), dtype=np.float32)

    # Normalise
    input_basin = input_basin / input_basin.sum().clip(1e-10)
    target_basin = target_basin / target_basin.sum().clip(1e-10)

    # Fisher-Rao distance
    b_in = torch.tensor(input_basin).unsqueeze(0)
    b_tgt = torch.tensor(target_basin).unsqueeze(0)
    fr_dist = fisher_rao_distance(b_in, b_tgt).item()

    # Curvature
    curvature = compute_basin_curvature(input_basin)

    # Stability (lighter computation)
    stability = compute_basin_stability(input_basin, n_perturbations=5, eps=0.005)

    # Compute regime confidence (distance to uniform distribution)
    uniform = np.ones(64, dtype=np.float32) / 64
    b_uniform = torch.tensor(uniform).unsqueeze(0)
    dist_to_uniform = fisher_rao_distance(b_in, b_uniform).item()
    max_fr = math.pi
    regime_confidence = dist_to_uniform / max_fr

    # Update geometry
    record["geometry"] = {
        "fisher_distance_io": float(fr_dist),
        "curvature_at_input": float(curvature),
        "geodesic_length": float(fr_dist),
        "basin_stability": float(stability),
        "regime_confidence": float(regime_confidence),
        "dist_to_uniform": float(dist_to_uniform),
    }

    # Update curriculum difficulty based on geometry
    record["curriculum"]["difficulty"] = float(
        min(1.0, (fr_dist / max_fr + curvature * 10) / 2)
    )

    return record


def process_file(input_path: str, output_path: str, max_records: int = None) -> int:
    """Annotate a QKTJ JSONL file with enhanced geometry."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with open(input_path) as fin, open(output_path, "w") as fout:
        for line_num, line in enumerate(fin):
            if max_records is not None and count >= max_records:
                break
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                enriched = annotate_record(record)
                fout.write(json.dumps(enriched) + "\n")
                count += 1
                if count % 500 == 0:
                    print(f"  Annotated {count} records...", file=sys.stderr)
            except Exception as e:
                print(f"  Warning line {line_num}: {e}", file=sys.stderr)

    return count


def main():
    parser = argparse.ArgumentParser(description="Annotate QKTJ records with geometry")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max", type=int, default=None)
    args = parser.parse_args()

    count = process_file(args.input, args.output, args.max)
    print(f"Annotated {count} records -> {args.output}")


if __name__ == "__main__":
    main()

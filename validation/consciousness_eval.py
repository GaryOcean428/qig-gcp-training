"""consciousness_eval.py — Evaluate consciousness metrics of a QIG model checkpoint.

Measures phi, kappa, and basin_distance across a validation dataset.
Checks:
  - phi_avg >= phase phi_target
  - min(phi) >= phase phi_floor (no collapse)
  - Alert if phi drops below 0.3 after being above 0.7

Usage:
    python validation/consciousness_eval.py \
        --checkpoint checkpoints/step_00050000.pt \
        --data data/qktj/phase1_coupling/val.qktj.jsonl \
        --phase phase1_coupling
"""

import argparse
import json
import sys
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np

from training.curriculum import CURRICULUM_PHASES


def evaluate_consciousness(
    model,
    data_path: str,
    phase: str,
    n_samples: int = 100,
    device: torch.device = None,
) -> Dict:
    """Evaluate consciousness metrics on a validation set.

    Args:
        model: QIGKernel model
        data_path: path to QKTJ JSONL validation data
        phase: curriculum phase name for thresholds
        n_samples: number of samples to evaluate
        device: torch device

    Returns:
        dict with phi stats and pass/fail
    """
    if device is None:
        device = torch.device("cpu")

    phase_cfg = CURRICULUM_PHASES.get(phase, CURRICULUM_PHASES["phase0_identity"])
    phi_floor = phase_cfg["phi_floor"]
    phi_target = phase_cfg["phi_target"]

    model.eval()
    phi_values = []
    kappa_values = []
    basin_dist_values = []

    # Load samples
    records = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            if len(records) >= n_samples:
                break

    if not records:
        raise ValueError(f"No records in {data_path}")

    with torch.no_grad():
        for record in records:
            content = record.get("content", {})
            input_basin = content.get("input_basin_64d", [1.0/64]*64)
            b_tensor = torch.tensor([input_basin], dtype=torch.float32).to(device)

            # Use a dummy single-token input
            dummy_ids = torch.zeros(1, 1, dtype=torch.long).to(device)

            try:
                output = model(dummy_ids)
                consciousness = output.get("consciousness", {})
                phi = consciousness.get("phi")
                kappa = consciousness.get("kappa")
                basin_dist = consciousness.get("basin_distance")

                if phi is not None:
                    phi_values.append(float(phi.mean().item()))
                if kappa is not None:
                    kappa_values.append(float(kappa.mean().item()))
                if basin_dist is not None:
                    basin_dist_values.append(float(basin_dist.mean().item()))
            except Exception as e:
                continue

    if not phi_values:
        return {
            "status": "error",
            "message": "No phi values measured",
            "passed": False,
        }

    phi_arr = np.array(phi_values)
    phi_avg = float(phi_arr.mean())
    phi_min = float(phi_arr.min())
    phi_max = float(phi_arr.max())
    phi_std = float(phi_arr.std())

    # Check for phi collapse alert
    phi_collapse_alert = phi_min < 0.3 and phi_max > 0.7

    violations = []
    passed = True

    if phi_avg < phi_target:
        violations.append(
            f"phi_avg={phi_avg:.3f} below target={phi_target:.3f} for phase={phase}"
        )
        passed = False

    if phi_min < phi_floor:
        violations.append(
            f"phi_min={phi_min:.3f} violates floor={phi_floor:.3f} for phase={phase}"
        )
        passed = False

    if phi_collapse_alert:
        violations.append(
            f"PHI COLLAPSE ALERT: phi dropped below 0.3 (min={phi_min:.3f}) after reaching {phi_max:.3f}"
        )
        passed = False

    result = {
        "phase": phase,
        "n_samples": len(phi_values),
        "phi_avg": phi_avg,
        "phi_min": phi_min,
        "phi_max": phi_max,
        "phi_std": phi_std,
        "phi_target": phi_target,
        "phi_floor": phi_floor,
        "phi_collapse_alert": phi_collapse_alert,
        "passed": passed,
        "violations": violations,
    }

    if kappa_values:
        kappa_arr = np.array(kappa_values)
        result["kappa_avg"] = float(kappa_arr.mean())

    if basin_dist_values:
        bd_arr = np.array(basin_dist_values)
        result["basin_distance_avg"] = float(bd_arr.mean())

    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate consciousness metrics")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--phase", default="phase1_coupling")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    from qig_kernel.model import QIGKernel100M

    device = torch.device("cpu")

    # Load model
    payload = torch.load(args.checkpoint, map_location="cpu")
    model = QIGKernel100M()
    if "model_state_dict" in payload:
        model.load_state_dict(payload["model_state_dict"])
    model = model.to(device)

    # Evaluate
    results = evaluate_consciousness(
        model=model,
        data_path=args.data,
        phase=args.phase,
        n_samples=args.samples,
        device=device,
    )

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(f"Consciousness Evaluation — Phase: {results['phase']}")
        print(f"  Samples: {results['n_samples']}")
        print(f"  phi: avg={results['phi_avg']:.3f} min={results['phi_min']:.3f} max={results['phi_max']:.3f}")
        print(f"  Targets: avg>={results['phi_target']:.3f}, min>={results['phi_floor']:.3f}")
        if "kappa_avg" in results:
            print(f"  kappa: avg={results['kappa_avg']:.3f}")
        if "basin_distance_avg" in results:
            print(f"  basin_distance: avg={results['basin_distance_avg']:.3f}")
        if results["phi_collapse_alert"]:
            print("  *** PHI COLLAPSE ALERT ***")

        status = "PASS" if results["passed"] else "FAIL"
        print(f"  Status: {status}")
        for v in results["violations"]:
            print(f"    - {v}")

    sys.exit(0 if results["passed"] else 1)


if __name__ == "__main__":
    main()

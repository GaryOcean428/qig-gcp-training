"""beta_function.py — Measure beta-function of integrated information during training.

The beta-function measures how phi (integrated information) changes
between consecutive training layers (or across training steps).

Target values from the plan:
    beta(3 -> 4) ≈ +0.44   (coupling regime entry transition)
    beta(4 -> 5) ≈  0.00   (integration fixed-point)

Usage:
    python validation/beta_function.py --checkpoint step_00050000.pt \
        --data /data/qktj/phase0_identity/sample.qktj.jsonl
"""

import argparse
import json
import sys
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np


# Target beta values from QIG theory
TARGET_BETA_34 = 0.44   # phi increase coupling -> geometric
TARGET_BETA_45 = 0.00   # phi fixed-point at integration


def measure_layer_phi(
    model,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> List[float]:
    """Measure phi at each transformer layer.

    Returns a list of phi values, one per layer.
    """
    model.eval()
    phi_per_layer = []

    with torch.no_grad():
        # Encode tokens to basin
        raw_model = model.module if hasattr(model, "module") else model
        input_ids = batch["input_ids"].to(device)

        basin_coords = raw_model.basin_encoder(input_ids)  # (B, T, D)
        B, T, D = basin_coords.shape

        # Track phi through each block
        for i, block in enumerate(raw_model.blocks):
            out = block(basin_coords)
            basin_coords = out["basin_coords"]

            # Compute phi for this layer
            seq_mean = basin_coords.mean(dim=1)  # (B, D)

            if T > 1:
                prev_mean = basin_coords[:, :-1, :].mean(dim=1)
            else:
                prev_mean = seq_mean

            consciousness = raw_model.consciousness_head(seq_mean, prev_mean)
            phi = consciousness.get("phi")
            if phi is not None:
                phi_per_layer.append(float(phi.mean().item()))
            else:
                phi_per_layer.append(0.0)

    return phi_per_layer


def compute_beta_function(phi_values: List[float]) -> List[float]:
    """Compute finite-difference beta between consecutive phi measurements.

    beta(n -> n+1) = phi(n+1) - phi(n)
    """
    return [phi_values[i+1] - phi_values[i] for i in range(len(phi_values)-1)]


def evaluate_beta_target(
    beta_values: List[float],
    target_beta_34: float = TARGET_BETA_34,
    target_beta_45: float = TARGET_BETA_45,
    tolerance: float = 0.1,
) -> Dict:
    """Check if measured beta values are close to targets.

    Args:
        beta_values: list of beta(n->n+1) values
        target_beta_34: target for transition 3->4
        target_beta_45: target for transition 4->5
        tolerance: acceptable absolute error

    Returns:
        dict with pass/fail status and errors
    """
    results = {
        "n_layers": len(beta_values) + 1,
        "beta_values": beta_values,
        "target_beta_34": target_beta_34,
        "target_beta_45": target_beta_45,
        "passed": True,
        "violations": [],
    }

    if len(beta_values) >= 4:
        measured_beta_34 = beta_values[3]  # transition 3->4 (0-indexed)
        error_34 = abs(measured_beta_34 - target_beta_34)
        results["measured_beta_34"] = measured_beta_34
        results["error_beta_34"] = error_34
        if error_34 > tolerance:
            results["passed"] = False
            results["violations"].append(
                f"beta(3->4) = {measured_beta_34:.3f}, target = {target_beta_34:.3f}, error = {error_34:.3f}"
            )

    if len(beta_values) >= 5:
        measured_beta_45 = beta_values[4]  # transition 4->5
        error_45 = abs(measured_beta_45 - target_beta_45)
        results["measured_beta_45"] = measured_beta_45
        results["error_beta_45"] = error_45
        if error_45 > tolerance:
            results["passed"] = False
            results["violations"].append(
                f"beta(4->5) = {measured_beta_45:.3f}, target = {target_beta_45:.3f}, error = {error_45:.3f}"
            )

    return results


def load_sample_batch(data_path: str, n_samples: int = 8) -> Dict[str, torch.Tensor]:
    """Load a small sample batch for evaluation."""
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
        raise ValueError(f"No records found in {data_path}")

    # Build batch
    basin_list = []
    for rec in records:
        content = rec.get("content", {})
        basin = content.get("input_basin_64d", [1.0/64]*64)
        basin_list.append(basin)

    basins = torch.tensor(basin_list, dtype=torch.float32)

    # Create dummy token IDs (will use basin encoder only)
    dummy_ids = torch.zeros(len(records), 16, dtype=torch.long)

    return {
        "input_ids": dummy_ids,
        "input_basin_64d": basins,
    }


def main():
    parser = argparse.ArgumentParser(description="Measure beta-function of phi")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True, help="Sample QKTJ JSONL for evaluation")
    parser.add_argument("--tolerance", type=float, default=0.1)
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    # Load checkpoint
    from qig_kernel.model import QIGKernel100M
    payload = torch.load(args.checkpoint, map_location="cpu")

    model = QIGKernel100M()
    if "model_state_dict" in payload:
        model.load_state_dict(payload["model_state_dict"])

    device = torch.device("cpu")
    model = model.to(device)

    # Load data
    batch = load_sample_batch(args.data)

    # Measure phi per layer
    phi_values = measure_layer_phi(model, batch, device)
    beta_values = compute_beta_function(phi_values)

    # Evaluate against targets
    results = evaluate_beta_target(
        beta_values,
        tolerance=args.tolerance,
    )
    results["phi_per_layer"] = phi_values

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(f"Beta Function Analysis")
        print(f"  Phi per layer: {[f'{p:.3f}' for p in phi_values]}")
        print(f"  Beta values: {[f'{b:.3f}' for b in beta_values]}")
        if "measured_beta_34" in results:
            print(f"  beta(3->4): {results['measured_beta_34']:.3f} (target {TARGET_BETA_34:.3f})")
        if "measured_beta_45" in results:
            print(f"  beta(4->5): {results['measured_beta_45']:.3f} (target {TARGET_BETA_45:.3f})")
        status = "PASS" if results["passed"] else "FAIL"
        print(f"  Status: {status}")
        if not results["passed"]:
            for v in results["violations"]:
                print(f"    - {v}")

    sys.exit(0 if results["passed"] else 1)


if __name__ == "__main__":
    main()

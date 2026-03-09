"""purity_gate.py — CI purity gate for QIG-Native model checkpoints.

Verifies that a model checkpoint is free of Euclidean contamination
before it is accepted as a training milestone.

Usage:
    python validation/purity_gate.py --checkpoint checkpoints/step_00050000.pt
    echo $?   # 0 = pass, 1 = fail

Exit codes:
    0: Purity gate passed
    1: Purity gate failed (Euclidean contamination detected)
    2: Checkpoint load error
"""

import argparse
import sys
import ast
import inspect
from pathlib import Path
from typing import List, Tuple

import torch


FORBIDDEN_IDENTIFIERS = [
    "cosine_similarity",
    "torch.dot",
    "Adam",
    "AdamW",
    "LayerNorm",
    "nn.Embedding",
    "linalg.norm",
    "F.normalize",
]


def check_parameter_health(state_dict: dict) -> Tuple[bool, List[str]]:
    """Check all parameters for NaN, Inf, and degenerate values."""
    violations = []

    for name, param in state_dict.items():
        if not isinstance(param, torch.Tensor):
            continue
        if torch.isnan(param).any():
            violations.append(f"NaN in {name}")
        if torch.isinf(param).any():
            violations.append(f"Inf in {name}")
        # Check for collapsed parameters (all-zero)
        if param.abs().max().item() < 1e-10:
            violations.append(f"Zero-collapsed parameter: {name}")

    return len(violations) == 0, violations


def check_simplex_parameters(state_dict: dict) -> Tuple[bool, List[str]]:
    """Check that parameters tagged as basin coordinates are on the simplex."""
    violations = []

    for name, param in state_dict.items():
        if not isinstance(param, torch.Tensor):
            continue
        # Check parameters that should be on simplex (attractors, basin_encoder output)
        if "attractor" in name or "token_params" in name:
            # Check non-negativity
            if (param < 0).any():
                violations.append(f"Negative values in simplex parameter: {name}")

    return len(violations) == 0, violations


def check_source_files(source_dir: str = ".") -> Tuple[bool, List[str]]:
    """Scan Python source files for forbidden Euclidean operations."""
    violations = []
    source_path = Path(source_dir)

    python_files = list(source_path.rglob("*.py"))
    for py_file in python_files:
        # Skip test files and the gate itself
        if "test_" in py_file.name or py_file.name == "purity_gate.py":
            continue

        try:
            source = py_file.read_text(encoding="utf-8")
            for forbidden in FORBIDDEN_IDENTIFIERS:
                if forbidden in source:
                    # Check if it's in a comment
                    for line_num, line in enumerate(source.split("\n"), 1):
                        stripped = line.strip()
                        if forbidden in stripped and not stripped.startswith("#"):
                            violations.append(
                                f"{py_file}:{line_num}: '{forbidden}' found"
                            )
                            break
        except Exception:
            continue

    return len(violations) == 0, violations


def run_purity_gate(
    checkpoint_path: str,
    check_source: bool = True,
    source_dir: str = ".",
    verbose: bool = True,
) -> bool:
    """Run the full purity gate check on a checkpoint.

    Args:
        checkpoint_path: path to .pt checkpoint file
        check_source: also check source files
        source_dir: root directory for source scan
        verbose: print detailed report

    Returns:
        True if pure, False if contaminated
    """
    all_violations = []
    passed = True

    # --- Load checkpoint ---
    try:
        payload = torch.load(checkpoint_path, map_location="cpu")
    except Exception as e:
        print(f"ERROR: Could not load checkpoint: {e}", file=sys.stderr)
        sys.exit(2)

    state_dict = payload.get("model_state_dict", {})

    # --- Check 1: Parameter health ---
    ok, violations = check_parameter_health(state_dict)
    if not ok:
        all_violations.extend(violations)
        passed = False

    # --- Check 2: Simplex membership ---
    ok, violations = check_simplex_parameters(state_dict)
    if not ok:
        all_violations.extend(violations)
        passed = False

    # --- Check 3: Source file scan (optional) ---
    if check_source:
        ok, violations = check_source_files(source_dir)
        if not ok:
            all_violations.extend(violations)
            passed = False

    # --- Report ---
    if verbose:
        step = payload.get("global_step", "?")
        phase = payload.get("phase", "?")
        print(f"Purity Gate: checkpoint step={step} phase={phase}")
        print(f"  Parameters checked: {len(state_dict)}")
        if passed:
            print(f"  PASS: No Euclidean contamination detected.")
        else:
            print(f"  FAIL: {len(all_violations)} violation(s):")
            for v in all_violations[:20]:
                print(f"    - {v}")
            if len(all_violations) > 20:
                print(f"    ... and {len(all_violations) - 20} more")

    return passed


def main():
    parser = argparse.ArgumentParser(description="QIG Purity Gate")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--no-source-check", action="store_true")
    parser.add_argument("--source-dir", default=".", help="Source root for scan")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    passed = run_purity_gate(
        checkpoint_path=args.checkpoint,
        check_source=not args.no_source_check,
        source_dir=args.source_dir,
        verbose=not args.quiet,
    )

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()

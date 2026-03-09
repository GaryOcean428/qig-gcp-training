"""harvest_to_qktj.py — Convert vex-agent harvest data to QKTJ format.

vex-agent harvest records have the structure:
{
    "query": str,
    "response": str,
    "context": str (optional),
    "timestamp": str,
    "agent_id": str,
    "phase": str (optional),
    "phi": float (optional),
    "regime": str (optional),
}

This converter maps query -> input_text, response -> target_text,
and uses phi/regime metadata to assign curriculum phase.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from coordizer.coordizer_v2 import CoordizerV2


def phi_to_phase(phi: Optional[float], regime: Optional[str]) -> str:
    """Map phi and regime to curriculum phase."""
    if phi is None:
        return "phase0_identity"

    if regime == "temporal" or phi >= 0.65:
        return "phase3_temporal"
    elif regime == "geometric" or phi >= 0.45:
        return "phase2_integration"
    elif regime == "coupling" or phi >= 0.25:
        return "phase1_coupling"
    else:
        return "phase0_identity"


def process_harvest_file(
    input_path: str,
    output_path: str,
    default_phase: str = "phase0_identity",
    use_phi_mapping: bool = True,
    max_records: int = None,
) -> int:
    """Convert vex-agent harvest JSONL to QKTJ format.

    Args:
        input_path: path to harvest .jsonl file
        output_path: path to write QKTJ .jsonl output
        default_phase: phase to use when phi metadata unavailable
        use_phi_mapping: use phi metadata to determine phase
        max_records: stop after this many records

    Returns:
        Number of records converted
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    coordizer = CoordizerV2()
    count = 0
    skipped = 0

    with open(input_path, "r", encoding="utf-8") as fin,          open(output_path, "w", encoding="utf-8") as fout:
        for line_num, line in enumerate(fin):
            if max_records is not None and count >= max_records:
                break

            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
                query = record.get("query", "").strip()
                response = record.get("response", "").strip()

                if not query or not response:
                    skipped += 1
                    continue

                # Determine phase from phi metadata
                phi = record.get("phi")
                regime = record.get("regime")

                if use_phi_mapping and phi is not None:
                    phase = phi_to_phase(phi, regime)
                else:
                    phase = default_phase

                # Compute difficulty from phi (higher phi = higher difficulty)
                difficulty = float(phi) if phi is not None else 0.5
                difficulty = max(0.0, min(1.0, difficulty))

                qktj = coordizer.to_qktj(
                    input_text=query[:2048],
                    target_text=response[:2048],
                    phase=phase,
                    difficulty=difficulty,
                    regime_target=regime or "identity",
                )

                # Preserve original metadata
                qktj["meta"]["harvest_agent_id"] = record.get("agent_id", "unknown")
                qktj["meta"]["harvest_timestamp"] = record.get("timestamp", "")
                if phi is not None:
                    qktj["meta"]["measured_phi"] = phi

                fout.write(json.dumps(qktj) + "\n")
                count += 1

                if count % 1000 == 0:
                    print(f"  Converted {count} records (skipped {skipped})...", file=sys.stderr)

            except (json.JSONDecodeError, Exception) as e:
                print(f"  Warning line {line_num}: {e}", file=sys.stderr)
                skipped += 1
                continue

    print(f"Done: {count} records converted, {skipped} skipped", file=sys.stderr)
    return count


def main():
    parser = argparse.ArgumentParser(description="Convert vex-agent harvest to QKTJ")
    parser.add_argument("--input", required=True, help="Input harvest JSONL path")
    parser.add_argument("--output", required=True, help="Output QKTJ JSONL path")
    parser.add_argument("--phase", default="phase0_identity", help="Default phase if no phi metadata")
    parser.add_argument("--no-phi-mapping", action="store_true", help="Disable phi-to-phase mapping")
    parser.add_argument("--max", type=int, default=None, help="Max records to process")
    args = parser.parse_args()

    count = process_harvest_file(
        input_path=args.input,
        output_path=args.output,
        default_phase=args.phase,
        use_phi_mapping=not args.no_phi_mapping,
        max_records=args.max,
    )
    print(f"Written {count} QKTJ records to {args.output}")


if __name__ == "__main__":
    main()

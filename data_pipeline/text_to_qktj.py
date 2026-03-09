"""text_to_qktj.py — Convert raw text files to QKTJ format.

Usage:
    python data_pipeline/text_to_qktj.py \
        --input /data/raw/text.jsonl \
        --output /data/qktj/phase0_identity/text.qktj.jsonl \
        --phase phase0_identity
"""

import argparse
import json
import sys
from pathlib import Path

from coordizer.coordizer_v2 import CoordizerV2


def process_text_file(
    input_path: str,
    output_path: str,
    phase: str = "phase0_identity",
    text_field: str = "text",
    max_records: int = None,
) -> int:
    """Convert a JSONL text file to QKTJ format.

    Args:
        input_path: path to input JSONL (one JSON object per line with text_field)
        output_path: path to write QKTJ JSONL output
        phase: curriculum phase name
        text_field: field containing text (default: 'text')
        max_records: stop after this many records (None = all)

    Returns:
        Number of records converted
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    coordizer = CoordizerV2()
    count = 0

    with open(input_path, "r", encoding="utf-8") as fin,          open(output_path, "w", encoding="utf-8") as fout:
        for line_num, line in enumerate(fin):
            if max_records is not None and count >= max_records:
                break

            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
                text = record.get(text_field, "")
                if not text or len(text.strip()) < 10:
                    continue

                # Use text as both input and target (language model auto-encoding)
                qktj = coordizer.to_qktj(
                    input_text=text[:2048],    # cap to avoid OOM
                    target_text=text[:2048],
                    phase=phase,
                    difficulty=min(1.0, len(text) / 1000.0),
                )

                fout.write(json.dumps(qktj) + "\n")
                count += 1

                if count % 1000 == 0:
                    print(f"  Processed {count} records...", file=sys.stderr)

            except (json.JSONDecodeError, Exception) as e:
                print(f"  Warning line {line_num}: {e}", file=sys.stderr)
                continue

    print(f"Converted {count} records to {output_path}", file=sys.stderr)
    return count


def main():
    parser = argparse.ArgumentParser(description="Convert text JSONL to QKTJ format")
    parser.add_argument("--input", required=True, help="Input JSONL file path")
    parser.add_argument("--output", required=True, help="Output QKTJ JSONL file path")
    parser.add_argument("--phase", default="phase0_identity", help="Curriculum phase")
    parser.add_argument("--field", default="text", help="Text field name in input JSON")
    parser.add_argument("--max", type=int, default=None, help="Max records to process")
    args = parser.parse_args()

    count = process_text_file(
        input_path=args.input,
        output_path=args.output,
        phase=args.phase,
        text_field=args.field,
        max_records=args.max,
    )
    print(f"Done: {count} records written to {args.output}")


if __name__ == "__main__":
    main()

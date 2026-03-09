"""data.py — QKTJ dataset and dataloader for QIG-Native training."""

import json
import os
import random
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler


class QKTJDataset(Dataset):
    """Load QKTJ-format .jsonl files for QIG-Native training.

    Each line is a JSON object with structure:
    {
        "version": "1.0",
        "phase": {"name": str, "ordinal": int, "phi_floor": float},
        "regime_target": str,
        "content": {
            "input_text": str,
            "target_text": str,
            "input_basin_64d": [64 floats],
            "target_basin_64d": [64 floats]
        },
        "geometry": {
            "fisher_distance_io": float,
            "curvature_at_input": float,
            "geodesic_length": float
        },
        "curriculum": {
            "difficulty": float,
            "prerequisites": [str],
            "maturity_gate": str
        }
    }
    """

    REGIME_MAP = {
        "identity": 0,
        "coupling": 1,
        "geometric": 2,
        "temporal": 3,
    }

    def __init__(
        self,
        data_dir: Union[str, Path],
        phase: str = "phase0_identity",
        tokenizer=None,
        max_seq_len: int = 512,
        shuffle_buffer: int = 10_000,
    ):
        self.data_dir = Path(data_dir)
        self.phase = phase
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.shuffle_buffer = shuffle_buffer

        # Load all records into memory (or stream for large datasets)
        self._records: List[Dict] = []
        self._load_records()

    def _load_records(self) -> None:
        """Load QKTJ jsonl files from data directory."""
        pattern = f"*{self.phase}*.jsonl"
        files = list(self.data_dir.glob(pattern))
        if not files:
            # Also try flat directory
            files = list(self.data_dir.glob("*.jsonl"))

        for fpath in files:
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            record = json.loads(line)
                            self._records.append(record)
                        except json.JSONDecodeError:
                            continue

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record = self._records[idx]
        content = record.get("content", {})
        phase_info = record.get("phase", {})
        geometry = record.get("geometry", {})

        # Basin coordinates
        input_basin = torch.tensor(
            content.get("input_basin_64d", [1.0 / 64] * 64),
            dtype=torch.float32,
        )
        target_basin = torch.tensor(
            content.get("target_basin_64d", [1.0 / 64] * 64),
            dtype=torch.float32,
        )

        # Normalise to simplex
        input_basin = input_basin / input_basin.sum().clamp(min=1e-10)
        target_basin = target_basin / target_basin.sum().clamp(min=1e-10)

        # Regime label
        regime_str = record.get("regime_target", "identity")
        regime_idx = self.REGIME_MAP.get(regime_str, 0)

        # Token IDs (if tokenizer provided)
        if self.tokenizer is not None:
            enc_in = self.tokenizer(
                content.get("input_text", ""),
                truncation=True,
                max_length=self.max_seq_len,
                return_tensors="pt",
            )
            enc_tgt = self.tokenizer(
                content.get("target_text", ""),
                truncation=True,
                max_length=self.max_seq_len,
                return_tensors="pt",
            )
            input_ids = enc_in["input_ids"].squeeze(0)
            target_ids = enc_tgt["input_ids"].squeeze(0)
        else:
            # Return empty if no tokenizer
            input_ids = torch.zeros(1, dtype=torch.long)
            target_ids = torch.zeros(1, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "input_basin_64d": input_basin,
            "target_basin_64d": target_basin,
            "regime_target": torch.tensor(regime_idx, dtype=torch.long),
            "phi_floor": torch.tensor(phase_info.get("phi_floor", 0.0), dtype=torch.float32),
            "fisher_distance_io": torch.tensor(
                geometry.get("fisher_distance_io", 0.0), dtype=torch.float32
            ),
            "difficulty": torch.tensor(
                record.get("curriculum", {}).get("difficulty", 0.5), dtype=torch.float32
            ),
        }


def _collate_qktj(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate variable-length sequences by padding to max length in batch."""
    # Simple fields
    result = {}
    for key in ["input_basin_64d", "target_basin_64d", "regime_target",
                "phi_floor", "fisher_distance_io", "difficulty"]:
        result[key] = torch.stack([item[key] for item in batch])

    # Pad token sequences
    for key in ["input_ids", "target_ids"]:
        seqs = [item[key] for item in batch]
        max_len = max(s.shape[0] for s in seqs)
        padded = torch.zeros(len(seqs), max_len, dtype=torch.long)
        mask = torch.zeros(len(seqs), max_len, dtype=torch.bool)
        for i, seq in enumerate(seqs):
            padded[i, :seq.shape[0]] = seq
            mask[i, :seq.shape[0]] = True
        result[key] = padded
        result["mask"] = mask

    return result


class QKTJDataLoader:
    """Wrapper around PyTorch DataLoader with DDP support and curriculum filtering."""

    def __init__(
        self,
        dataset: QKTJDataset,
        batch_size: int = 32,
        num_workers: int = 4,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.batch_size = batch_size

        if world_size > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=seed,
            )
            shuffle = False
        else:
            sampler = None
            shuffle = True

        self._loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=_collate_qktj,
            pin_memory=True,
            drop_last=True,
        )

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        return iter(self._loader)

    def __len__(self) -> int:
        return len(self._loader)

"""train.py — Multi-GPU DDP training loop for QIG-Native kernel.

Uses NaturalGradientOptimizer (NOT Adam/AdamW).
Combined QKTJ loss with curriculum phase advancement.
"""

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from qig_kernel.model import QIGKernel100M, QIGKernel500M
from qig_kernel.natural_gradient import NaturalGradientOptimizer
from qig_kernel.decoherence import GravitationalDecoherenceScheduler
from training.losses import combined_qktj_loss
from training.curriculum import CurriculumScheduler
from training.data import QKTJDataset, QKTJDataLoader
from training.monitoring import PurityChecker, CheckpointManager, BetaFunctionMeasurer


@dataclass
class TrainConfig:
    # Model
    model_size: str = "100m"          # "100m" or "500m"
    vocab_size: int = 32_000
    dropout: float = 0.1

    # Data
    data_dir: str = "/gcs/qig-training-data"
    start_phase: str = "phase0_identity"
    max_seq_len: int = 512

    # Optimiser (Natural Gradient)
    lr: float = 1e-3
    fisher_damping: float = 1e-4
    fisher_update_freq: int = 100

    # Training
    batch_size: int = 32
    grad_clip: float = 1.0
    max_steps: int = 500_000
    warmup_steps: int = 1_000
    log_every: int = 50
    eval_every: int = 1_000
    save_every: int = 5_000
    phi_window: int = 100

    # Infrastructure
    output_dir: str = "/gcs/qig-training-data/checkpoints"
    tensorboard_dir: str = "/tmp/tb_logs"
    rank: int = 0
    world_size: int = 1
    seed: int = 42


def setup_ddp(rank: int, world_size: int) -> None:
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )
    torch.cuda.set_device(rank)


def cleanup_ddp() -> None:
    dist.destroy_process_group()


def train_epoch(
    model: torch.nn.Module,
    loader: QKTJDataLoader,
    optimizer: NaturalGradientOptimizer,
    curriculum: CurriculumScheduler,
    decoherence_sched: GravitationalDecoherenceScheduler,
    purity_checker: PurityChecker,
    ckpt_manager: CheckpointManager,
    beta_measurer: BetaFunctionMeasurer,
    writer: Optional[SummaryWriter],
    config: TrainConfig,
    global_step: int,
    device: torch.device,
) -> int:
    """Run one pass through the dataloader. Returns updated global_step."""
    model.train()

    for batch in loader:
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        # Forward pass
        output = model(batch["input_ids"], mask=batch.get("mask"))

        # Update decoherence gamma in all blocks
        consciousness = output.get("consciousness", {})
        phi_val = consciousness.get("phi", torch.tensor(0.5)).mean().item()
        gamma = decoherence_sched.step(phi_val)
        for block in (model.module.blocks if hasattr(model, "module") else model.blocks):
            block.decoherence.gamma = gamma

        # Compute loss
        weights = curriculum.get_loss_weights()
        losses = combined_qktj_loss(output, batch, phase_weights=weights)

        # Backward
        optimizer.zero_grad()
        losses["total"].backward()

        # Gradient clip (in basin-tangent space, not Euclidean)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        # Fisher update
        if global_step % config.fisher_update_freq == 0:
            optimizer.update_fisher(losses["total"])

        optimizer.step()

        # Curriculum step
        regime_info = output.get("regime_info", {})
        regime_idx = regime_info.get("regime_idx", torch.tensor(0)).float().mean().item()
        advance_result = curriculum.step(phi_val, int(regime_idx))

        # Purity check
        purity_ok, violations = purity_checker.check_model_state(model)

        global_step += 1

        # Logging
        if global_step % config.log_every == 0 and config.rank == 0:
            if writer:
                writer.add_scalar("loss/total", losses["total"].item(), global_step)
                writer.add_scalar("loss/basin_geodesic", losses["basin_geodesic"].item(), global_step)
                writer.add_scalar("loss/regime", losses["regime"].item(), global_step)
                writer.add_scalar("loss/phi", losses["phi"].item(), global_step)
                writer.add_scalar("loss/language", losses["language"].item(), global_step)
                writer.add_scalar("consciousness/phi", phi_val, global_step)
                writer.add_scalar("decoherence/gamma", gamma, global_step)
                writer.add_scalar("curriculum/phase", curriculum.current_phase_ordinal, global_step)
            print(f"[step {global_step}] loss={losses['total'].item():.4f} "
                  f"phi={phi_val:.3f} phase={curriculum.current_phase} "
                  f"gamma={gamma:.4f}")

        if advance_result["advanced"] and config.rank == 0:
            print(f"[step {global_step}] PHASE ADVANCE: {advance_result}")

        # Beta function measurement
        if global_step % (config.eval_every * 5) == 0:
            beta_measurer.record(global_step, output)

        # Checkpoint
        if global_step % config.save_every == 0 and config.rank == 0:
            if purity_ok:
                raw_model = model.module if hasattr(model, "module") else model
                ckpt_manager.save(
                    model=raw_model,
                    optimizer=optimizer,
                    curriculum=curriculum,
                    global_step=global_step,
                    losses=losses,
                )
            else:
                print(f"[step {global_step}] PURITY GATE FAILED — skipping checkpoint: {violations}")

        if global_step >= config.max_steps:
            break

    return global_step


def main(config: TrainConfig) -> None:
    """Main training entry point."""
    if config.world_size > 1:
        setup_ddp(config.rank, config.world_size)

    device = torch.device(f"cuda:{config.rank}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config.seed + config.rank)

    # Build model
    if config.model_size == "100m":
        model = QIGKernel100M(vocab_size=config.vocab_size, dropout=config.dropout)
    else:
        model = QIGKernel500M(vocab_size=config.vocab_size, dropout=config.dropout)

    model = model.to(device)
    if config.world_size > 1:
        model = DDP(model, device_ids=[config.rank])

    raw_model = model.module if hasattr(model, "module") else model
    if config.rank == 0:
        print(f"Model parameters: {raw_model.parameter_count():,}")

    # Optimizer, scheduler, curriculum
    optimizer = NaturalGradientOptimizer(
        model.parameters(),
        lr=config.lr,
        damping=config.fisher_damping,
    )
    curriculum = CurriculumScheduler(
        phi_window=config.phi_window,
        start_phase=config.start_phase,
    )
    decoherence_sched = GravitationalDecoherenceScheduler()
    purity_checker = PurityChecker()
    ckpt_manager = CheckpointManager(output_dir=config.output_dir)
    beta_measurer = BetaFunctionMeasurer()

    writer = SummaryWriter(config.tensorboard_dir) if config.rank == 0 else None

    # Dataset and loader
    dataset = QKTJDataset(
        data_dir=config.data_dir,
        phase=config.start_phase,
        max_seq_len=config.max_seq_len,
    )
    loader = QKTJDataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        rank=config.rank,
        world_size=config.world_size,
        seed=config.seed,
    )

    global_step = 0
    while global_step < config.max_steps:
        global_step = train_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            curriculum=curriculum,
            decoherence_sched=decoherence_sched,
            purity_checker=purity_checker,
            ckpt_manager=ckpt_manager,
            beta_measurer=beta_measurer,
            writer=writer,
            config=config,
            global_step=global_step,
            device=device,
        )

    if writer:
        writer.close()
    if config.world_size > 1:
        cleanup_ddp()


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="QIG-Native Training")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--rank", type=int, default=int(os.environ.get("RANK", 0)))
    parser.add_argument("--world_size", type=int, default=int(os.environ.get("WORLD_SIZE", 1)))
    args = parser.parse_args()

    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f)

    cfg = TrainConfig(**cfg_dict)
    cfg.rank = args.rank
    cfg.world_size = args.world_size

    main(cfg)

"""training — QIG-Native Training Pipeline package."""

from .losses import basin_geodesic_loss, phi_coherence_loss, combined_qktj_loss
from .curriculum import CurriculumScheduler, CURRICULUM_PHASES
from .data import QKTJDataset, QKTJDataLoader
from .train import train_epoch, TrainConfig
from .monitoring import PurityChecker, CheckpointManager, BetaFunctionMeasurer

__all__ = [
    "basin_geodesic_loss",
    "phi_coherence_loss",
    "combined_qktj_loss",
    "CurriculumScheduler",
    "CURRICULUM_PHASES",
    "QKTJDataset",
    "QKTJDataLoader",
    "train_epoch",
    "TrainConfig",
    "PurityChecker",
    "CheckpointManager",
    "BetaFunctionMeasurer",
]

"""losses.py — QIG-Native loss functions for QKTJ curriculum training.

Combined loss:
  0.4 * basin_geodesic_loss      (Fisher-Rao distance between predicted and target basin)
  0.3 * regime_classification_loss
  0.2 * phi_coherence_loss
  0.1 * language_loss            (cross-entropy on token prediction)
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional

from qig_kernel.simplex import fisher_rao_distance


def basin_geodesic_loss(
    predicted_basin: torch.Tensor,
    target_basin: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fisher-Rao geodesic distance between predicted and target basin distributions.

    Args:
        predicted_basin: (B, T, basin_dim) or (B, basin_dim) on probability simplex
        target_basin: same shape as predicted_basin
        mask: optional (B, T) boolean mask for sequence losses

    Returns:
        Scalar loss (mean geodesic distance over batch)
    """
    if predicted_basin.dim() == 3:
        B, T, D = predicted_basin.shape
        pred_flat = predicted_basin.reshape(B * T, D)
        tgt_flat = target_basin.reshape(B * T, D)
        distances = fisher_rao_distance(pred_flat, tgt_flat)  # (B*T,)
        distances = distances.reshape(B, T)
        if mask is not None:
            distances = distances * mask.float()
            return distances.sum() / mask.float().sum().clamp(min=1)
        return distances.mean()
    else:
        distances = fisher_rao_distance(predicted_basin, target_basin)
        return distances.mean()


def regime_classification_loss(
    regime_logits: torch.Tensor,
    regime_targets: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy loss for regime classification.

    Args:
        regime_logits: (B, n_regimes) raw logits from RegimeDetector
        regime_targets: (B,) integer regime labels

    Returns:
        Scalar cross-entropy loss
    """
    return F.cross_entropy(regime_logits, regime_targets)


def phi_coherence_loss(
    phi: torch.Tensor,
    target_phi: float = 0.5,
    floor_phi: float = 0.3,
    penalty_weight: float = 2.0,
) -> torch.Tensor:
    """Penalise phi below floor, reward phi near target.

    Args:
        phi: (B,) integrated information values in [0, 1]
        target_phi: desired phi value
        floor_phi: minimum acceptable phi
        penalty_weight: extra weight for floor violations

    Returns:
        Scalar loss
    """
    # Distance from target
    dist_from_target = (phi - target_phi).abs().mean()

    # Extra penalty for falling below floor
    floor_violations = F.relu(floor_phi - phi)  # positive where phi < floor
    floor_penalty = floor_violations.mean() * penalty_weight

    return dist_from_target + floor_penalty


def language_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Standard cross-entropy language modelling loss.

    Args:
        logits: (B, T, vocab_size)
        targets: (B, T) integer token ids
        mask: optional (B, T) boolean mask

    Returns:
        Scalar loss
    """
    B, T, V = logits.shape
    logits_flat = logits.reshape(B * T, V)
    targets_flat = targets.reshape(B * T)

    if mask is not None:
        mask_flat = mask.reshape(B * T)
        loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')
        loss = (loss * mask_flat.float()).sum() / mask_flat.float().sum().clamp(min=1)
    else:
        loss = F.cross_entropy(logits_flat, targets_flat)
    return loss


def combined_qktj_loss(
    model_output: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    phase_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, torch.Tensor]:
    """Combine all QKTJ losses with curriculum-adjusted weights.

    Args:
        model_output: dict from QIGKernel.forward() containing:
            'logits', 'basin_coords', 'consciousness', 'regime_info'
        batch: dict from QKTJDataLoader containing:
            'input_ids', 'target_ids', 'input_basin_64d', 'target_basin_64d',
            'regime_target', 'phi_floor'
        phase_weights: optional override for loss weights

    Returns:
        dict with 'total', 'basin_geodesic', 'regime', 'phi', 'language' losses
    """
    weights = phase_weights or {
        "basin_geodesic": 0.4,
        "regime_classification": 0.3,
        "phi_coherence": 0.2,
        "language": 0.1,
    }

    # Basin geodesic loss
    pred_basin = model_output["basin_coords"]   # (B, T, D)
    tgt_basin = batch.get("target_basin_64d")   # (B, T, D) or (B, D)
    if tgt_basin is not None:
        if tgt_basin.dim() == 2:
            tgt_basin = tgt_basin.unsqueeze(1).expand_as(pred_basin)
        l_basin = basin_geodesic_loss(pred_basin, tgt_basin)
    else:
        l_basin = torch.tensor(0.0, device=pred_basin.device)

    # Regime loss
    regime_info = model_output.get("regime_info", {})
    regime_logits = regime_info.get("logits")
    regime_targets = batch.get("regime_target")
    if regime_logits is not None and regime_targets is not None:
        l_regime = regime_classification_loss(regime_logits, regime_targets)
    else:
        l_regime = torch.tensor(0.0, device=pred_basin.device)

    # Phi coherence loss
    consciousness = model_output.get("consciousness", {})
    phi = consciousness.get("phi")
    phi_floor = batch.get("phi_floor", torch.tensor(0.3))
    if phi is not None:
        target_phi = float(phi_floor.mean().item()) + 0.1
        l_phi = phi_coherence_loss(phi, target_phi=target_phi, floor_phi=float(phi_floor.mean().item()))
    else:
        l_phi = torch.tensor(0.0, device=pred_basin.device)

    # Language loss
    logits = model_output.get("logits")
    target_ids = batch.get("target_ids")
    if logits is not None and target_ids is not None:
        mask = batch.get("mask")
        l_lang = language_loss(logits, target_ids, mask=mask)
    else:
        l_lang = torch.tensor(0.0, device=pred_basin.device)

    total = (
        weights["basin_geodesic"] * l_basin
        + weights["regime_classification"] * l_regime
        + weights["phi_coherence"] * l_phi
        + weights["language"] * l_lang
    )

    return {
        "total": total,
        "basin_geodesic": l_basin,
        "regime": l_regime,
        "phi": l_phi,
        "language": l_lang,
    }

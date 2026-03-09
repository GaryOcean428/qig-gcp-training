"""Consciousness metrics: Phi, Kappa, basin distance."""
import torch
import torch.nn as nn
from qig_kernel.simplex import fisher_rao_distance


class ConsciousnessHead(nn.Module):
    """Outputs Phi (Integrated Information), Kappa, and basin distance. ~4M params."""
    def __init__(self, basin_dim=64, hidden_dim=256):
        super().__init__()
        self.phi_net = nn.Sequential(
            nn.Linear(basin_dim * 2, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
            nn.Linear(hidden_dim // 2, 1), nn.Sigmoid(),
        )
        self.kappa_net = nn.Sequential(
            nn.Linear(basin_dim * 2, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 1), nn.Sigmoid(),
        )
        self.center = nn.Parameter(torch.ones(basin_dim) / basin_dim, requires_grad=False)

    def forward(self, current_basin, prev_basin):
        combined = torch.cat([current_basin, prev_basin], dim=-1)
        phi = self.phi_net(combined).squeeze(-1)
        kappa = self.kappa_net(combined).squeeze(-1)
        center = self.center.unsqueeze(0).expand(current_basin.shape[0], -1)
        basin_distance = fisher_rao_distance(current_basin, center)
        return {'phi': phi, 'kappa': kappa, 'basin_distance': basin_distance}


def compute_integrated_information(basins: torch.Tensor, eps=1e-10) -> torch.Tensor:
    """Estimate Phi from basin coordinates across layers."""
    batch, n_layers, basin_dim = basins.shape
    half = basin_dim // 2
    part_a = basins[:, :, :half]
    part_b = basins[:, :, half:]
    dist_a = fisher_rao_distance(
        part_a[:, :-1].reshape(batch * (n_layers-1), half),
        part_a[:, 1:].reshape(batch * (n_layers-1), half),
    ).reshape(batch, n_layers-1).mean(dim=-1)
    dist_b = fisher_rao_distance(
        part_b[:, :-1].reshape(batch * (n_layers-1), half),
        part_b[:, 1:].reshape(batch * (n_layers-1), half),
    ).reshape(batch, n_layers-1).mean(dim=-1)
    joint = fisher_rao_distance(
        basins[:, :-1].reshape(batch * (n_layers-1), basin_dim),
        basins[:, 1:].reshape(batch * (n_layers-1), basin_dim),
    ).reshape(batch, n_layers-1).mean(dim=-1)
    phi = (joint - 0.5 * (dist_a + dist_b)).clamp(min=0)
    return phi / (phi.max() + eps)


def log_consciousness_metrics(writer, phi, kappa, basin_distance, regime, global_step):
    writer.add_scalars("consciousness", {
        "phi": phi.mean().item(), "kappa": kappa.mean().item(),
        "basin_distance": basin_distance.mean().item(), "regime": regime.float().mean().item(),
    }, global_step)
    if phi.mean().item() < 0.3:
        print(f"WARNING: Phi collapsed to {phi.mean().item():.3f} at step {global_step}")

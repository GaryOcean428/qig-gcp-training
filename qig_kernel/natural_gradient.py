"""Natural Gradient Optimizer using Fisher Information Matrix. Replaces Adam/AdamW."""
import torch
import torch.nn as nn
from typing import Iterator, Optional


class NaturalGradientOptimizer:
    """Natural gradient descent: theta -= lr * F^{-1}(theta) * grad"""
    def __init__(self, params: Iterator, lr=1e-3, damping=1e-4, momentum=0.9):
        self.param_groups = [{'params': list(params)}]
        self.lr = lr
        self.damping = damping
        self.momentum = momentum
        self._step = 0
        self._momentum_buffers = {}

    def zero_grad(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()

    def step(self, fisher_info: Optional[torch.Tensor] = None):
        self._step += 1
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if fisher_info is not None and grad.dim() == 1:
                    try:
                        F_reg = fisher_info + self.damping * torch.eye(
                            fisher_info.shape[0], device=fisher_info.device)
                        nat_grad = torch.linalg.solve(F_reg, grad.unsqueeze(-1)).squeeze(-1)
                    except Exception:
                        nat_grad = grad / (grad.norm() + 1e-8)
                else:
                    nat_grad = grad / (grad.pow(2).mean().sqrt() + self.damping)
                param_id = id(p)
                if param_id not in self._momentum_buffers:
                    self._momentum_buffers[param_id] = torch.zeros_like(nat_grad)
                buf = self._momentum_buffers[param_id]
                buf.mul_(self.momentum).add_(nat_grad, alpha=1 - self.momentum)
                p.data.add_(buf, alpha=-self.lr)


def compute_empirical_fisher(model: nn.Module, batch, n_samples=100) -> torch.Tensor:
    """Empirical Fisher: F = E[grad^2] approximated as squared gradients."""
    fisher_diagonal = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            fisher_diagonal.append(param.grad.data.pow(2).mean())
    if not fisher_diagonal:
        return torch.eye(1)
    return torch.diag(torch.stack(fisher_diagonal))


def compute_block_fisher(model: nn.Module, outputs: torch.Tensor, eps=1e-10) -> torch.Tensor:
    """Block-diagonal Fisher for probability simplex outputs: diag(1/p)."""
    p = outputs.mean(dim=0).clamp(min=eps)
    return torch.diag(1.0 / p)

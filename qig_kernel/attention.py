"""
QFI-Metric Attention: Fisher-Rao distance kernel (NOT dot-product).
"""
import torch
import torch.nn as nn
from qig_kernel.simplex import fisher_rao_distance, project_to_simplex, frechet_normalize


def fisher_rao_attention(p, q, temperature=0.5, eps=1e-10):
    """Fisher-Rao attention: exp(-d_FR(p,q)/T). Replaces dot(Q,K)/sqrt(d)."""
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    sqrt_inner = torch.sum(torch.sqrt(p) * torch.sqrt(q), dim=-1)
    d_fr = 2.0 * torch.acos(sqrt_inner.clamp(-1.0 + eps, 1.0 - eps))
    return torch.exp(-d_fr / temperature)


class QFIAttentionHead(nn.Module):
    """Single QFI attention head operating on probability simplex subspace."""
    def __init__(self, basin_dim=64, head_dim=8, temperature=0.5):
        super().__init__()
        self.basin_dim = basin_dim
        self.head_dim = head_dim
        self.temperature = temperature
        self.query_proj = nn.Linear(basin_dim, head_dim, bias=False)
        self.key_proj = nn.Linear(basin_dim, head_dim, bias=False)
        self.value_proj = nn.Linear(basin_dim, head_dim, bias=False)

    def forward(self, x):
        q = project_to_simplex(self.query_proj(x))
        k = project_to_simplex(self.key_proj(x))
        v = project_to_simplex(self.value_proj(x))
        batch, seq, _ = q.shape
        q_exp = q.unsqueeze(2).expand(-1, -1, seq, -1)
        k_exp = k.unsqueeze(1).expand(-1, seq, -1, -1)
        weights = fisher_rao_attention(q_exp, k_exp, self.temperature)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-10)
        sqrt_v = torch.sqrt(v.clamp(min=1e-10))
        attended_sqrt = torch.bmm(weights, sqrt_v.view(batch, seq, -1)).view(batch, seq, -1)
        attended_sqrt = attended_sqrt / (attended_sqrt.norm(dim=-1, keepdim=True) + 1e-10)
        attended = attended_sqrt ** 2
        return attended / (attended.sum(dim=-1, keepdim=True) + 1e-10)


class QFIMultiHeadAttention(nn.Module):
    """Multi-head QFI attention. Each head operates on a simplex subspace."""
    def __init__(self, basin_dim=64, n_heads=8, temperature=0.5):
        super().__init__()
        assert basin_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = basin_dim // n_heads
        self.heads = nn.ModuleList([
            QFIAttentionHead(basin_dim, self.head_dim, temperature)
            for _ in range(n_heads)
        ])
        self.out_proj = nn.Linear(basin_dim, basin_dim, bias=False)

    def forward(self, x):
        head_outputs = [head(x) for head in self.heads]
        concat = torch.cat(head_outputs, dim=-1)
        out = self.out_proj(concat)
        return project_to_simplex(out)

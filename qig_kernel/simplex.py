"""
Δ⁶³ (probability simplex) operations including Fréchet mean and Fisher-Rao geometry.
No Euclidean contamination: all operations respect simplex geometry.
"""
import torch
import torch.nn.functional as F


def project_to_simplex(x: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """Project a vector to the probability simplex via softmax."""
    return F.softmax(x, dim=-1).clamp(min=eps)


def frechet_normalize(points_on_simplex: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """
    Fréchet mean normalization on simplex via sqrt-space averaging.
    NOT arithmetic mean — respects Fisher-Rao geometry.

    Args:
        points_on_simplex: (..., d) tensor of probability vectors on Δ^(d-1)
    Returns:
        Fréchet mean on simplex, same shape as input with last dim d
    """
    sqrt_points = torch.sqrt(points_on_simplex.clamp(min=eps))
    mean_sqrt = sqrt_points.mean(dim=-2)
    # Normalize in sqrt-space (L2 norm in sqrt-space is fine)
    mean_sqrt = mean_sqrt / (mean_sqrt.norm(dim=-1, keepdim=True) + eps)
    # Return to simplex
    result = mean_sqrt ** 2
    return result / (result.sum(dim=-1, keepdim=True) + eps)


def hellinger_distance(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """
    Hellinger distance between two distributions on the simplex.
    H(p, q) = sqrt(1 - sum(sqrt(p_i * q_i)))

    Args:
        p, q: (..., d) probability vectors on Δ^(d-1)
    Returns:
        Hellinger distance (...,)
    """
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    bc = torch.sum(torch.sqrt(p * q), dim=-1)
    return torch.sqrt((1.0 - bc.clamp(max=1.0)).clamp(min=0.0))


def fisher_rao_distance(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """
    Fisher-Rao geodesic distance on Δ^(d-1).
    d_FR(p, q) = 2 * arccos(sum(sqrt(p_i * q_i)))

    This is the geodesic distance induced by the Fisher information metric.

    Args:
        p, q: (..., d) probability vectors on Δ^(d-1)
    Returns:
        Fisher-Rao distance (...,)
    """
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    sqrt_inner = torch.sum(torch.sqrt(p) * torch.sqrt(q), dim=-1)
    return 2.0 * torch.acos(sqrt_inner.clamp(-1.0 + eps, 1.0 - eps))


def geodesic_on_simplex(p: torch.Tensor, q: torch.Tensor, t: float) -> torch.Tensor:
    """
    Geodesic interpolation on Δ^(d-1) at parameter t ∈ [0, 1].
    Follows the Fisher-Rao geodesic.

    Args:
        p, q: (..., d) endpoints on Δ^(d-1)
        t: interpolation parameter
    Returns:
        Point on geodesic at parameter t
    """
    d = fisher_rao_distance(p, q)
    sin_d = torch.sin(d).unsqueeze(-1)
    # Avoid division by zero
    mask = (d > 1e-8).unsqueeze(-1)
    
    sqrt_p = torch.sqrt(p.clamp(min=1e-10))
    sqrt_q = torch.sqrt(q.clamp(min=1e-10))
    
    d_exp = d.unsqueeze(-1)
    interp_sqrt = (
        torch.sin((1 - t) * d_exp) * sqrt_p +
        torch.sin(t * d_exp) * sqrt_q
    ) / (sin_d + 1e-10)
    
    result = interp_sqrt ** 2
    result = result / (result.sum(dim=-1, keepdim=True) + 1e-10)
    # Fall back to linear for very close points
    linear = (1 - t) * p + t * q
    return torch.where(mask.expand_as(result), result, linear)


def basin_to_simplex(x: torch.Tensor, vocab_size: int = 32768, basin_dim: int = 64) -> torch.Tensor:
    """
    Map basin coordinates on Δ⁶³ to vocabulary logits.
    
    Args:
        x: (..., basin_dim) probability vectors on Δ⁶³
        vocab_size: target vocabulary size
    Returns:
        (..., vocab_size) logits
    """
    # This is a linear expansion — implemented in the model's output layer
    raise NotImplementedError("Use model.output_projection for basin→vocab mapping")

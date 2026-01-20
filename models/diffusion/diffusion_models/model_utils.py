import torch
import torch.nn as nn

from .mlp import *
from ..p_path import Alpha, Beta

class ScoreFromVectorField(nn.Module):
    """
    Parameterization of score via learned vector field (for the special case of a Gaussian conditional probability path)
    """
    def __init__(self, vector_field: MLPVectorField, alpha: Alpha, beta: Beta):
        super().__init__()
        self.vector_field = vector_field
        self.alpha = alpha
        self.beta = beta

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        Args:
        - x: (bs, dim)
        Returns:
        - score: (bs, dim)
        """
        vf_t = self.vector_field(x, t)
        alpha_t = self.alpha(t)
        beta_t = self.beta(t)
        dt_alpha_t = self.alpha.dt(t)
        dt_beta_t = self.beta.dt(t)

        score = (alpha_t * vf_t - dt_alpha_t * x) / (beta_t**2 * dt_alpha_t - alpha_t * dt_beta_t * beta_t)
        return score
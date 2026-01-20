import torch
import torch.nn as nn

from .SDE import ODE, SDE
from .p_path import ConditionalProbabilityPath


"""
This code is from the 'MIT Computer Science Class 6.S184: Generative AI with Stochastic Differential Equations' LAB
You can check the details in https://diffusion.csail.mit.edu/2025/index.html
"""


class ConditionalVectorFieldSDE(SDE):
    def __init__(self, path: ConditionalProbabilityPath, z: torch.Tensor, sigma: float):
        """
        Args:
            - path: the ConditionalProbabilityPath object to which this vector field corresponds
            - z: the corresponding variable. Shape - (1, ...)
        """
        super().__init__()
        self.path = path
        self.z = z
        self.sigma = sigma

    def drift_coefficient(self, x: torch.Tensor, t: torch.Tensor):
        """
        Returns the conditional vector field u_t(x|z)
        Args:
            - x: state at time t. Shape - (batch_size, dim)
            - t: time. Shape - (batch_size,)
        Returns;
            - u_t(x|z): Shape - (batch_size, dim)
        """
        bs = x.shape[0]
        z = self.z.expand(bs, *self.z.shape[1:])
        return self.path.conditional_vector_field(x, z, t) +(0.5 * self.sigma**2 * self.path.conditional_score(x, z, t))
    
    def diffusion_coefficient(self, x: torch.Tensor, t: torch.Tensor):
        return self.sigma * torch.randn_like(x)
    

class LearnedVectorField(SDE):
    """
    SDE equals to ODe when diffusion_coefficient = 0 (when sigma = 0)
    """
    def __init__(self, net: nn.Module, sigma: float = 0):
        super().__init__()
        self.net = net
        self.sigma = sigma

    def drift_coefficient(self, x: torch.Tensor, t: torch.Tensor):
        return self.net(x, t)
    
    def diffusion_coefficient(self, x: torch.Tensor, t: torch.Tensor):
        return self.sigma * torch.randn_like(x)

       
        
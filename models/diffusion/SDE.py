import torch

from .density_utils import Density
from abc import ABC, abstractmethod


"""
This code is from the 'MIT Computer Science Class 6.S184: Generative AI with Stochastic Differential Equations' LAB
You can check the details in https://diffusion.csail.mit.edu/2025/index.html
"""

class ODE(ABC):
    @abstractmethod
    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor):
        """
        Returns the drift coefficients of the ODE
        Args:
            - xt: state at time t. Shape - (batch_size, dim)
            - t: time. Shape - (batch_size, 1)
        Returns:
            - drift_coefficient. Shape - (batch_size, dim)
        """
        pass


class SDE(ODE): # ODE is a special case of SED
    def __init__(self):
        super().__init__()

    @abstractmethod
    def diffusion_coefficient(self, xt: torch.Tensor, t: torch.Tensor):
        """
        Returns the diffusion coefficients of the SDE
        Args:
            - xt: state at time t. Shape - (batch_size, dim)
            - t: time. Shape - (batch_size, 1)
        Returns:
            - diffusion_coefficient. Shape - (batch_size, dim)
        """
        pass


class BrownianMotion(SDE):
    def __init__(self, sigma: float):
        self.sigma = sigma

    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor):
        # Brownian Motion's vector field = 0.
        return torch.zeros_like(xt)
    
    def diffusion_coefficient(self,xt: torch.Tensor, t: torch.Tensor):
        return self.sigma * torch.ones_like(xt)
    

class OUProcess(SDE):
    """
    Ornstein-Uhlenbeck Process
    It's vector field is defined as u_{t}(x) = -theta * x
    """
    def __init__(self, theta: float, sigma: float):
        self.theta = theta
        self.sigma = sigma

    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor):
        return -self.theta * xt

    def diffusion_coefficient(self, xt: torch.Tensor, t: torch.Tensor):
        return self.sigma * torch.one_like(xt)
    

class LangevinSDE(SDE):
    """
    Simulating the Langevin Dynamics. Its SDE is only composed of a score function.
    """
    def __init__(self, sigma: float, density: Density):
        self.sigma = sigma
        self.density = density

    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor):
        return 0.5 * (self.sigma**2) * self.density.score(xt)
    
    def diffusion_coefficient(self, xt: torch.Tensor, t: torch.Tensor):
        return self.sigma * torch.ones_like(xt)
    

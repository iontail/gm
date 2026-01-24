import torch

from abc import ABC, abstractmethod

from .density_utils import Density
from .diffusion_models import *


"""
This code is from the 'MIT Computer Science Class 6.S184: Generative AI with Stochastic Differential Equations' LAB
You can check the details in https://diffusion.csail.mit.edu/2025/index.html
"""

class ODE(ABC):
    @abstractmethod
    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor, **kwargs):
        """
        Returns the drift coefficients of the ODE
        Args:
            - xt: state at time t. Shape - (batch_size, c, h, w)
            - t: time. Shape - (batch_size, 1, 1, 1)
        Returns:
            - drift_coefficient. Shape - (batch_size, c, h, w)
        """
        pass


class SDE(ODE): # ODE is a special case of SED
    def __init__(self):
        super().__init__()

    @abstractmethod
    def diffusion_coefficient(self, xt: torch.Tensor, t: torch.Tensor, **kwargs):
        """
        Returns the diffusion coefficients of the SDE
        Args:
            - xt: state at time t. Shape - (batch_size, c, h, w)
            - t: time. Shape - (batch_size, 1, 1, 1)
        Returns:
            - diffusion_coefficient. Shape - (batch_size, c, h, w)
        """
        pass


class BrownianMotion(SDE):
    def __init__(self, sigma: float):
        self.sigma = sigma

    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor, **kwargs):
        # Brownian Motion's vector field = 0.
        return torch.zeros_like(xt)
    
    def diffusion_coefficient(self,xt: torch.Tensor, t: torch.Tensor, **kwargs):
        return self.sigma * torch.ones_like(xt)
    

class OUProcess(SDE):
    """
    Ornstein-Uhlenbeck Process
    It's vector field is defined as u_{t}(x) = -theta * x
    """
    def __init__(self, theta: float, sigma: float):
        self.theta = theta
        self.sigma = sigma

    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor, **kwargs):
        return -self.theta * xt

    def diffusion_coefficient(self, xt: torch.Tensor, t: torch.Tensor, **kwargs):
        return self.sigma * torch.one_like(xt)
    

class LangevinSDE(SDE):
    """
    Simulating the Langevin Dynamics. Its SDE is only composed of a score function.
    """
    def __init__(self, sigma: float, density: Density):
        self.sigma = sigma
        self.density = density

    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor, **kwargs):
        return 0.5 * (self.sigma**2) * self.density.score(xt)
    
    def diffusion_coefficient(self, xt: torch.Tensor, t: torch.Tensor, **kwargs):
        return self.sigma * torch.ones_like(xt)
    

# ---------------
# Trained
# ---------------
class LearnedBVectorFieldODE(ODE):
    def __init__(self, net: MLPVectorField):
        self.net = net

    def drift_coefficient(self, x: torch.Tensor, t: torch.Tensor, **kwargs):
        """
        Args:
            - x: (bs, dim)
            - t: (bs, dim)
        Returns:
            - u_t: (bs, dim)
        """
        return self.net(x, t)
    

class LangevinFlowSDE(SDE):
    def __init__(self, flow_model: MLPVectorField, score_model: MLPScore, sigma: float):
        """
        Args:
        - path: the ConditionalProbabilityPath object to which this vector field corresponds
        - z: the conditioning variable, (1, dim)
        """
        super().__init__()
        self.flow_model = flow_model
        self.score_model = score_model
        self.sigma = sigma

    def drift_coefficient(self, x: torch.Tensor, t: torch.Tensor, **kwargs):
        """
        Args:
            - x: state at time t, shape (batch_size, dim)
            - t: time, shape (bs,.)
        Returns:
            - u_t(x|z): shape (batch_size, dim)
        """
        return self.flow_model(x,t) + 0.5 * self.sigma ** 2 * self.score_model(x, t)

    def diffusion_coefficient(self, x: torch.Tensor, t: torch.Tensor, **kwargs):
        """
        Args:
            - x: state at time t, shape (batch_size, dim)
            - t: time, shape (batch_size,.)
        Returns:
            - u_t(x|z): shape (batch_size, dim)
        """
        return self.sigma * torch.randn_like(x)


class ConditionalVectorField(nn.Module, ABC):
    """
    MLP-parameterization of the learned vector field u_t^theta(x)
    """
    @abstractmethod
    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
        """
        Args:
        - x: (bs, c, h, w)
        - t: (bs, 1, 1, 1)
        - y: (bs,)
        Returns:
        - u_t^theta(x|y): (bs, c, h, w)
        """
        pass

class CFGVectorField(ODE):
    def __init__(self, net: ConditionalVectorField, guidance_scale: float = 1.0):
        self.net = net
        self.guidance_scale = guidance_scale

    def drift_coefficient(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
        guided_vector_field = self.net(x, t, y)
        unguided_y = torch.ones_like(y) * 10
        unguided_vector_field = self.net(x, t, unguided_y)
        return (1 - self.guidance_scale) * unguided_vector_field + self.guidance_scale * guided_vector_field
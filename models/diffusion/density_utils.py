import torch
import torch.nn as nn
import torch.distributions as D
import numpy as np

from torch.func import vmap, jacrev
from abc import ABC, abstractmethod
from typing import Tuple, Optional, List

"""
This code is from the 'MIT Computer Science Class 6.S184: Generative AI with Stochastic Differential Equations' LAB
You can check the details in https://diffusion.csail.mit.edu/2025/index.html
"""


class Density(ABC):
    """
    Distribution with tractable density
    """
    @abstractmethod
    def log_density(self, x: torch.Tensor):
        """
        Returns the log density at x.
        Args:
            - x: Shape - (batch_size, dim)
        Returns:
            - log_density: Shape - (batch_size, 1)
        """
        pass

    def score(self, x: torch.Tensor):
        """
        Returns the score dx log density(x)
        Args:
            - x: (batch_size, dim)
        Returns:
            - score: (batch_size, dim)
        """
        x = x.unsqueeze(1) # (B, 1, C)
        score = vmap(jacrev(self.log_density))(x) # (B, 1, 1, 1, ...)
        return score.squeeze((1, 2, 3))
    

class Sampleable(ABC):
    """
    Distribution which can be sampled from
    """
    @abstractmethod
    def sample(self, num_samples: int) -> Tuple[torch.Tensor, Optional[[torch.Tensor]]]:
        """
        Returns the log density at x.
        Args:
            - num_samples: the desired number of samples
        Returns:
            - samples: Shape - (batch_size, ...)
            - labels: Shape - (batch_size, label_dim)
        """
        pass

class IsotropicGaussian(nn.Module, Sampleable):
    def __init__(self, shape: List[int], std: float = 1.0):
        super().__init__()

        self.shape = shape
        self.std = std
        self.dummy = nn.Buffer(torch.zeros(1))

    def sample(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.std * torch.randn(num_samples, *self.shape).to(self.dummy.device), None


class Gaussian(nn.Module, Sampleable, Density):
    """
    Multivariate Gaussian Distribution
    """
    def __init__(self, mean, cov):
        """
        mean: Shape - (dim,)
        cov: Shape - (dim, dim)
        """
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("cov", cov)

    @property
    def dim(self):
        return self.mean.shape[0]
    
    @property
    def distribution(self):
        return D.MultivariateNormal(self.mean, self.cov, validate_args=False)
    
    def sample(self, num_samples: int):
        return self.distribution.sample((num_samples, ))
    
    def log_density(self, x: torch.Tensor):
        return self.distribution.log_prob(x).view(-1, 1)
    
    @classmethod
    def isotropic(cls, dim: int, std: float):
        mean = torch.zeroz(dim)
        cov = torch.eye(dim) * std**2
        return cls(mean, cov)

class GaussianMixture(torch.nn.Module, Sampleable, Density):
    """
    Two-dimensional Gaussian mixture model, and is a Density and a Sampleable. Wrapper around torch.distributions.MixtureSameFamily.
    """
    def __init__(
        self,
        means: torch.Tensor,  # nmodes x data_dim
        covs: torch.Tensor,  # nmodes x data_dim x data_dim
        weights: torch.Tensor,  # nmodes
    ):
        """
        means: Shape - (nmodes, 2)
        covs: Shape - (nmodes, 2, 2)
        weights: Shape - (nmodes, 1)
        """
        super().__init__()
        self.nmodes = means.shape[0]
        self.register_buffer("means", means)
        self.register_buffer("covs", covs)
        self.register_buffer("weights", weights)

    @property
    def dim(self) -> int:
        return self.means.shape[1]

    @property
    def distribution(self):
        return D.MixtureSameFamily(
                mixture_distribution=D.Categorical(probs=self.weights, validate_args=False),
                component_distribution=D.MultivariateNormal(
                    loc=self.means,
                    covariance_matrix=self.covs,
                    validate_args=False,
                ),
                validate_args=False,
            )

    def log_density(self, x: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(x).view(-1, 1)

    def sample(self, num_samples: int) -> torch.Tensor:
        return self.distribution.sample(torch.Size((num_samples,)))

    @classmethod
    def random_2D(
        cls, nmodes: int, std: float, scale: float = 10.0, seed = 0.0
    ) -> "GaussianMixture":
        torch.manual_seed(seed)
        means = (torch.rand(nmodes, 2) - 0.5) * scale
        covs = torch.diag_embed(torch.ones(nmodes, 2)) * std ** 2
        weights = torch.ones(nmodes)
        return cls(means, covs, weights)

    @classmethod
    def symmetric_2D(
        cls, nmodes: int, std: float, scale: float = 10.0,
    ) -> "GaussianMixture":
        angles = torch.linspace(0, 2 * np.pi, nmodes + 1)[:nmodes]
        means = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1) * scale
        covs = torch.diag_embed(torch.ones(nmodes, 2) * std ** 2)
        weights = torch.ones(nmodes) / nmodes
        return cls(means, covs, weights)



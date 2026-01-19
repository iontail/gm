import torch

from abc import ABC, abstractmethod


"""
This code is from the 'MIT Computer Science Class 6.S184: Generative AI with Stochastic Differential Equations' lab
You can check the details in https://diffusion.csail.mit.edu/2025/index.html
"""

class ODE(ABC):
    @abstractmethod
    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor):
        """
        Returns the drift coefficients of the ODE
        Args:
            - xt: state at time t. Shape - (batch_size, dim)
            - t: time. Shape - ()
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
            - t: time. Shape - ()
        Returns:
            - diffusion_coefficient. Shape - (batch_size, dim)
        """
        pass


class Simulator(ABC):
    @abstractmethod
    def step(self, xt: torch.Tensor, t: torch.Tensor, dt: torch.Tensor):
        """
        Takes one simulation step.
        Subtext:
            - As we cannot compute the flow explitcitly when the vector field is complex (e.g. MLPs),
              We simulate the ODE(or SDE) with numerical methods such as Euler methods.

        Args:
            - xt: state at time t. Shape - (batch_size, dim)
            - t: time. Shape - ()
            - dt: tine. Shape - ()
        Returns:
            - nxt: state at time (t + dt)
        """
        pass

    @torch.no_grad()
    def simulate(self, x: torch.Tensor, ts: torch.Tensor):
        """
        Simulates using the discretization given by ts
        Args:
            - x: initial state at time ts[0]. Shape - (batch_size, dim)
            - ts: timesteps. Shape - (num_timesteps, )
        Returns:
            - x_final: final state at time ts[-1]. Shape - (batch_size, dim)
        """
        for t_idx in range(len(ts) - 1):
            t = ts[t_idx]
            h = ts[t_idx + 1] - ts[t_idx]
            x = self.step(x, t, h)
        return x
    
    @torch.no_grad()
    def simulate_with_trajectory(self, x: torch.Tensor, ts: torch.Tensor):
        """
        Simulates using the discretization given by ts
        Returns the list of the position to the times step 'ts' when simulating
        Args:
            - x: initial state at time ts[0]. Shape - (batch_size, dim)
            - ts: timesteps. Shape - (num_timesteps, )
        Returns:
            - xs: trajectory of x_ts over ts. Shape - (batch_size, num_timesteps, dim)
        """

        xs = [x.clone()]
        for t_idx in range(len(ts) - 1):
            t = ts[t_idx]
            h = ts[t_idx + 1] - ts[t_idx]
            x = self.step(x, t, h)
            xs.append(x.clone())
        return torch.stack(xs, dim=1)
    

class EulerSimulator(Simulator):
    def __init__(self, ode: ODE):
        self.ode = ode
    
    def step(self, xt: torch.Tensor, t: torch.Tensor, h: torch.Tensor):
        # h: shape - ()
        return self.ode.drift_coefficient(xt, t) * h
    
class EulerMaruyamaSimulator(Simulator):
    def __init__(self, sde: SDE):
        self.sde = sde

    def step(self, xt: torch.Tensor, t: torch.Tensor, h: torch.Tensor):
        """
        dW_{t} must meet the two conditions
        1. Normal Increments
        2. Indepedent Increments

        To meet the conditions above, I used 'toch.rand_like()' method.
        See detils in page 9 of https://arxiv.org/abs/2506.02070
        """
        return self.sde.drift_coefficient(xt, t) * h + torch.sqrt(h) * self.sde.diffusion_coefficient * torch.rand_like(xt)
    

class BrownianMotion(SDE):
    def __init__(self, sigma: float):
        self.sigma = sigma

    def drift_coefficient(self, xt, t):
        # Brownian Motion has no vector field.
        return torch.zeros_like(xt)
    
    def diffusion_coefficient(self, xt, t):
        return self.sigma * torch.ones_like(xt)
    

class OUProcess(SDE):
    """
    Ornstein-Uhlenbeck Process
    It's vector field is defined as u_{t}(x) = -theta * x
    """
    def __init__(self, theta: float, sigma: float):
        self.theta = theta
        self.sigma = sigma

    def drift_coefficient(self, xt, t):
        return -self.theta * torch.ones_like(xt)

    def diffusion_coefficient(self, xt, t):
        return self.sigma * torch.one_like(xt)
    





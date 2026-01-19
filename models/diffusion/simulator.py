import torch

from abc import ABC, abstractmethod

from .SDE import ODE, SDE


"""
This code is from the 'MIT Computer Science Class 6.S184: Generative AI with Stochastic Differential Equations' LAB
You can check the details in https://diffusion.csail.mit.edu/2025/index.html
"""

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
            - t: time. Shape - (batch_size, 1)
            - dt: tine. Shape - (batch_size, 1)
        Returns:
            - nxt: state at time (t + dt). Shape - (batch_size, dim)
        """
        pass

    @torch.no_grad()
    def simulate(self, x: torch.Tensor, ts: torch.Tensor):
        """
        Simulates using the discretization gives by ts
        Args:
            - x: initial state at time ts[0]. Shape - (batch_size, dim)
            - ts: timesteps. Shape - (batch_size, num_timesteps, 1)
        Returns:
            - x_final: final state at time ts[-1]. Shape - (batch_size, dim)
        """
        for t_idx in range(len(ts) - 1):
            t = ts[:, t_idx]
            h = ts[:, t_idx + 1] - ts[:, t_idx]
            x = self.step(x, t, h)
        return x
    
    @torch.no_grad()
    def simulate_with_trajectory(self, x: torch.Tensor, ts: torch.Tensor):
        """
        Simulates using the discretization gives by ts
        Returns the list of the position to the times step 'ts' when simulating
        Args:
            - x: initial state at time ts[0]. Shape - (batch_size, dim)
            - ts: timesteps. Shape - (batch_size, num_timesteps, 1)
        Returns:
            - xs: trajectory of x_ts over ts. Shape - (batch_size, num_timesteps, dim)
        """

        xs = [x.clone()]
        for t_idx in range(len(ts) - 1):
            t = ts[:, t_idx]
            h = ts[:, t_idx + 1] - ts[:, t_idx]
            x = self.step(x, t, h)
            xs.append(x.clone())
        return torch.stack(xs, dim=1)
    

class EulerSimulator(Simulator):
    def __init__(self, ode: ODE):
        self.ode = ode
    
    def step(self, xt: torch.Tensor, t: torch.Tensor, h: torch.Tensor):
        # h: shape - ()
        return xt + self.ode.drift_coefficient(xt, t) * h
    
class EulerMaruyamaSimulator(Simulator):
    def __init__(self, sde: SDE):
        self.sde = sde

    def step(self, xt: torch.Tensor, t: torch.Tensor, h: torch.Tensor):
        """
        dW_{t} must meet the two conditions
        1. Normal Increments
        2. Indepedent Increments

        To meet the conditions above, I used 'toch.randn_like()' method.
        It must follow normal distribution because each W is defined under Normal distribution.
        (so I did not use 'torch.rand_like()' method.)
        
        See detils in page 9 of https://arxiv.org/abs/2506.02070
        """
        return xt + (self.sde.drift_coefficient(xt, t) * h) + (torch.sqrt(h) * self.sde.diffusion_coefficient * torch.randn_like(xt))
    








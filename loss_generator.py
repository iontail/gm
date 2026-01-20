import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import VAE
from .models.diffusion import ConditionalProbabilityPath, Alpha, Beta, LinearAlpha, SquareRootBeta

"""
The target model should be assigned to self.model!!

This limitation is for the alignment with Trainer class
"""

class VAELossGenerator(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.model = VAE(args)

        self.recon_loss_fn = nn.MSELoss(reduction='sum')
        self.kld_loss_fn = nn.KLDivLoss(reduction='sum')
        
    def forward(self, x: torch.Tensor):
        """
        Return reconstruction loss and KL divergence loss
        Args:
            x (torch.Tensor): a minibatch of dataset 
            - batch size should be greater or equal to 100 for Monte Carlo estimation with single iteration

        Returns:
            recon_loss (torch.Tensor): reconstruction loss
            kld_loss (torch.Tensor): KL divergence loss
        """
        pass



# -----------------
# ODEs/SDEs
# -----------------

class RepresentationTransformer(nn.Module):
    def __init__(self,
                 vf_predictor: nn.Module = None,
                 score_predictor: nn.Module = None,
                 noise_predictor: nn.Module = None,
                 target_domain: str = 'score',
                 alpha: Alpha = LinearAlpha(),
                 beta: Beta = SquareRootBeta()
                 ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mode = None
        self.transform = None

        if vf_predictor is not None and (score_predictor is None and noise_predictor is None):
            self.model = vf_predictor
            self.mode = 'vf'
        elif score_predictor is not None and (vf_predictor is None and noise_predictor is None):
            self.model = score_predictor
            self.mode = 'score'
        elif noise_predictor is not None and (vf_predictor is None and score_predictor is None):
            self.model = noise_predictor
            self.mode = 'noise'
        else:
            raise ValueError("Exactly one of vf_predictor, score_predictor, or noise_predictor must be provided (non-None).")
        
        if target_domain not in {'vf', 'score', 'noise'}:
            raise ValueError("target_domain must be one of {'vf', 'score', 'noise'}.")
        
        if target_domain == self.mode:
            raise ValueError("target_domain equals to the model domain.")
        
        transform_name = f"{self.mode}2{target_domain}"
        self.transform = getattr(type(self), transform_name)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        if self.transform is not None:
            return self.transform(x, t)
        raise ValueError("self.transform is not callable (None).")

    def vf2score(self, x: torch.Tensor, t: torch.Tensor):
        """
        Transforms vector field to Score
        Args:
            - x: Shape - (batch_size, dim)
        Returns:
            - score: Shape - (batch_size, dim)
        """
        vf_t = self.model(x, t)
        alpha_t = self.alpha(t)
        beta_t = self.beta(t)
        dt_alpha_t = self.alpha.dt(t)
        dt_beta_t = self.beta.dt(t)

        score = (alpha_t * vf_t - dt_alpha_t * x) / (beta_t**2 * dt_alpha_t - alpha_t * dt_beta_t * beta_t + 1e-7)
        return score

    def score2noise(self, x: torch.Tensor, t: torch.Tensor):
        score_t = self.model(x, t)
        beta_t = self.beta(t)
        return -beta_t * score_t

    def noise2score(self, x: torch.Tensor, t: torch.Tensor):
        noise_t = self.model(x, t)
        beta_t = self.beta(t)
        return -noise_t / (beta_t + 1e-7)

    def score2vf(self, x: torch.Tensor, t: torch.Tensor):
        score_t = self.model(x, t)
        alpha_t = self.alpha(t)
        beta_t = self.beta(t)
        dt_alpha_t = self.alpha.dt(t)
        dt_beta_t = self.beta.dt(t)
        
        vf = (dt_alpha_t - dt_beta_t / (beta_t * alpha_t + 1e-7)) * score_t + dt_alpha_t / alpha_t * x
        return vf

    def vf2noise(self, x: torch.Tensor, t: torch.Tensor):
        return self.score2noise(self.vf2score(x, t), t)

    def noise2vf(self, x: torch.Tensor, t: torch.Tensor):
        return self.score2vf(self.noise2score(x, t), t)

        
class ConditionalFlowMatchingLossGenerator:
    """
    Generate the flow matching loss:
    E_{z ~ p_data, t ~ Unif[0, 1], x ~ p_t(x|z)}[||u_theta - u_target||^2]

    Before defining Loss Generator, you should define the Sampler class for the data to use
    The newly defined Sampler class is used as p_data in Probability Path calss.
    """
    def __init__(self, path: ConditionalProbabilityPath, model: nn.Module, **kwargs):
        super().__init__()
        self.path = path
        self.model = model

    def forward(self, batch_size: int):
        z = self.path.p_data.sample(batch_size) # (bs, dim)
        t = torch.rand(batch_size, 1).to(z)     # (bs, 1)
        x = self.path.sample_conditional_path(x, z, t)

        ut_theta = self.model(x, t)
        ut_target = self.path.conditional_vector_field(x, z, t)
        mse = F.mse_loss(ut_theta, ut_target)
        return mse


class ConditionalScoreMatchingLossGenerator:
    """
    Generate the score matching loss:
    E_{z ~ p_data, t ~ Unif[0, 1], x ~ p_t(x|z)}[||s_theta - s_target||^2]

    Before defining Loss Generator, you should define the Sampler class for the data to use
    The newly defined Sampler class is used as p_data in Probability Path calss.
    """
    def __init__(self, path: ConditionalProbabilityPath, model: nn.Module, **kwargs):
        super().__init__()
        self.path = path
        self.model = model

    def forward(self, batch_size: int):
        z = self.path.p_data.sample(batch_size) # (bs, dim)
        t = torch.rand(batch_size, 1).to(z)     # (bs, 1)
        x = self.path.sample_conditional_path(x, z, t)

        s_theta = self.model(x, t)
        s_target = self.path.conditional_score(x, z, t)
        mse = F.mse_loss(s_theta, s_target)
        return mse

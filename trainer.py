import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import os

from typing import List
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
from tqdm import tqdm
from datetime import datetime

from utils import Utils
from .models import VAE
from .models.diffusion import ConditionalProbabilityPath, Alpha, Beta, LinearAlpha, SquareRootBeta

"""
The target model should be assigned to self.model!!

This condition is for the alignment with Trainer class
"""

class Trainer(ABC):
    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 args,
                 scheduler: torch.optim.lr_scheduler.LambdaLR = None,
                 utils: Utils = Utils(),
                 device: str = 'cpu',
                 **kwargs):

    
        self.model = model
        self.optimizer = optimizer
        self.args = args
        self.scheduler = scheduler
        self.utils = utils
        self.device = device

        self.use_wandb = args.use_wandb
        self.use_board = args.use_board
        self.val_freq = args.val_freq

        os.makedirs('./checkpoints', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m_%H%M")
        self.prefix = f"{args.model.lower()}_{timestamp}"

    @abstractmethod
    def get_loss(self, x: torch.Tensr):
        """
        Don't forget to move data and target to self.model's hardware. 
        Use '.to(self.model.device)' method.
        """
        pass

    def train(self, train_dl: DataLoader, val_dl: DataLoader = None):

        if self.use_wandb:
            wandb.init(
                project=f"LCH_Generative_Models",
                name=f"{self.args.model.lower()}",
                config={
                    'model': self.args.model,
                    'data': self.args.data_name,
                    'optimizer': self.args.optimizer,
                    'scheduler': self.args.scheduler,
                    'epochs': self.args.epochs,
                    'batch_size': self.args.batch_size,
                    'lr': self.args.lr,
                    'weight_decay': self.args.weight_decay
                    }
                )
            
        if self.use_board:
            pass

        train_best_loss = float('inf')
        val_best_loss = float('inf')
        epochs = self.args.epochs
        for epoch in range(epochs):
            self.model.train()

            total_loss = 0
            samples = 0
            for i, batch in enumerate(tqdm(train_dl, leave=False)):

                loss = self.get_loss(batch)
                self.optimizer.zero_grad()
                loss.backward()

                if self.args.grad_clip() > 0:
                    nn.utils.clip_grad_norm(self.model.parameters(), self.args.grad_clip)
                self.optimizer.step()

                total_loss += loss.item() * batch[0].size(0)
                samples += batch[0].size(0)

            total_loss /= samples
            train_metrics = {'loss': total_loss}
            if total_loss < train_best_loss:
                    train_best_loss = total_loss


            val_metrics = {}
            if (val_dl is not None) and ((epoch + 1) % self.val_freq == 0 and (epoch == epochs - 1)):
                val_total_loss = self.validate(val_dl)
                val_metrics['loss'] = val_total_loss

                if val_total_loss < val_best_loss:
                    val_best_loss = val_total_loss

                    save_path = os.path.join('./checkpoints', f"{self.prefix}_best")
                    torch.save(self.model.parameters(), save_path)

            if self.scheduler is not None:
                self.scheduler.step()

            all_metrics = {'train': train_metrics, 'val': val_metrics}
            self.log_metrics(all_metrics, [train_best_loss, val_best_loss], epoch)

            save_path = os.path.join('./checkpoints', self.prefix)
            torch.save({
                "epoch": epoch,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict() if self.scheduler else None,
                "metric": all_metrics, # needs to change whether saving the best checkpoints showing best performance
                }, save_path)


    @torch.no_grad()
    def validate(self, dl: DataLoader):
        self.model.eval()

        total_loss = 0
        samples = 0
        for batch in tqdm(dl, leave=False):
            loss = self.get_loss(batch)

            total_loss += loss.item() * batch[0].size(0)
            samples += batch[0].size(0)

        total_loss /= samples
        return total_loss

    def log_metrics(self, metrics: dict, best: List[float], epoch: int):
        log_list = []
        for phase, results in metrics.items(): # phase == 'train' or 'val'
            if not results:
                continue

            if self.use_wandb:
                metric_dict = {f"{self.args.data_name}/{phase}/{k}": v for k, v in results.items()}
                wandb.log(metric_dict, step=epoch)

            metric_items = [f"{k}: {v:.4f}" for k, v in results.items()]
            log_list.append(f"{phase}: {' | '.join(metric_items)}")

        print(f"Epoch {epoch} | {' || '.join(log_list)}")


        current_lr = self.optimizer.param_groups[0]['lr']
        if self.use_wandb:
            wandb.log({
                'learning_rate': current_lr,
                'train_best': best[0],
                'val_best': best[1]
            }, step=epoch)


class VAETrainer(Trainer):
    def __init__(self, args):
        super().__init__()

        self.model = VAE(args)

        self.recon_loss_fn = nn.MSELoss(reduction='sum')
        self.kld_loss_fn = nn.KLDivLoss(reduction='sum')
        
    def get_loss(self, x: torch.Tensor):
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

        
class ConditionalFlowMatchingTrainer(Trainer):
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

    def get_loss(self, batch_size: int):
        z = self.path.p_data.sample(batch_size) # (bs, dim)
        t = torch.rand(batch_size, 1).to(z)     # (bs, 1)
        x = self.path.sample_conditional_path(x, z, t)

        ut_theta = self.model(x, t)
        ut_target = self.path.conditional_vector_field(x, z, t)
        mse = F.mse_loss(ut_theta, ut_target)
        return mse


class ConditionalScoreMatchingTrainer(Trainer):
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

    def get_loss(self, batch_size: int):
        z = self.path.p_data.sample(batch_size) # (bs, dim)
        t = torch.rand(batch_size, 1).to(z)     # (bs, 1)
        x = self.path.sample_conditional_path(x, z, t)

        s_theta = self.model(x, t)
        s_target = self.path.conditional_score(x, z, t)
        mse = F.mse_loss(s_theta, s_target)
        return mse

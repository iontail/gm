import torch
import torch.nn as nn

from .models import VAE

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


        
        
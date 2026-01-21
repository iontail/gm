import torch
import os
import random
import numpy as np
import torch.nn as nn

from PIL import Image
from torchvision import transforms, datasets
from torch.optim import SGD, Adam, AdamW

from .scheduler import get_scheduler

from ..models import VAE
from ..trainer import *



class Utils:
    @staticmethod
    def _load_img(dir_path, img_size, transform_compose = None):
        if transform_compose is None:
            transform = transforms.Compose([transforms.Resize((img_size, img_size))])
        else:
            transform = transforms.Compose(
                [transforms.Resize((img_size, img_size))] + transform_compose.transforms
            )

        dataset = datasets.ImageFolder(dir_path, transform=transform)
        return dataset
    
    @staticmethod
    def _load_img_list(dir_path, img_size):
        fractal_img_paths = [os.path.join(dir_path, fname) for fname in os.listdir(dir_path) if fname.endswith(('.png', '.jpg', '.jpeg'))]
        return [Image.open(path).convert('RGB').resize((img_size, img_size)) for path in fractal_img_paths]
    

    @staticmethod
    def _setup_reproducibility(seed: int = 42, deterministic: bool = True):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True


    @staticmethod
    def _setup_device(device: str = 'auto'):
        pass


    @staticmethod
    def _get_model(model_name: str, **kwargs):
        model_name = model_name.lower()

        if model_name == 'vae':
            return VAE(**kwargs)
        else:
            raise NotImplementedError(f'Model {model_name} is not implemented.')
        

    @staticmethod
    def _get_trainer(model_name: str, **kwargs):
        model_name = model_name.lower()

        if model_name == 'vae':
            return VAETrainer(**kwargs)
        else:
            raise NotImplementedError(f'Trainer for {model_name} is not implemented.')

    @staticmethod   
    def _get_scheduler(
                       optimizer,
                       scheduler_name: str = 'constant',
                       warmup_epochs: int = 0,
                       warmup_start_lr: int = 0.0,
                       total_epochs: int = -1,
                       min_lr: float = 1e-6,
                       milestones: list[int] = None,
                       gamma: float = 0.1):
        return get_scheduler(optimizer,
                             scheduler_name,
                             warmup_epochs,
                             warmup_start_lr,
                             total_epochs,
                             min_lr,
                             milestones,
                             gamma)
    
    @staticmethod
    def _setup_optimizer(params, args):
        
        optimizer_name = args.optimizer.lower()
        if optimizer_name == 'sgd':
            return SGD(params,
                       lr = args.lr,
                       momentum=0.9,
                       weight_decay=args.weight_decay
                       )
        
        elif optimizer_name == 'adam':
            return Adam(params,
                        lr=args.lr,
                        betas=(0.9, 0.999), # uses default (change if needed)
                        weight_decay=args.weight_decay
                        )
        
        elif optimizer_name == 'adamw':
            return AdamW(params,
                        lr=args.lr,
                        betas=(0.9, 0.999), # uses default (change if needed)
                        weight_decay=args.weight_decay
                        )
    

        
        
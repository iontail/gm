import torch
import os
import random
import numpy as np
import torch.nn as nn

from PIL import Image
from torchvision import transforms, datasets

from ..models import VAE
from ..loss_generator import VAELossGenerator



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
        

    def _get_loss_generator(self, model: nn.Module, **kwargs):
        loss_name = loss_name.lower()

        if loss_name == 'vae':
            return VAELossGenerator(model, **kwargs)
        else:
            raise NotImplementedError(f'Loss generator {loss_name} is not implemented.')
        
        
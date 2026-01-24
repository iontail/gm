import torch

from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from utils.utils import Utils
from typing import Callable, List

from ..models.diffusion import Sampleable

class CustomDataset(Dataset):
    def __init__(self, dir_path: str, img_size: int = 64, transform: transforms.Compose = None):

        # define how to load your data 
        self.utils = Utils()
        dataset = self.utils.load_img(dir, img_size=img_size, transform_compose=transform)
        self.data = [data[0] for data in dataset]
        self.labels = [data[1] for data in dataset]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            data = self.transform(data)
        return data, label
    

def custom_collate_fn(batch):
    data = torch.stack([item[0] for item in batch])
    targets = torch.tensor([item[1] for item in batch])
    return {'data': data, 'targets': targets}
    

def get_dataloader(data_name: str,
                   batch_size: int,
                   data_path: str = './data',
                   transform : transforms.Compose = None,
                   img_size: int = 32,
                   train: bool = True,
                   shuffle: bool = True,
                   num_workers: int = 2,
                   collate_fn: Callable[List] = None
                   ):
    data_name = data_name.lower()

    if transform is None:
        transform = transforms.ToTensor()

    if data_name == 'cifar10':
        if train:
            dataset = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
        else:
            dataset = datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)
    else:
        dataset = CustomDataset(dir_path=data_path, img_size=img_size, transform=transform)


    if collate_fn is None:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)


class DataSampler(nn.Module, Sampleable):
    """
    Sampleable wrapper for the MNIST dataset
    """
    def __init__(self, dataset:Dataset):
        super().__init__()
        self.dataset = dataset

        self.dummy = nn.Buffer(torch.zeros(1))

    def sample(self, num_samples: int):
        """
        Samples the data points. z ~ p_data
        Args:
            - num_samples: the desired number of samples
        Returns:
            - samples: Shape - (batch_size, c, h, w)
            - labels: Shape - (batch_size, label_dim)
        """
        if num_samples > len(self.dataset):
            raise ValueError(f'num_samples exceeds data size: {len(self.dataset)}.')

        indices = torch.randperm(len(self.dataset))[:num_samples]
        samples, labels = zip(*[self.dataset[idx] for idx in indices])
        samples = torch.stack(samples).to(self.dummy.device)
        labels = torch.tensor(labels, dtype=torch.int64).to(self.dummy.device)
        return samples, labels
        

def get_sampler(dir_path: str, img_size: int = 64, transform: transforms.Compose = None):
    if transform is None:
        transform = transforms.ToTensor()

    dataset = CustomDataset(dir_path, img_size, transform)
    return DataSampler(dataset)
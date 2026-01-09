import torch

from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from utils.utils import Utils


class CustomDataset(Dataset):
    def __init__(self, dir_path: str, transform: transforms.Compose = None):

        # define how to load your data 
        self.utils = Utils()
        dataset = self.utils.load_img(dir, img_size=64, transform_compose=transform)
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
                   collate_fn: bool = False
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


    if collate_fn:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=custom_collate_fn)
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)



import torch
import argparse
import torch.nn as nn

from torchvision import transforms

from trainer import VAELossGenerator
from utils import get_dataloader, Utils


def get_args():
    args = argparse.ArgumentParser()

    args.add_argument('--model', type=str, default='vae', choices=['vae'])
    args.add_argument('--seed', type=int, default=42)
    args.add_argument('--device', type=str, default='auto')
    args.add_argument('--data_name', type=str, default='cifar10', choices=['mnist', 'cifar10'])


    args.add_argument('--epochs', type=int, default=100)
    args.add_argument('--batch_size', type=int, default=128)
    args.add_argument('--optimizer', type=str, default='adam', choices=['sgd', 'adam', 'adamw'])
    args.add_argument('--lr', type=float, default=1e-3)
    args.add_argument('--weight_decay', type=float, default=1e-5)
    args.add_argument('--grad_clip', type=float, default=1.0, help="set to negative value to disable grad clip")


    args.add_argument('--train_path', type=str, default='./data/train')
    args.add_argument('--val_path', type=str, default='./data/val')
    args.add_argument('--test_path', type=str, default='./data/test')

    args.add_argument('--num_workers', type=int, default=2)
    args.add_argument('--img_size', type=int, default=32)


    args = args.parse_args()

    if args.data in ['mnist', 'cifar10']:
        args.img_size = 32
    else:
        args.img_size = 64 # please define image size for other dataset

    return args


class Trainer:
    def __init__(self,
                 loss_generator: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 args,
                 utils: Utils,
                 device: str = 'cpu',
                 **kwargs):

        
        self.loss_generator = loss_generator
        self.model = loss_generator.model
        self.optimizer = optimizer
        self.args = args
        self.utils = utils
        self.device = device



    def train():
        pass

    def validate():
        pass




def main():
    args = get_args()
    utils = Utils()

    utils._setup_reproducibility(seed=args.seed, deterministic=True)

    device = utils._setup_device(device='auto') # not implemented yet

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(args.img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor()
    ])


    train_dl = get_dataloader(data_name=args.data_name,
                              batch_size=args.batch_size,
                              data_path=args.train_path,
                              transform=train_transforms,
                              img_size=args.img_size,
                              train=True,
                              shuffle=True,
                              num_workers=args.num_workers,
                              collate_fn=False,
                              num_workers=args.num_workers
                              )
    
    val_transforms = get_dataloader(data_name=args.data_name,
                                    batch_size=args.batch_size,
                                    data_path=args.val_path,
                                    transform=val_transforms,
                                    img_size=args.img_size,
                                    train=False,
                                    shuffle=False,
                                    num_workers=args.num_workers,
                                    collate_fn=False,
                                    num_workers=args.num_workers
                                    )
    
    model = utils._get_model(model_name=args.model, img_size=args.img_size).to(device)
    loss_generator = utils._get_trainer(model_name=args.model).to(device)
    optimizer = utils._setup_optimizer(params=loss_generator.model.parameters(), args=args)

    trainer = Trainer(loss_generator=loss_generator,
                      optimizer=optimizer,
                      args=args,
                      utils=utils,
                      device=device
                      )
    

    trainer.train()
    trainer.validate()


if __name__ == "__main__":
    main()
    print("Training complete.")

    
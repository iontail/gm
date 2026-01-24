import torch
import argparse
import torch.nn as nn

from torchvision import transforms
from typing import List

from trainer import VAELossGenerator
from utils import get_dataloader, Utils, get_sampler
from models.diffusion.p_path import GaussianConditionalProbabilityPath, LinearAlpha, LinearBeta



def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='vae', choices=['vae'])
    parser.add_argument('--learning', type=str, default='gan', choices=['vae', 'gan', 'sde'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--data_name', type=str, default='cifar10', choices=['mnist', 'cifar10'])
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--use_board', action='store_true', help="Whether to use tensro board for logging.")


    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--optimizer', type=str, default='adam', choices=['sgd', 'adam', 'adamw'])
    parser.add_argument('--scheduler', type=str, default='constant', choices=['constant', 'cosine', 'step'])
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--grad_clip', type=float, default=1.0, help="Set to negative value to disable grad clip")
    parser.add_argument('--val_freq', type=int, default=-1, help="Evaluation frequency. Set default -1 to unactivate the evaluation step.")


    parser.add_argument('--train_path', type=str, default='./data/train')
    parser.add_argument('--val_path', type=str, default='./data/val')
    parser.add_argument('--test_path', type=str, default='./data/test')

    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--img_size', type=int, default=32)


    args = parser.parse_args()

    if args.data in ['mnist', 'cifar10']:
        args.img_size = 32
    else:
        args.img_size = 64 # please define image size for other dataset

    return args

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


    if args.learning == 'sde':
        train_sampler = get_sampler(train_path, args.img_size, train_transforms).to(device)
        train_path = GaussianConditionalProbabilityPath(
            p_data=train_sampler,
            p_simple_shape=[1, 32, 32], # please check the shape of data
            alpha=LinearAlpha(),
            beta= LinearBeta()
        ).to(device)

        val_sampler = get_sampler(val_path, args.img_size, train_transforms).to(device)
        val_path = GaussianConditionalProbabilityPath(
            p_data=val  _sampler,
            p_simple_shape=[1, 32, 32], # please check the shape of data
            alpha=LinearAlpha(),
            beta= LinearBeta()
        ).to(device)
    else:
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
        
        val_dl = get_dataloader(data_name=args.data_name,
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
    optimizer = utils._setup_optimizer(params=loss_generator.model.parameters(), args=args)
    scheduler = utils._get_scheduler(optimizer, args.scheduler) # more arguments
    
    trainer_dict = {
        "model": model,
        "optimizer": optimizer,
        "args": args,
        "scheduler": scheduler,
        "utils": utils,
        "device": device
    }

    trainer = utils._get_trainer(args.model, trainer_dict)

    if args.learning == 'sde':
        trainer.train(train_path, val_path)
        final_loss = trainer.validate(val_path)
    else:
        trainer.train(train_dl=train_dl, val_dl=val_dl)
        final_loss = trainer.validate(val_dl)

    
    print(f"Final Loss: {final_loss}")



if __name__ == "__main__":
    main()
    print("Training complete.")

    
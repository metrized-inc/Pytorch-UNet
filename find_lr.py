import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import wandb
import logging
import json
from PIL import Image
from pathlib import Path

from torch_lr_finder import LRFinder
from unet import UNet

from utils.data_loading import BasicDataset
from torch.utils.data import DataLoader, random_split


dir_img = Path('./data/imgs/')
dir_mask = Path('./data/masks/')
dir_checkpoint = Path('./checkpoints/')
val_percent = 0.1

def sample_lr(args):

    run = wandb.init(
        project='U-Net',
        entity='hojinchang',
        job_type="lr tuning",
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    model = UNet(n_channels=3, n_classes=2, bilinear=True)
    logging.info(f'Network:\n'
                f'\t{model.n_channels} input channels\n'
                f'\t{model.n_classes} output channels (classes)\n'
                f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        model.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    

    # 1. Create dataset
    dataset = BasicDataset(dir_img, dir_mask, args.img_scale)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=args.batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    # val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)


    # Choose optimizer settings
    # optimizer = optim.Adam(model.parameters(), lr=args.start_lr, weight_decay=args.weight_decay)
    # optimizer = optim.SGD(model.parameters(), lr=args.start_lr, weight_decay=args.weight_decay, momentum=0.9)
    optimizer = optim.RMSprop(model.parameters(), lr=args.start_lr, weight_decay=args.weight_decay, momentum=0.9)         
    criterion = nn.CrossEntropyLoss()
    lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
    lr_finder.range_test(train_loader, end_lr=args.end_lr, num_iter=args.num_iter)

    for x, y in zip(lr_finder.history["lr"], lr_finder.history["loss"]):
        wandb.log({"lr": x, "loss": y})

    run.finish()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--img_scale", type=float, default=1, help='Downscaling factor of the images'
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-3, help="Weight decay l2 factor"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size"
    )
    parser.add_argument(
        "--start_lr", type=float, default=1e-6, help="Start learning rate sampling"
    )
    parser.add_argument(
        "--end_lr", type=float, default=10, help="End learning rate sampling"
    )
    parser.add_argument(
        "--num_iter", type=int, default=100, help="Num of plot samples"
    )
    parser.add_argument(
        '--load', type=str, default=False, help='Load model from a .pth file'
    )

    args = parser.parse_args()

    import gc
    # del variables
    gc.collect()

    sample_lr(args)

    print(torch.cuda.is_available())


if __name__ == "__main__":
    main()

import argparse
import logging
import sys
import glob
import re
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss
from utils.output_file_path import increment_path
from utils.weighted_cross_entropy import weighted_cross_entropy_loss
from evaluate import evaluate
from unet import UNet


def train_net(args,
              net,
              device,
              save_dir,
              epochs: int = 5,
              batch_size: int = 1,
              learning_rate: float = 0.001,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              img_size: int = 480,
              amp: bool = False):

    # 1. Create dataset
    try:
        dataset = CarvanaDataset(dir_img, dir_mask, img_size)
    except (AssertionError, RuntimeError):
        dataset = BasicDataset(dir_img, dir_mask, img_size)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, img_size=img_size,
                                  val_percent=val_percent, save_checkpoint=save_checkpoint, amp=amp))

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Images size:  {img_size}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        verbose=False,
    )
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    global_step = 0

    save_dir = str(increment_path(Path(dir_checkpoint) / 'exp'))

    # 5. Begin training
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image']
                true_masks = batch['mask']

                assert images.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.cuda.amp.autocast(enabled=amp):
                    masks_pred = net(images)

                    # loss = weighted_cross_entropy_loss(masks_pred, true_masks) + dice_loss(F.softmax(masks_pred, dim=1).float(),
                    #                                                                 F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
                    #                                                                 multiclass=True)

                    loss = weighted_cross_entropy_loss(masks_pred, true_masks)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (10 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in net.named_parameters():
                            tag = tag.replace('/', '.')
                            histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score = evaluate(net, val_loader, device)
                        scheduler.step(val_score)

                        logging.info('Validation Dice score: {}'.format(val_score))
                        experiment.log({
                            'learning rate': optimizer.param_groups[0]['lr'],
                            'validation Dice': val_score,
                            'images': wandb.Image(images[0].cpu()),
                            'masks': {
                                'true': wandb.Image(true_masks[0].float().cpu()),
                                'pred': wandb.Image(torch.softmax(masks_pred, dim=1).argmax(dim=1)[0].float().cpu()),
                            },
                            'step': global_step,
                            'epoch': epoch,
                            **histograms
                        })

        if save_checkpoint:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), str(os.path.join(save_dir, 'checkpoint_epoch{}.pth'.format(epoch + 1))))
            logging.info(f'Checkpoint {epoch + 1} saved!')

    logging.info('Checkpoints saved to {}'.format(save_dir))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default=None, help='Path to images')
    parser.add_argument('--mask_path', type=str, default=None, help='Path to segmentation masks')
    parser.add_argument('--img_size', type=int, default=480, help='Train/val image size')
    parser.add_argument('--n_classes', type=int, default=2, help='Number of classes (n_classes = 2 means 1 background and 1 object')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.00001, help='Learning rate')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--validation', type=float, default=0.1, help='Percent of the data that is used as validation (0-1)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')

    return parser.parse_args()


def main():
    args = get_args()

    global dir_img, dir_mask, dir_checkpoint
    dir_img = Path(args.img_path)
    dir_mask = Path(args.mask_path)
    dir_checkpoint = Path('./checkpoints/')
    save_dir = str(increment_path(Path(dir_checkpoint) / 'exp'))

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    net = UNet(n_channels=3, n_classes=args.n_classes, bilinear=True)

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    try:
        train_net(args=args,
                  net=net,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  device=device,
                  img_size=args.img_size,
                  val_percent=args.val,
                  amp=args.amp,
                  save_dir=save_dir)

    except KeyboardInterrupt:
        torch.save(net.state_dict(), os.path.join(save_dir,'INTERRUPTED.pth'))
        logging.info('Saved interrupt')
        sys.exit(0)


if __name__ == '__main__':
    main()

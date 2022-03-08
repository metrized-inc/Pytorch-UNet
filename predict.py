import argparse
import logging
import os
import glob
from tqdm import tqdm
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask
from utils.output_file_path import increment_path


def get_output_filenames(args):
    def _generate_name(fn):
        split = os.path.splitext(fn)
        return f'{split[0]}_OUT{split[1]}'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))


def predict_img(net,
                full_img,
                device,
                img_size=480,
                out_threshold=0.5):
                
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(full_img, img_size, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)[0]
        else:
            probs = torch.sigmoid(output)[0]

        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((full_img.size[1], full_img.size[0])),
            transforms.ToTensor()
        ])

        full_mask = tf(probs.cpu()).squeeze()

    if net.n_classes == 1:
        return (full_mask > out_threshold).numpy()
    else:
        return F.one_hot(full_mask.argmax(dim=0), net.n_classes).permute(2, 0, 1).numpy()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Model filepath')
    parser.add_argument('--img_path', type=str, nargs='+', help='Path to test images', required=True)
    parser.add_argument('--img_size', type=int, default=480, help='Size to reshape image')
    parser.add_argument('--n_classes', type=int, default=2, help='Number of classes (n_classes = 2 means 1 background and 1 object')
    parser.add_argument('--output', type=str, default='output', help='Output path')
    parser.add_argument('--viz',  action='store_true', help='Visualize the images as they are processed')
    parser.add_argument('--no-save', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', type=float, default=0.5, help='Minimum probability value to consider a mask pixel white')

    return parser.parse_args()

def main():
    args = get_args()
    input_folder = args.input[0]

    image_paths = []
    image_names = []
    for image in os.listdir(input_folder):
        full_image_path = os.path.join(input_folder, image)
        image_paths.append(full_image_path)
        image_names.append(image)

    save_dir = str(increment_path(Path(args.output[0]) / 'inference'))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    net = UNet(n_channels=3, n_classes=args.n_classes)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info('Model loaded!')

    for i, (img_path, img_name) in enumerate(tqdm(zip(image_paths, image_names), total=len(image_paths))):
        logging.info(f'\nPredicting image {img_name} ...')
        img = Image.open(img_path)

        mask = predict_img(net=net,
                           full_img=img,
                           img_size=args.img_size,
                           out_threshold=args.mask_threshold,
                           device=device)

        if not args.no_save:
            out_filename = os.path.join(save_dir, img_name)
            result = mask_to_image(mask)
            result.save(out_filename)
            logging.info(f'Mask saved to {out_filename}')

        if args.viz:
            logging.info(f'Visualizing results for image {img_name}, close to continue...')
            plot_img_and_mask(img, mask)

    logging.info('Inference results saved to {}'.format(save_dir))


if __name__ == '__main__':
    main()
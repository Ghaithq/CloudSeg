import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import pickle
from os.path import splitext
from os import listdir
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
import tifffile as tiff
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch import Tensor
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import pandas as pd
import argparse
from skimage.transform import resize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BasicDataset(Dataset):
    def __init__(self, imgs_dir, scale=1):
        self.imgs_dir = imgs_dir
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        
        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        self.ids.sort()        
        print(len(self.ids))

    def __len__(self):
        return len(self.ids)

    def preprocess(cls, tif_img, scale):
        img_nd = np.array(tif_img, dtype=np.float32)
        # Ensure 3D (H,W,channels)
        if img_nd.ndim == 2:
            img_nd = np.expand_dims(img_nd, axis=2)
        # Channel-first conversion: (C, H, W)
        img_trans = img_nd.transpose((2, 0, 1))

        # Subtract per-band minimum so each channel starts from zero
        # shape: (C, 1, 1)
        _, height, width = img_trans.shape
        if height == 384 and width == 384:
            img_trans = np.clip((img_trans - 5000) / 3, a_min=0, a_max=None)
        band_max = img_trans.max(axis=(1, 2), keepdims=True)
        img_trans = img_trans / np.maximum(band_max, 1)

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        img_file = glob(f"{self.imgs_dir}/{idx}.tif")

        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        img = tiff.imread(img_file[0])
        img = self.preprocess(img, self.scale)
        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
        }


def predict_img(net, full_img, device, scale_factor=1, out_threshold=0.5):
    net.eval()
    img = full_img
    img = img.unsqueeze(0)
    img = img.to(device=device)
    with torch.no_grad():
        mask = net(img) > out_threshold
    return mask.cpu().squeeze().numpy()

import numpy as np

def rle_encode(mask):
    """
    Encodes a binary mask using Run-Length Encoding (RLE).    
    Args:
        mask (np.ndarray): 2D binary mask (0s and 1s).
    Returns:
        str: RLE-encoded string, or a single space " " if mask is all zeros.
    """
    if np.sum(mask) == 0:
        return " "  # As it seems that kaggle reject nulls. We'll handle cloud-free images with empty spaces.
    
    pixels = mask.flatten(order='F')  # Flatten in column-major order
    pixels = np.concatenate([[0], pixels, [0]])  # Add padding to detect transitions
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1  # Get transition indices
    runs[1::2] -= runs[::2]  # Compute run lengths
    runs[::2] -= 1  # Make it 0-indexed instead of 1-indexed

    return " ".join(map(str, runs))  # Convert to string format

def rle_decode(mask_rle: str, shape=(256, 256)) -> np.ndarray:
    """Decodes an RLE-encoded string into a binary mask with validation checks."""
    
    if not isinstance(mask_rle, str) or not mask_rle.strip() or mask_rle.lower() == 'nan':
        # Return all-zero mask if RLE is empty, invalid, or NaN
        return np.zeros(shape, dtype=np.uint8)
    
    try:
        s = list(map(int, mask_rle.split()))
    except:
        raise Exception("RLE segmentation must be a string and containing only integers")
    
    if len(s) % 2 != 0:
        raise Exception("RLE segmentation must have even-length (start, length) pairs")
    
    if any(x < 0 for x in s):
        raise Exception("RLE segmentation must not contain negative values")
    
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    starts, lengths = s[0::2], s[1::2]
    
    for start, length in zip(starts, lengths):
        if start >= mask.size or start + length > mask.size:
            raise Exception("RLE indices exceed image size")
        mask[start:start + length] = 1
    
    return mask.reshape(shape, order='F')  # Convert to column-major order


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict masks for images in the dataset')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the directory containing the images')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pickled model file')
    parser.add_argument('--output_file', type=str, default='submission.csv', help='Output CSV file name')
    args = parser.parse_args()

    # Initialize dataset with the provided path
    dataset = BasicDataset(args.dataset_path+"/")

    # Load model from the provided path and move to device
    model_path = args.model_path
    model = torch.load(model_path, map_location=device) 
    model.eval()

    records = []
    for x in range(0,len(dataset)):
        img=dataset[x]['image']
        out=predict_img(model,img,device,out_threshold=0.5)
        out = resize(out, (256, 256), order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)
        img_np = img.cpu().numpy().transpose(1, 2, 0) 
        print(dataset.ids[x])
        records.append({
            'id': str(dataset.ids[x]),
            'segmentation': str(rle_encode(out))
        })

    # Save results to the specified output file
    df = pd.DataFrame(records, columns=['id', 'segmentation'])
    df.to_csv(args.output_file, index=False)
    print(f"Written {len(df)} rows to {args.output_file}")
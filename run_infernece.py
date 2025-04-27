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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BasicDataset(Dataset):
    def __init__(self, imgs_dir, scale=1):
        self.imgs_dir = imgs_dir
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        print(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, tif_img, scale):
        img_nd = np.array(tif_img)
        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 65535
        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        img_file = glob(self.imgs_dir + idx + ".tif")
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

def rle_encode(mask):
    """
    Encodes a binary mask using Run-Length Encoding (RLE).
    
    Args:
        mask (np.ndarray): 2D binary mask (0s and 1s).
    
    Returns:
        str: RLE-encoded string.
    """
    pixels = mask.flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    runs[::2] -= 1
    return " ".join(map(str, runs))

def rle_decode(mask_rle, shape):
    """
    Decodes an RLE-encoded string into a binary mask.
    
    Args:
        mask_rle (str): RLE-encoded string.
        shape (tuple): (height, width) of the output mask.
    
    Returns:
        np.ndarray: Decoded binary mask.
    """
    if not mask_rle:
        return np.zeros(shape, dtype=np.uint8)

    s = list(map(int, mask_rle.split()))
    starts, lengths = s[0::2], s[1::2]
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for start, length in zip(starts, lengths):
        mask[start:start + length] = 1
    return mask.reshape(shape, order='F')


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
    for x in range(len(dataset)):
        img = dataset[x]['image']
        out=np.resize(predict_img(model,img,device,out_threshold=0.5),(256,256))
        records.append({
            'id': str(dataset.ids[x]),
            'segmentation': str(rle_encode(out))
        })

    # Save results to the specified output file
    df = pd.DataFrame(records, columns=['id', 'segmentation'])
    df.to_csv(args.output_file, index=False)
    print(f"Written {len(df)} rows to {args.output_file}")
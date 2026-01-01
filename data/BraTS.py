import os
import torch
from torch.utils.data import Dataset
import random
import numpy as np
from torchvision.transforms import transforms
import pickle
from scipy import ndimage
from scipy.ndimage import zoom


MODALITY_SETS = {
    "flair":     [0],
    "ct1":       [1],
    "t1":        [2],
    "t2":        [3],
    "ct1_flair": [1, 0],
    "t1_t2":     [2, 3],
    "all":       [0, 1, 2, 3]
}

import pickle
import importlib
import sys
import numpy as np

def pkload(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

class MaxMinNormalization(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        Max = np.max(image)
        Min = np.min(image)
        image = (image - Min) / (Max - Min)

        return {'image': image, 'label': label}


class Random_Flip(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        if random.random() < 0.5:
            image = np.flip(image, 0)
            label = np.flip(label, 0)
        if random.random() < 0.5:
            image = np.flip(image, 1)
            label = np.flip(label, 1)
        if random.random() < 0.5:
            image = np.flip(image, 2)
            label = np.flip(label, 2)

        return {'image': image, 'label': label}

class PadToSize(object):
    def __init__(self, target_size):
        self.target_size = target_size  # (H, W, D)

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        H, W, D = image.shape[:3]
        th, tw, td = self.target_size

        pad_h = max(th - H, 0)
        pad_w = max(tw - W, 0)
        pad_d = max(td - D, 0)

        image = np.pad(
            image,
            ((0, pad_h), (0, pad_w), (0, pad_d), (0, 0)),
            mode='constant'
        )
        label = np.pad(
            label,
            ((0, pad_h), (0, pad_w), (0, pad_d)),
            mode='constant'
        )

        return {'image': image, 'label': label}

class CenterCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size  # (H, W, D)

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        H, W, D = image.shape[:3]
        ch, cw, cd = self.crop_size

        h0 = (H - ch) // 2
        w0 = (W - cw) // 2
        d0 = (D - cd) // 2

        image = image[h0:h0+ch, w0:w0+cw, d0:d0+cd, :]
        label = label[h0:h0+ch, w0:w0+cw, d0:d0+cd]

        return {'image': image, 'label': label}

class Random_Crop(object):
    def __init__(self, crop_size=(128, 128, 128)):
        self.crop_size = crop_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        H, W, D = image.shape[:3]
        ch, cw, cd = self.crop_size

        h0 = random.randint(0, H - ch)
        w0 = random.randint(0, W - cw)
        d0 = random.randint(0, D - cd)

        image = image[h0:h0+ch, w0:w0+cw, d0:d0+cd, :]
        label = label[h0:h0+ch, w0:w0+cw, d0:d0+cd]

        return {'image': image, 'label': label}
    

class Random_intencity_shift(object):
    def __call__(self, sample, factor=0.1):
        image = sample['image']
        label = sample['label']

        scale_factor = np.random.uniform(1.0-factor, 1.0+factor, size=[1, image.shape[1], 1, image.shape[-1]])
        shift_factor = np.random.uniform(-factor, factor, size=[1, image.shape[1], 1, image.shape[-1]])

        image = image*scale_factor+shift_factor

        return {'image': image, 'label': label}


class Random_rotate(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        angle = round(np.random.uniform(-10, 10), 2)
        image = ndimage.rotate(image, angle, axes=(0, 1), reshape=False)
        label = ndimage.rotate(label, angle, axes=(0, 1), reshape=False)

        return {'image': image, 'label': label}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image = sample['image']
        image = np.ascontiguousarray(image.transpose(3, 0, 1, 2))
        label = sample['label']
        label = np.ascontiguousarray(label)

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long()

        return {'image': image, 'label': label}

def transform(sample):
    trans = transforms.Compose([
        PadToSize((240, 240, 160)),   # minimum safe BraTS size
        # Random_rotate(),  # time-consuming
        Random_Crop((128, 128, 128)),
        Random_Flip(),
        Random_intencity_shift(),
        ToTensor()
    ])
    return trans(sample)


def transform_valid(sample):
    trans = transforms.Compose([
        PadToSize((240, 240, 160)),
        CenterCrop((128, 128, 128)),
        ToTensor()
    ])
    return trans(sample)

def down_up_sample_image(image, scale):
    """
    image: np.ndarray [H, W, D, C]
    scale: float (e.g. 0.5, 0.75, 1.0)
    """
    if scale == 1.0:
        return image

    H, W, D, C = image.shape

    down = zoom(
        image,
        zoom=(scale, scale, scale, 1),
        order=1
    )

    up = zoom(
        down,
        zoom=(H / down.shape[0], W / down.shape[1], D / down.shape[2], 1),
        order=1
    )

    up = up[:H, :W, :D, :]

    return up

class BraTS(Dataset):
    def __init__(self, list_file, root='', mode='train', modality_set = 'all', resolution = 1.0):
        self.lines = []
        paths, names = [], []
        with open(list_file) as f:
            for line in f:
                line = line.strip()
                name = line.split('/')[-1]
                names.append(name)
                path = os.path.join(root, line)
                paths.append(path)
                self.lines.append(line)
        self.mode = mode
        self.names = names
        self.paths = paths
        self.modality_idx = MODALITY_SETS[modality_set]
        self.resolution = resolution  # e.g., 1.0, 0.75, 0.5


    def __getitem__(self, item):
        path = self.paths[item]

        if self.mode in ['train', 'valid']:
            image, label = pkload(os.path.join(path, 'data_f32b0.pkl'))
            image = image[..., self.modality_idx] #slices "all" image into modality set (e.g. t1_t2)
            if image.ndim == 3:  
                image = image[..., np.newaxis]

            # Downsample the image
            down_up_sample_image(image, self.resolution)

            sample = {'image': image, 'label': label}

            if self.mode == 'train':
                sample = transform(sample)
            else:
                sample = transform_valid(sample)

            return sample['image'], sample['label']
        else:
            image = pkload(os.path.join((path,'data_f32b0.pkl')))
            image = image[..., self.modality_idx]
            if image.ndim == 3:  
                image = image[..., np.newaxis]
            image = np.pad(image, ((0, 0), (0, 0), (0, 5), (0, 0)), mode='constant')
            image = np.ascontiguousarray(image.transpose(3, 0, 1, 2))
            image = torch.from_numpy(image).float()
            return image

    def __len__(self):
        return len(self.names)

    def collate(self, batch):
        return [torch.cat(v) for v in zip(*batch)]




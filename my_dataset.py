import os

import numpy as np
import pandas as pd
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader

from Utils.utils import load_image, npy_to_float_tensor, channels_first
from Utils.constants import ALL_ATTRIBUTES, IMAGE_PATH, MASK_PATH


class MyDataset(Dataset):

    def __init__(self, train_test_id: pd.DataFrame, args, train: str):

        self.train_test_id = train_test_id[train_test_id['Split'] == train].reset_index(drop=True)
        self.image_path = args.image_path
        self.pretrained = args.pretrained
        self.attribute = args.attribute
        self.mask_use = args.mask_use
        self.augment_list = args.augment_list
        self.prob = args.prob
        self.train = train
        self.cell = args.cell
        self.cell_size = args.cell_size
        self.normalize = args.normalize
        self.aux = args.aux
        self.aux_batch = args.aux_batch

        self.n = self.train_test_id.shape[0]

    def __len__(self):
        return self.n

    def transform_fn(self, image: np.array, mask: np.array):

        image = TF.to_pil_image(image)
        mask_pil_array = [None] * mask.shape[-1]
        for i in range(mask.shape[-1]):
            mask_pil_array[i] = TF.to_pil_image(mask[:, :, i])
        if 'hflip' in self.augment_list:
            if np.random.random() > 0.5:
                image = TF.hflip(image)
                for i in range(mask.shape[-1]):
                    mask_pil_array[i] = TF.hflip(mask_pil_array[i])
        if 'vflip' in self.augment_list:
            if np.random.random() > 0.5:
                image = TF.vflip(image)
                for i in range(mask.shape[-1]):
                    mask_pil_array[i] = TF.vflip(mask_pil_array[i])
        for i in range(mask.shape[-1]):
            mask[:, :, i] = np.array(mask_pil_array[i])
        if 'affine' in self.augment_list:
            angle = np.random.randint(0, 90)
            translate = (np.random.uniform(0, 100), np.random.uniform(0, 100))
            scale = np.random.uniform(0.5, 2)
            shear = np.random.uniform(-10, 10)
            image = TF.affine(image, angle, translate, scale, shear)
            for i in range(mask.shape[-1]):
                mask_pil_array[i] = TF.affine(mask_pil_array[i], angle, translate, scale, shear, fillcolor=-1)
        if 'adjust_brightness' in self.augment_list:
            if np.random.random() < 0.3:
                brightness_factor = np.random.uniform(0.8, 1.2)
                image = TF.adjust_brightness(image, brightness_factor)
        if 'adjust_saturation' in self.augment_list:
            if np.random.random() < 0.3:
                saturation_factor = np.random.uniform(0.8, 1.2)
                image = TF.adjust_saturation(image, saturation_factor)

        image = np.array(image)

        if self.mask_use:
            for i in range(mask.shape[-1]):
                mask[:, :, i] = np.array(mask_pil_array[i])

        mask[mask==0] = -1

        return image, mask

    def __getitem__(self, index: int):

        name = self.train_test_id.iloc[index].ID
        path = self.image_path
        # Load image and masks from npy
        image = np.load(os.path.join(path, IMAGE_PATH, '%s.npy' % name[5:]))
        image = (image / 255.0)

        mask = np.empty([image.shape[0], image.shape[1], len(self.attribute)], dtype='int')
        if self.mask_use:
            file = np.load(os.path.join(path, MASK_PATH, '%s.npy' % name[5:]))
            for i, attr in enumerate(ALL_ATTRIBUTES):
                if attr in self.attribute:
                    mask[:, :, i] = file[:, :, i]
        else:
            mask = image

        if self.train == 'train':
            if self.augment_list:
                image, mask = self.transform_fn(image, mask)

        if self.pretrained:
            if self.normalize:
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                image = (image - mean) / std

        mask[mask == 0] = -1

        if self.mask_use and not self.aux:
            if self.train == 'valid':
                mask.fill(0.)
            elif self.cell:
                mask = quatro_mask_clear(mask, image.shape[0], self.cell_size, self.prob)
            else:
                mask = full_mask_clear(mask, self.prob)
            image_with_mask = np.dstack((image, mask))
        else:
            image_with_mask = image

        labels = np.array([self.train_test_id.loc[index, attr[10:]] for attr in self.attribute])

        if self.aux and self.mask_use:
            im, l = np.array([]), np.array([])
            if self.train == 'train':
                for i in range(self.aux_batch):
                    prob = np.random.choice([0.01, 0.2, 0.8])
                    fill_method = np.random.choice([0, 1])
                    cell_size = np.random.choice([14, 28, 56])
                    if fill_method == 0:
                        mask = quatro_mask_clear(mask, image.shape[0], cell_size, prob)
                    else:
                        mask = full_mask_clear(mask, prob)
                    im = np.dstack((image, mask))
                    if i == 0:
                        image_with_mask = np.array([im])
                    else:
                        image_with_mask = np.concatenate((image_with_mask, np.array([im])), axis=0)
                labels = np.array([labels for i in range(self.aux_batch)])
            else:
                mask.fill(0.)
                image_with_mask = np.dstack((image, mask))

        if self.aux:
            image_with_mask = channels_first(image_with_mask, channel=1)
        else:
            image_with_mask = channels_first(image_with_mask)

        return npy_to_float_tensor(image_with_mask), npy_to_float_tensor(labels), name


def full_mask_clear(mask: np.array, prob: float) -> np.array:
    if np.random.uniform(0, 1) < prob:
        mask.fill(0.)
    return mask


def quatro_mask_clear(mask: np.array, shape, cell_size: int, prob: float) -> np.array:
    for i in range(0, shape, cell_size):
        for j in range(0, shape, cell_size):
            p = np.random.uniform(0, 1)
            if p < prob:
                mask[i:i + cell_size, j:j + cell_size, :].fill(0.)
    return mask


def make_loader(train_test_id: pd.DataFrame, args, train: str = 'train', shuffle: bool = True) -> DataLoader:

    data_set = MyDataset(train_test_id=train_test_id,
                         args=args,
                         train=train)

    batch_size = args.batch_size if train == 'train' else args.batch_size*10

    data_loader = DataLoader(data_set,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=args.workers,
                             pin_memory=True
                             )
    return data_loader

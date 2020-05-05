import os

import numpy as np
import pandas as pd
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader

from Utils.utils import npy_to_float_tensor, channels_first
from Utils.constants import ALL_ATTRIBUTES, IMAGE_PATH, MASK_PATH


def read_all_images(path, train_test_id):
    images_dict = {
        name: np.load(os.path.join(path, IMAGE_PATH, '%s.npy' % name[5:])) for name in train_test_id.ID
    }
    return images_dict


def read_all_masks(path, train_test_id):
    masks_dict = {
        name: np.load(os.path.join(path, MASK_PATH, '%s.npy' % name[5:])) for name in train_test_id.ID
    }
    return masks_dict


class MyDataset(Dataset):

    def __init__(self, train_test_id: pd.DataFrame, args, train: str):

        self.train_test_id = train_test_id[train_test_id['Split'] == train].reset_index(drop=True)

        self.images_dict = read_all_images(args.image_path, self.train_test_id)
        self.masks_dict = read_all_masks(args.image_path, self.train_test_id)
        self.pretrained = args.pretrained
        self.attribute = args.attribute
        self.augment_list = args.augment_list
        self.prob = args.prob
        self.train = train
        self.normalize = args.normalize
        self.indexes = np.isin(ALL_ATTRIBUTES, self.attribute)

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

        for i in range(mask.shape[-1]):
            mask[:, :, i] = np.array(mask_pil_array[i])

        mask[mask==0] = -1

        return image, mask

    def __getitem__(self, index: int):

        name = self.train_test_id.iloc[index].ID

        image = self.images_dict[name]
        image = (image / 255.0)

        mask = self.masks_dict[name][:, :, self.indexes]

        if self.train == 'train':
            if self.augment_list:
                image, mask = self.transform_fn(image, mask)

        if self.pretrained:
            if self.normalize:
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                image = (image - mean) / std

        labels = np.array([self.train_test_id.loc[index, attr[10:]] for attr in self.attribute])

        image = channels_first(image)
        mask = channels_first(mask)

        return npy_to_float_tensor(image), npy_to_float_tensor(mask), npy_to_float_tensor(labels), name


def make_loader(train_test_id: pd.DataFrame, args, train: str = 'train', shuffle: bool = True) -> DataLoader:

    data_set = MyDataset(train_test_id=train_test_id,
                         args=args,
                         train=train)

    data_loader = DataLoader(data_set,
                             batch_size=args.batch_size,
                             shuffle=shuffle,
                             num_workers=args.workers,
                             pin_memory=True
                             )
    return data_loader

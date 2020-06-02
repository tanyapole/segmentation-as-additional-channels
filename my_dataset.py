import os

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from albumentations import Compose, RandomRotate90, Flip, Transpose, ShiftScaleRotate

from Utils.utils import npy_to_float_tensor, npy_to_int_tensor, channels_first, _augment_duo, _augment_one
from Utils.constants import ALL_ATTRIBUTES, IMAGE_PATH, MASK_PATH, YNET, PRETRAIN


class MyDataset(Dataset):

    def __init__(self, train_test_id: pd.DataFrame, args, train: str, train_type: str):

        self.train_test_id = train_test_id[train_test_id['Split'] == train].reset_index(drop=True)
        self.train_type = train_type
        self.image_path = args.image_path
        self.pretrained = args.pretrained
        self.train = train
        self.normalize = args.normalize
        self.transforms = Compose([RandomRotate90(), Flip(), Transpose(), ShiftScaleRotate()])

        self.n = self.train_test_id.shape[0]
        print(self.n, train_type, train)

    def __len__(self):
        return self.n

    def __getitem__(self, index: int):

        name = self.train_test_id.iloc[index].filename
        path = self.image_path

        image = np.load(os.path.join(path, IMAGE_PATH, '%s.npy' % name))
        image = (image / 255.0)

        if self.pretrained:
            if self.normalize:
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                image = (image - mean) / std

        if self.train_type in [PRETRAIN]:
            mask = np.load(os.path.join(path, MASK_PATH, '%s.npy' % name))
            image, mask = _augment_duo(self.transforms, image, mask)
            mask = channels_first(mask)
            image = channels_first(image)
            return npy_to_float_tensor(image), npy_to_float_tensor(mask), name
        else:
            labels = np.array([self.train_test_id.loc[index, attr] for attr in ALL_ATTRIBUTES])
            image = _augment_one(self.transforms, image)
            image = channels_first(image)
            return npy_to_float_tensor(image), npy_to_float_tensor(labels), name


def make_loader(train_test_id: pd.DataFrame, args, train_type: str,
                train: str = 'train', shuffle: bool = True) -> DataLoader:

    data_set = MyDataset(train_test_id=train_test_id,
                         args=args,
                         train=train,
                         train_type=train_type)

    data_loader = DataLoader(data_set,
                             batch_size=args.batch_size,
                             shuffle=shuffle,
                             num_workers=args.workers,
                             pin_memory=True
                             )
    return data_loader

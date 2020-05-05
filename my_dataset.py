import os

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from Utils.utils import npy_to_float_tensor, channels_first
from Utils.constants import ALL_ATTRIBUTES, IMAGE_PATH, MASK_PATH, YNET


class MyDataset(Dataset):

    def __init__(self, train_test_id: pd.DataFrame, args, train: str, train_type: str):

        self.train_test_id = train_test_id[train_test_id['Split'] == train].reset_index(drop=True)
        self.train_type = train_type
        self.image_path = args.image_path
        self.pretrained = args.pretrained
        self.attribute = args.attribute
        self.train = train
        self.normalize = args.normalize
        self.indexes = np.isin(ALL_ATTRIBUTES, self.attribute)

        self.n = self.train_test_id.shape[0]
        print(self.n, train_type, train)

    def __len__(self):
        return self.n

    def __getitem__(self, index: int):

        name = self.train_test_id.iloc[index].ID
        path = self.image_path

        image = np.load(os.path.join(path, IMAGE_PATH, '%s.npy' % name[5:]))
        image = (image / 255.0)

        if self.pretrained:
            if self.normalize:
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                image = (image - mean) / std

        labels = np.array([self.train_test_id.loc[index, attr[10:]] for attr in self.attribute])

        image = channels_first(image)

        if self.train_type == YNET:
            mask = np.load(os.path.join(path, MASK_PATH, '%s.npy' % name[5:]))[:, :, self.indexes]
            mask = channels_first(mask)
            return npy_to_float_tensor(image), npy_to_float_tensor(mask), npy_to_float_tensor(labels), name
        else:
            return npy_to_float_tensor(image), None, npy_to_float_tensor(labels), name


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

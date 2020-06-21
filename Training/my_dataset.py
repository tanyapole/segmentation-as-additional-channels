import os

from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

from Utils.utils import load_image, npy_to_float_tensor, channels_first


class MyDataset(Dataset):

    def __init__(self, train_test_id: pd.DataFrame, args, train: str):

        self.train_test_id = train_test_id[train_test_id['Split'] == train].reset_index(drop=True)
        self.image_path = args.image_path
        self.pretrained = args.pretrained
        self.attribute = args.attribute
        self.normalize = args.normalize
        self.train = train
        self.n = self.train_test_id.shape[0]

    def __len__(self):
        return self.n

    def __getitem__(self, index):

        name = self.train_test_id.iloc[index].ID
        path = self.image_path

        image = np.load(os.path.join(path, '%s.npy' % name[5:]))
        image = (image / 255.0)
        if self.pretrained and self.normalize:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = (image - mean) / std

        image = channels_first(image)

        labels = np.array([self.train_test_id.loc[index, attr[10:]] for attr in self.attribute])

        return npy_to_float_tensor(image), npy_to_float_tensor(labels), name


def make_loader(train_test_id: pd.DataFrame, args, train='train', shuffle=True) -> DataLoader:

    data_set = MyDataset(train_test_id=train_test_id,
                         args=args,
                         train=train)
    if train != 'train':
        batch_size = args.batch_size*10
    else:
        batch_size = args.batch_size
    data_loader = DataLoader(data_set,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=args.workers,
                             pin_memory=True
                             )
    return data_loader

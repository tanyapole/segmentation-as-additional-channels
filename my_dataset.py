import os
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
from PIL import Image
from torchvision import transforms
from Utils.utils import load_image
import torch

class MyDataset(Dataset):

    def __init__(self, train_test_id, args, train):

        self.train_test_id = train_test_id[train_test_id['Split'] == train].reset_index(drop=True)
        self.image_path = args.image_path
        self.pretrained = args.pretrained
        self.attribute = args.attribute
        self.train = train
        self.n = self.train_test_id.shape[0]

    def __len__(self):
        return self.n

    def __getitem__(self, index):

        name = self.train_test_id.iloc[index].ID
        path = self.image_path
        # Load image and from h5
        image = load_image(os.path.join(path, '%s.h5' % name))
        # image = torch.load(os.path.join(path, '%s.torch' % name))
        # print(np.unique(image))
        # print(np.unique(image))
        # image = TF.to_tensor(image)
        # image = np.array(image)
        # image = (image / 255.0)
        if self.pretrained and False:
            image = (image / 255.0)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = (image - mean) / std

        labels = np.array([self.train_test_id.loc[index, attr[10:]] for attr in self.attribute])

        return image, labels, name


def make_loader(train_test_id, args, train=True, shuffle=True):

    data_set = MyDataset(train_test_id=train_test_id,
                         args=args,
                         train=train)
    if not train:
        batch_size = args.batch_size * 10
    else:
        batch_size = args.batch_size
    data_loader = DataLoader(data_set,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=args.workers,
                             pin_memory=True
                             )
    return data_loader

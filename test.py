from torchvision import models
from torch import nn
import numpy as np
from Utils.constants import IMAGE_PATH, MASK_PATH, ALL_ATTRIBUTES
import os
from torch.optim import Adam
import pandas as pd
import torch

from models import create_model
from my_dataset import make_loader

class Args:
    def __init__(self):
        self.attribute = ALL_ATTRIBUTES
        print(self.attribute)
        self.model = 'resnet50'
        self.mask_use = True
        self.freezing = False
        self.pretrained = False
        self.batch_norm = False
        self.optimizer = 'adam'
        self.lr = 0.0001
        self.batch_size = 2
        self.workers = 1
        self.cell = True
        self.cell_size = 14
        self.prob = 0.2
        self.image_path = 'D:/Data/albums/'
        self.augment_list = []
        self.normalize = False
args = Args()
model = create_model(args=args)
#print(model)
print(model)
path = 'D:/Data/albums/images_npy/'
name = os.listdir(path)[0]
img = np.load(path + name)
img = np.moveaxis(img, -1, 0)
timg = torch.from_numpy(img).float()
btimg = torch.unsqueeze(timg, dim=0)

model(btimg)
"""train_test_id = pd.read_csv('Data/train_test_id_with_masks.csv')
path = 'D:/Data/albums/'
batch =np.zeros([2, 224, 224, 8])
for i, name in enumerate(train_test_id.iloc[:2].ID):
    image = np.load(os.path.join(path, IMAGE_PATH, '%s.npy' % name[5:]))
    image = (image / 255.0)
    masks = np.load(os.path.join(path, MASK_PATH, '%s.npy' % name[5:]))
    masks[masks==0] = -1
    image_with_masks = np.dstack((image, masks))
    batch[i] = image_with_masks
batch = np.moveaxis(batch, -1, 1)
batch = torch.tensor(batch, dtype=torch.float32)
out = model(batch)
print(out)

trn_dl = make_loader(train_test_id, args, train='train', shuffle=True)

for i, (image_batch, label_batch, name) in enumerate(trn_dl):
    out = model(image_batch)
    print(out)
    break
"""
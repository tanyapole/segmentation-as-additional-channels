import time
import numpy as np
import pandas as pd
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch.backends import cudnn
from pathlib import Path
from loss import LossBinary
from Utils.utils import write_tensorboard, save_weights
from my_dataset import make_loader
from models import create_model
from metrics import Metrics


class Args:

    def __init__(self):
        self.model ='resnet50'
        self.lr = 0.0001
        self.freezing = False
        self.pretrained = False
        self.mask_use = True
        self.attribute = ['attribute_milia_like_cyst', 'attribute_pigment_network']
        self.batch_size = 2
        self.image_path = 'D:/Data/h5/'
        self.augment_list = []
        self.prob = 0.2
        self.cell = False
        self.cell_size = [14]
        self.aux = False
        self.aux_batch = 2
        self.workers = 2


"""device = 'cuda'
args = Args()
args.mask_use = False
model, optimizer = create_model(args, device)
model.to(device)

print(model)
cp = torch.load('model.pt')['model']

print(cp['conv1.weight'].shape)

model.load_state_dict(cp)
args.mask_use = True
print(args.mask_use)
input_num = 3 + len(args.attribute)
model.conv1 = nn.Conv2d(input_num, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
print(model.conv1.weight)
print(model.conv1.weight.shape)"""

def check_load_vals(image_path, augment_list, pretrained=False,  mode='train', batch_size=2):
    args = Args()
    args.batch_size = batch_size
    args.image_path = image_path
    args.augment_list = augment_list
    args.pretrained = pretrained
    args.mask_use = False
    train_test_id = pd.read_csv('Data/train_test_id_with_masks.csv')
    train_test_id = train_test_id.sample(frac=1).reset_index(drop=True)
    train_test_id.loc[:1900, 'Split'] = 'train'
    train_test_id.loc[1900:, 'Split'] = 'valid'
    loader = make_loader(train_test_id, args, train=mode, shuffle=True)
    for i, (image_batch, labels_batch, names) in enumerate(loader):
        print(image_batch.shape)
        if i > 0:
            return


if __name__ == '__main__':

    #augment_list = ['hflip', 'vflip', 'adjust_saturation', 'adjust_brightness']
    augment_list = []
    check_load_vals(image_path='D:/Data/h5/', augment_list=augment_list, pretrained=True, mode='train', batch_size=2)
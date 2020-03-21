import time
import numpy as np
import pandas as pd
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch.backends import cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau
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


device = 'cuda'
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
print(model.conv1.weight.shape)

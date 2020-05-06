from torchvision import models
from torch import nn
import numpy as np
from Utils.constants import IMAGE_PATH, MASK_PATH, ALL_ATTRIBUTES, YNET, SEED_LIST
from Utils.utils import read_split_data
import os
from torch.optim import Adam
import pandas as pd
import torch

from models import create_model
from my_dataset import make_loader
from metrics import Metrics

device='cuda'
class Args:
    def __init__(self):
        self.attribute = ['attribute_globules', 'attribute_milia_like_cyst']
        self.pretrained = False
        self.optimizer = 'adam'
        self.lr = 0.0001
        self.batch_size = 1
        self.workers = 1
        self.image_path = '/data/ISIC/'
        self.augment_list = []
        self.normalize = True
        self.model_path = 'D:/Data/'
        self.N = 0
args = Args()
model = create_model(args, train_type=YNET)
checkpoint = torch.load('/data/ISIC/model_ynet_pretrain_2cl/' + 'model_ynet_0.pt')
model.load_state_dict(checkpoint['model'])
model.to(device)
train_test_id = read_split_data(SEED=SEED_LIST[0], train_type=YNET)
val_dl = make_loader(train_test_id, args, train_type=YNET, train='valid')
metrics = Metrics(args)
with torch.no_grad():
    for i, (image_batch, masks_batch, labels_batch, names) in enumerate(val_dl):
        image_batch = image_batch.to(device)
        labels_batch = labels_batch.to(device)
        clsf_output, segm_output = model(image_batch)
        segm = (segm_output.data.cpu().numpy() > 0) * 1
        clsf = (clsf_output.data.cpu().numpy() > 0) * 1
        for i in range(2):
            s = segm[:, i, :, :].ravel()
            print(s[s == 1].shape, s[s == 0].shape, 'class %d'%i, clsf[:,i], 'true_labels ',labels_batch[:,i])
        loss = loss1 = loss2 = torch.zeros(1).to(device)
        metrics.valid.update(labels_batch, clsf_output, loss, loss1, loss2)
    print(metrics.valid.compute())
del model

"""path = 'D:/Data/albums/images_npy/'
name = os.listdir(path)[0]
img = np.load(path + name)
img = np.moveaxis(img, -1, 0)
timg = torch.from_numpy(img).float()
btimg = torch.unsqueeze(timg, dim=0)

model(btimg)"""
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
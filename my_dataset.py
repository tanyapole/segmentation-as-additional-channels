import os
import torchvision.transforms.functional as TF
#from keras.preprocessing.image import array_to_img, img_to_array
from torch.utils.data import Dataset, DataLoader
from Utils.utils import load_image
import random
import numpy as np
import torch


class MyDataset(Dataset):

    def __init__(self, train_test_id, args, train):

        self.train_test_id = train_test_id[train_test_id['Split'] == train]
        self.image_path = args.image_path
        self.pretrained = args.pretrained
        self.attribute = args.attribute
        self.mask_use = args.mask_use
        self.augment_list = args.augment_list
        self.prob = args.prob
        self.train = train
        self.all_attributes = ['attribute_globules', 'attribute_milia_like_cyst', 'attribute_negative_network',
                               'attribute_pigment_network', 'attribute_streaks']

        self.cell = args.cell
        self.cell_size = args.cell_size
        self.aux = args.aux
        self.aux_batch = args.aux_batch

        self.n = self.train_test_id.shape[0]

    def __len__(self):
        return self.n

    def transform_fn(self, image, mask):

        image = TF.to_pil_image(image)
        #mask_dupl = mask.copy()
        #mask_dupl[mask_dupl==-1] = 255
        #mask_dupl[mask_dupl==1] = 128
        mask_pil_array = [None] * mask.shape[-1]
        #mask_pil_array_dupl = [None] * mask.shape[-1]
        for i in range(mask.shape[-1]):
            mask_pil_array[i] = TF.to_pil_image(mask[:, :, i])
            #mask_pil_array_dupl[i] = TF.to_pil_image(mask_dupl[:, :, i])
            #mask_pil_array_dupl[i].show()
            #print(np.unique(np.array(mask_pil_array[i])), 'before')
        if 'hflip' in self.augment_list:
            if random.random() > 0.5:
                image = TF.hflip(image)
                for i in range(mask.shape[-1]):
                    mask_pil_array[i] = TF.hflip(mask_pil_array[i])
                    #mask_pil_array_dupl[i] = TF.hflip(mask_pil_array[i])
        if 'vflip' in self.augment_list:
            if random.random() > 0.5:
                image = TF.vflip(image)
                for i in range(mask.shape[-1]):
                    mask_pil_array[i] = TF.vflip(mask_pil_array[i])
                    #mask_pil_array_dupl[i] = TF.hflip(mask_pil_array_dupl[i])
        for i in range(mask.shape[-1]):
            mask[:, :, i] = np.array(mask_pil_array[i])
        print(np.unique(mask), 'middle')
        if 'affine' in self.augment_list:
            angle = random.randint(0, 90)
            translate = (random.uniform(0, 100), random.uniform(0, 100))
            scale = random.uniform(0.5, 2)
            shear = random.uniform(-10, 10)
            image = TF.affine(image, angle, translate, scale, shear)
            for i in range(mask.shape[-1]):
                mask_pil_array[i] = TF.affine(mask_pil_array[i], angle, translate, scale, shear, fillcolor=-1)
                #mask_pil_array_dupl[i] = TF.affine(mask_pil_array_dupl[i], angle, translate, scale, shear, fillcolor=-1)
                #mask_pil_array_dupl[i].show()
        if 'adjust_brightness' in self.augment_list:
            if random.random() < 0.3:
                brightness_factor = random.uniform(0.8, 1.2)
                image = TF.adjust_brightness(image, brightness_factor)
        if 'adjust_saturation' in self.augment_list:
            if random.random() < 0.3:
                saturation_factor = random.uniform(0.8, 1.2)
                image = TF.adjust_saturation(image, saturation_factor)

        image = np.array(image)

        if self.mask_use:
            for i in range(mask.shape[-1]):
                mask[:, :, i] = np.array(mask_pil_array[i])
                #mask_dupl[:, :, i] = np.array(mask_pil_array[i])

        #print(np.unique(mask), 'after1')
        image = (image / 255.0)
        mask[mask==0] = -1
        #print(np.unique(mask), 'after2')
        if self.pretrained:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = (image - mean)/std
        #return
        return image, mask

    def __getitem__(self, index):

        name = self.train_test_id.iloc[index].ID
        path = self.image_path
        # Load image and from h5
        image = load_image(os.path.join(path, '%s.h5' % name))
        mask = np.empty([image.shape[0], image.shape[1], len(self.attribute)], dtype='int')
        if self.mask_use:
            for i, attr in enumerate(self.attribute):
                mask[:, :, i] = load_image(os.path.join(path, '{}_{}.h5'.format(name, attr)))[:, :, 0]
        else:
            mask = image

        if self.train == 'train':
            if self.augment_list:
                image, mask = self.transform_fn(image, mask)
        """fig = plt.figure(figsize=(5, 5))
        ax = []
        for channel in range(mask.shape[2]):
            im = mask[:, :, channel]
            ax.append(fig.add_subplot(2, mask.shape[2], channel + 1))
            ax[channel].set_title(name[5:]+str(np.unique(im)))
            plt.imshow(im)
        print(np.unique(mask))"""
        if self.mask_use and not self.aux:
            p = np.random.uniform(0, 1)
            if self.train == 'valid':
                mask.fill(0.)
            elif self.cell:
                mask = quatro_mask_clear(mask, image.shape[0], self.cell_size, self.prob)
            else:
                mask = full_mask_clear(mask, self.prob)
            image_with_mask = np.dstack((image, mask))
        else:
            image_with_mask = image

        """for channel in range(mask.shape[2]):
            im = mask[:, :, channel]
            ax.append(fig.add_subplot(2, mask.shape[2], mask.shape[2] + channel + 1))
            ax[mask.shape[2] + channel].set_title(name[5:]+str(np.unique(im)))
            plt.imshow(im)
        plt.show()"""
        labels = np.array([self.train_test_id.loc[index, attr[10:]] for attr in self.attribute])
        print(labels, 'lab', index)
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
                return image_with_mask, np.array([labels for i in range(self.aux_batch)]), name
            else:
                mask.fill(0.)
                image_with_mask = np.dstack((image, mask))
        return image_with_mask, labels, name


def full_mask_clear(mask, prob):
    if np.random.uniform(0, 1) < prob:
        mask.fill(0.)
    return mask


def quatro_mask_clear(mask, shape, cell_size, prob):
    for i in range(0, shape, cell_size):
        for j in range(0, shape, cell_size):
            p = np.random.uniform(0, 1)
            if p < prob:
                mask[i:i + cell_size, j:j + cell_size, :].fill(0.)
    return mask


def make_loader(train_test_id, args, train=True, shuffle=True):

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

import os
import torchvision.transforms.functional as TF
from keras.preprocessing.image import array_to_img, img_to_array
from torch.utils.data import Dataset, DataLoader
from Utils.utils import load_image
import random
import numpy as np


class MyDataset(Dataset):

    def __init__(self, train_test_id, labels_ids, args, train):

        self.train_test_id = train_test_id
        self.labels_ids = labels_ids
        self.image_path = args.image_path
        self.pretrained = args.pretrained
        self.attribute = args.attribute
        self.mask_use = args.mask_use
        self.augment_list = args.augment_list
        self.train = train
        self.all_attributes = ['attribute_globules', 'attribute_milia_like_cyst', 'attribute_negative_network',
                               'attribute_pigment_network', 'attribute_streaks']

        self.indexes = [i for i, val in enumerate(self.all_attributes) for attr in self.attribute if attr == val]

        if train == 'train':
            self.labels_ids = self.labels_ids[self.train_test_id['Split'] == 'train'].values.astype('uint8')
            self.train_test_id = self.train_test_id[self.train_test_id['Split'] == 'train'].ID.values
            #print('Train =', self.train, 'train_test_id.shape: ', self.train_test_id.shape)
        elif train == 'valid':
            self.labels_ids = self.labels_ids[self.train_test_id['Split'] != 'train'].values.astype('uint8')
            self.train_test_id = self.train_test_id[self.train_test_id['Split'] != 'train'].ID.values
            #print('Train =', self.train, 'train_test_id.shape: ', self.train_test_id.shape)

        self.n = self.train_test_id.shape[0]

    def __len__(self):
        return self.n

    def transform_fn(self, image, mask):

        image = array_to_img(image, data_format="channels_last")
        mask_pil_array = [None] * mask.shape[-1]
        for i in range(mask.shape[-1]):
            mask_pil_array[i] = array_to_img(mask[:, :, i, np.newaxis], data_format="channels_last")
        if 'hflip' in self.augment_list:
            if random.random() > 0.5:
                image = TF.hflip(image)
                for i in range(mask.shape[-1]):
                    mask_pil_array[i] = TF.hflip(mask_pil_array[i])
        if 'vflip' in self.augment_list:
            if random.random() > 0.5:
                image = TF.vflip(image)
                for i in range(mask.shape[-1]):
                    mask_pil_array[i] = TF.vflip(mask_pil_array[i])
        if 'affine' in self.augment_list:
            angle = random.randint(0, 90)
            translate = (random.uniform(0, 100), random.uniform(0, 100))
            scale = random.uniform(0.5, 2)
            image = TF.affine(image, angle, translate, scale)
            for i in range(mask.shape[-1]):
                mask_pil_array[i] = TF.affine(mask_pil_array[i], angle, translate, scale)
        if 'adjust_brightness' in self.augment_list:
            if random.random() > 0.3:
                brightness_factor = random.uniform(0.8, 1.2)
                image = TF.adjust_brightness(image, brightness_factor)
        if 'adjust_saturation' in self.augment_list:
            if random.random() > 0.3:
                saturation_factor = random.uniform(0.8, 1.2)
                image = TF.adjust_saturation(image, saturation_factor)

        image = img_to_array(image, data_format="channels_last")
        if self.mask_use:
            for i in range(mask.shape[-1]):
                mask[:, :, i] = img_to_array(mask_pil_array[i], data_format="channels_last")[:, :, 0].astype('uint8')

        image = (image / 255.0).astype('float32')
        mask = (mask / 255.0).astype('uint8')

        if self.pretrained:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = (image - mean)/std

        return image, mask

    def __getitem__(self, index):

        name = self.train_test_id[index]
        path = self.image_path
        # Load image and from h5
        image = load_image(os.path.join(path, '%s.h5' % name), 'image')
        mask = np.empty([image.shape[0], image.shape[1], len(self.attribute)])
        if self.mask_use:
            for i, attr in enumerate(self.attribute):
                mask[:, :, i] = load_image(os.path.join(path, '{}_{}.h5'.format(name, attr)), 'mask')[:, :, 0]
        else:
            mask = image

        if self.train == 'train':
            if self.augment_list:
                image, mask = self.transform_fn(image, mask)

        if self.mask_use:
            p = np.random.uniform(0, 1)
            if self.train == 'valid' or p > 0.5:
                mask.fill(0.)
            image_with_mask = np.dstack((image, mask))
        else:
            image_with_mask = image

        labels = np.array([self.labels_ids[index, ind] for ind in self.indexes])

        return image_with_mask, labels, name


def make_loader(train_test_id, labels_ids, args, train=True, shuffle=True):

    data_set = MyDataset(train_test_id=train_test_id,
                         labels_ids=labels_ids,
                         args=args,
                         train=train)
    data_loader = DataLoader(data_set,
                             batch_size=args.batch_size,
                             shuffle=shuffle
                             )
    return data_loader

import numpy as np
import argparse
from joblib import Parallel, delayed
import os
from PIL import Image
import albumentations


def get_ind(img_name):
    return img_name.split('.')[0]


def prepare_data(args):
    img_names = os.listdir(args.impath)
    img_names = filter(lambda x: x.endswith('jpg'), img_names)
    img_inds = list(map(get_ind, img_names))
    results = Parallel(n_jobs=args.jobs)(delayed(load_image)(ind, row, args) for ind, row in enumerate(img_inds))


def apply_album_tfm(tfm:albumentations.DualTransform, img, mask):
    augmented = tfm(image=img, mask=mask)
    return augmented['image'], augmented['mask']


def _resize(img, mask):
    h, w, ch = img.shape
    min_h_w = min(h, w)
    crop_tfm = albumentations.CenterCrop(min_h_w, min_h_w)
    resize_tfm = albumentations.Resize(224, 224)
    composite = albumentations.Compose([crop_tfm, resize_tfm])
    return apply_album_tfm(composite, img, mask)


def load_image(ind, img_id, args):
    print(f'\r{ind}', end='')
    print(img_id)
    """ image loading """
    image_file = os.path.join(args.impath, f'{img_id}.jpg')
    img = np.array(Image.open(image_file))
    h, w, ch = img.shape
    """ masks loading"""
    attr_types = ['globules', 'milia_like_cyst', 'negative_network', 'pigment_network', 'streaks']
    mask = np.zeros((h, w, len(attr_types)), dtype=np.uint8)
    for i, attr in enumerate(attr_types):
        mask_file = os.path.join(args.maskpath, f'{img_id}_attribute_{attr}.png')
        m = np.array(Image.open(mask_file)).astype(np.uint8)
        mask[:, :, i] = m
    """ saving """
    img, mask = _resize(img, mask)
    np.save(os.path.join(args.svpath, 'images_npy', f'{img_id[5:]}.npy'), img)
    np.save(os.path.join(args.svpath, 'masks_npy', f'{img_id[5:]}.npy'), mask)
    return None


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--impath', type=str, default='./Data/skin_images/')
    arg('--maskpath', type=str, default='./Data/skin_masks/')
    arg('--svpath', type=str, default='./Data/preprocessed/')
    arg('--size', type=int, default=224)
    arg('--jobs', type=int, default=1)
    args = parser.parse_args()

    if not os.path.exists(args.svpath):
        os.mkdir(args.svpath)
    if not os.path.exists(os.path.join(args.svpath, 'images_npy')):
        os.mkdir(os.path.join(args.svpath, 'images_npy'))
    if not os.path.exists(os.path.join(args.svpath, 'masks_npy')):
        os.mkdir(os.path.join(args.svpath, 'masks_npy'))

    prepare_data(args)

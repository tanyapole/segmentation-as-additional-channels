import numpy as np
import argparse
import h5py
from joblib import Parallel, delayed
import os
from PIL import Image


def get_ind(img_name):
    return img_name.split('.')[0]


def prepare_data():

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--impath', type=str, default='./Data/skin_images/')
    arg('--maskpath', type=str, default='./Data/skin_masks/')
    arg('--svpath', type=str, default='./Data/h5/')
    arg('--size', type=int, default=224)
    arg('--jobs', type=int, default=1)
    args = parser.parse_args()

    if not os.path.exists(args.svpath):
        os.mkdir(args.svpath)

    img_names = os.listdir(args.impath)
    img_names = filter(lambda x: x.endswith('jpg'), img_names)

    img_inds = list(map(get_ind, img_names))

    results = Parallel(n_jobs=args.jobs)(delayed(load_image)(ind, row, args) for ind, row in enumerate(img_inds))


def load_image(ind, img_id, args):
    print(f'\r{ind}', end='')
    ###############
    print(img_id)
    ### load image
    image_file = args.impath + '%s.jpg' % img_id
    # img = load_img(image_file, target_size=(args.size, args.size), color_mode='rgb')  # this is a PIL image
    img_2 = Image.open(image_file)  # this is a PIL image
    img_2 = img_2.resize((args.size, args.size))

    #print(type(img_2),type(img_np))
    #assert(np.array_equal(np.array(img_2), img_np))
    #print(np.array(img_2)[0,:5])
    #print(img_np[0,:5])
    ### only 0-255 integers
    # img_np = img_np.astype(np.uint8)
    #print(np.unique(img_np))

    hdf5_file = h5py.File(args.svpath + '%s.h5' % img_id, 'w')
    hdf5_file.create_dataset('img', data=np.array(img_2))
    hdf5_file.close()
    ################

    attr_types = ['globules', 'milia_like_cyst', 'negative_network', 'pigment_network', 'streaks']

    for i, attr in enumerate(attr_types):
        mask_file = args.maskpath + '%s_attribute_%s.png' % (img_id, attr)
        m = Image.open(mask_file)  # this is a PIL image
        m = m.resize((args.size, args.size))
        #m_np = img_to_array(m)
        #m_np = (m_np / 255).astype('int8')
        m_np = np.array(m)
        m_np = (m_np / 255).astype('int8')
        m_np[m_np == 0] = -1
        #print(np.unique(m_np))
        hdf5_file = h5py.File(args.svpath + '%s_attribute_%s.h5' % (img_id, attr), 'w')
        hdf5_file.create_dataset('img', data=m_np, dtype=np.int8)
        hdf5_file.close()

    return None


if __name__ == '__main__':
    prepare_data()
import random

ALL_ATTRIBUTES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                  'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                  'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
                  'tvmonitor']

IMAGE_PATH = 'images_npy'
MASK_PATH = 'masks_npy'

K1 = 100
K2 = 200

TRAIN_TRAIN_NUMBER = 13000
TRAIN_VALID_NUMBER = 2000

PRETRAIN = 'pretrain'
YNET = 'ynet'
BASELINE = 'baseline'

r = random.Random(0)
SEED_LIST = [r.randint(1, 500) for _ in range(10)]

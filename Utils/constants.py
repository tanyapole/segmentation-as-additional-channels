import random

ALL_ATTRIBUTES = ['attribute_globules', 'attribute_milia_like_cyst', 'attribute_negative_network',
                  'attribute_pigment_network', 'attribute_streaks']

IMAGE_PATH = 'images_npy'
MASK_PATH = 'masks_npy'

K1 = 100
K2 = 200

TRAIN_TRAIN_NUMBER = 1600
TRAIN_VALID_NUMBER = 400

PRETRAIN = 'pretrain'
YNET = 'ynet'
BASELINE = 'baseline'

r = random.Random(0)
SEED_LIST = [r.randint(1, 500) for _ in range(10)]

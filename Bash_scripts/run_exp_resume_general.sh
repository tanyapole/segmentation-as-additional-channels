#!/bin/bash
python my_main.py --model vgg16 --N 1 --lr 0.0001 --batch_size 5 --mask_use --resume --augment_list hflip vflip affine adjust_brightness adjust_saturation --attribute attribute_milia_like_cyst attribute_pigment_network --cell_size 14 28 56 --prob 0.2 0.8  --n_epochs 1 --workers 5 --image_path ./$1 --model_path ./$2

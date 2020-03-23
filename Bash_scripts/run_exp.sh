#!/bin/bash
python my_main.py --model resnet50 --N 15 --pretrained --lr 0.01 0.001 0.0001 --batch_size 100 --augment_list hflip vflip affine adjust_brightness adjust_saturation --attribute attribute_pigment_network attribute_milia_like_cyst --freezing --n_epochs 45 --image_path ./$1

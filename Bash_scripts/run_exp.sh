#!/bin/bash
python my_main.py --model resnet50 --N 15 --pretrained --lr 0.01 0.001 0.0001 --batch-size 100 --augment-list hflip vflip affine adjust_brightness adjust_saturation --attribute attribute_pigment_network attribute_milia_like_cyst --freezing --n-epochs 45 --image-path ./$1

#!/bin/bash
python my_main.py --model resnet50 --N 10 --lr 0.0001 --batch_size 100 --augment_list hflip vflip affine adjust_brightness adjust_saturation --attribute attribute_globules attribute_milia_like_cyst attribute_negative_network attribute_pigment_network attribute_streaks  --n_epochs 400 --workers 5 --image_path ./$1 --model_path ./$2

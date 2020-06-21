#!/bin/bash
python main.py --model vgg16 --N 5 --lr 0.0001 --batch_size 5 --attribute attribute_globules attribute_milia_like_cyst attribute_negative_network attribute_pigment_network attribute_streaks  --n_epochs 200 --workers 5 --image_path ./$1

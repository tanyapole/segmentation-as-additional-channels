#!/bin/bash
python my_main.py --model vgg16 --pretrained --N 5 --lr 0.0001 --batch_size 10 --attribute attribute_milia_like_cyst attribute_pigment_network --n_epochs 200 --workers 5 --image_path D:/Data/h5/ --optimizer adam

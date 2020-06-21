#!/bin/bash
python Preprocessing/preprocess.py --impath ./$1 --maskpath ./$2 --svpath ./$3 --size $4 --jobs $5

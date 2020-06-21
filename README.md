# Pair loss
This method uses segmentation data to correct results of main NN training.

## Data preparation
my_preprocess.py module transforms original images and masks to h5py format

1. Download skin images and their masks from [ISIC2018 lession attribute detection challenge](https://challenge.kitware.com/#phase/5abcbb6256357d0139260e5f)
2. run with following parameters:
~~~~
python Preprocessing/preprocess --impath /path/to/images/ --maskpath /path/to/masks/ --svpath /path/to/save/ --size 224 --jobs 12
~~~~
## Experiments

### First experiment. Pair MSE loss train.
In this experiment model resnet50 is trained during 100 epochs 1 times using all image classes.
1. Prepare data as it stated in "Data preporation" section
2. To repeat this experiment run command:
~~~~
python main.py --model resnet50 --pretrained --N 1 --lr 0.0001 --batch_size 10 --attribute attribute_globules attribute_milia_like_cyst attribute_negative_network attribute_pigment_network attribute_streaks  --n_epochs 100 --workers 1 --optimizer adam --image_path /path/to/images/ --normalize --pair
~~~~

During experiments you can observe metrics and losses using tensorboard as: (curently disabled)
~~~~
tensorboard --logdir runs
~~~~
All hyperparameters and log during experiments are saving in the *'./runs/debug' folder*. <br>
Also all results will be stored as csv file in the *'./Results/<start_day&time_of_the_experiment>/results.csv'*

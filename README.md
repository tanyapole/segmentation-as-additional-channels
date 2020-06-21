# Zeroing_masks
Models in this branch use segmentation directly.

## Data preparation
my_preprocess.py module transforms original images and masks to h5py format

1. Download skin images and their masks from [ISIC2018 lession attribute detection challenge](https://challenge.kitware.com/#phase/5abcbb6256357d0139260e5f)
2. run with following parameters:
~~~~
python Preprocessing/preprocess --impath /path/to/images/ --maskpath /path/to/masks/ --svpath /path/to/save/ --size 224 --jobs 12
~~~~
## Experiments

### First experiment. Zeroing squares or whole masks.
In this experiment model resnet50 is trained during 200 epochs 1 times using all image classes. Weights are saved by best f1 score. Training is running with fixed different parameters of cell or not zeroing, different cell sizes & different probabilities.
1. Prepare data as it stated in "Data preporation" section
2. To repeat this experiment run:
~~~~
python --mask_use --N 1 --pretrained --lr 0.0001 --batch_size 10 --image_path /path/to/images/ --n_epochs 200 --prob 0.2 0.8 --attribute attribute_globules attribute_milia_like_cysts attribute_negative_network attribute_pigmented_network attribute_streaks --cell --cell_size 28 56 --workers 1 --optimizer adam --save_model --model_path /path/to/model/ --normalize
~~~~
### Second experiment. Big zeroing experiment
In this experiment model resnet50 is trained during 200 epochs 3 times using all image classes. Weights are saved by best f1 score. Training is running with fixed different parameters of cell or not zeroing, different cell sizes & different probabilities.
1. Prepare data as it stated in "Data preporation" section
2. To repeat this experiment run:
~~~~
python --mask_use --N 3 --pretrained --lr 0.0001 --batch_size 10 --image_path /path/to/images/ --n_epochs 200 --prob 0.5 0.6 0.7 0.8 0.9 --attribute attribute_globules attribute_milia_like_cysts attribute_negative_network attribute_pigmented_network attribute_streaks --cell --cell_size 7 14 28 56 --workers 1 --optimizer adam --save_model --model_path /path/to/model/ --normalize
~~~~
During experiments you can observe metrics and losses using tensorboard as: (curently disabled)
~~~~
tensorboard --logdir runs
~~~~
All hyperparameters and log during experiments are saving in the *'./runs/debug' folder*. <br>
Also all results will be stored as csv file in the *'./Results/<start_day&time_of_the_experiment>/results.csv'*

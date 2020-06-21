# Std loss
This method uses std loss with additional segmentation data.

## Data preparation
my_preprocess.py module transforms original images and masks to h5py format

1. Download skin images and their masks from [ISIC2018 lession attribute detection challenge](https://challenge.kitware.com/#phase/5abcbb6256357d0139260e5f)
2. run with following parameters:
~~~~
python Preprocessing/preprocess --impath /path/to/images/ --maskpath /path/to/masks/ --svpath /path/to/save/ --size 224 --jobs 12
~~~~
## Experiments

### First experiment. Aux retrain
In this experiment model resnet50 is trained during 200 epochs 1 times using all image classes. Model resnet50 modificated so after output_size==28 auxiliary exit is added. And for all outputs from this output auxiliary loss is computed as sum of std of each unique image in batch. 
1. Prepare data as it stated in "Data preporation" section
2. To repeat third experiment run:
~~~~
python main.py --mask_use --N 1 --pretrained --lr 0.0001 --batch_size 1 --image_path /path/to/images/ --n_epochs 100 --optimizer adam --normalize --aux --aux_batch 10 
~~~~

During experiments you can observe metrics and losses using tensorboard as: (curently disabled)
~~~~
tensorboard --logdir runs
~~~~
All hyperparameters and log during experiments are saving in the *'./runs/debug' folder*. <br>
Also all results will be stored as csv file in the *'./Results/<start_day&time_of_the_experiment>/results.csv'*

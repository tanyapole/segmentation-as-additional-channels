# Y-Net training
Multi-task learning with ynet-like architecture.

## Data preparation
my_preprocess.py module transforms original images and masks to h5py format

1. Download skin images and their masks from [ISIC2018 lession attribute detection challenge](https://challenge.kitware.com/#phase/5abcbb6256357d0139260e5f)
2. run with following parameters:
~~~~
python Preprocessing/preprocess.py --impath /path/to/images/ --maskpath /path/to/masks/ --svpath /path/to/save/ --size 224 --jobs 12
~~~~
## Experiments

### Experiment pretrain, ynet, baseline.
In this experiment model resnet50 is trained during 400 epochs 10 times using *2 image classes*: milia_like_cyst and pigment_network. After 200 epoch each time model weights are saved.
1. Prepare data as it stated in "Data preparation" section
2. To repeat first experiment run:
~~~~
python my_main.py --batch_size 10 --workers 2 --normalize --pretrained --N 5 --image_path /path/to/images/ --model_path /path/to/model/ 
~~~~

All hyperparameters and log during experiments are saving in the *'./runs/debug' folder*. <br>
Also all results will be stored as csv file in the *'./Results/<start_day&time_of_the_experiment>/results.csv'*

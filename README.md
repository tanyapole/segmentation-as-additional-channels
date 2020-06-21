# NNmethods
This branch contains base experiments with classic neural networks without using mask's information, such as VGG16 & ResNet50.

## Data preparation
my_preprocess.py module transforms original images and masks to h5py format

1. Download skin images and their masks from [ISIC2018 lession attribute detection challenge](https://challenge.kitware.com/#phase/5abcbb6256357d0139260e5f)
2. run with following parameters:
~~~~
python Preprocessing/preprocess --impath /path/to/images/ --maskpath /path/to/masks/ --svpath /path/to/save/ --size 224 --jobs 12
~~~~
## Experiments
python main.py --model resnet50 --pretrained --N 5 --lr 0.0001 --batch_size 5 --attribute attribute_globules attribute_milia_like_cyst attribute_negative_network attribute_pigment_network attribute_streaks  --n_epochs 100 --workers 1 --optimizer adam --image_path ./$1

### First experiment. 5 classes VGG16 train.
In this experiment model resnet50 is trained during 100 epochs 5 times using *5 image classes*: globules, milia_like_cyst, negative_network, pigment_network, streaks.
1. Prepare data as it stated in "Data preporation" section
2. To repeat this experiment run command:
~~~~
python main.py --model vgg16 --pretrained --N 5 --lr 0.0001 --batch_size 5 --attribute attribute_globules attribute_milia_like_cyst attribute_negative_network attribute_pigment_network attribute_streaks  --n_epochs 200 --workers 1 --optimizer adam --image_path /path/to/images/
~~~~

### Second experiment. 5 classes resnet50 train.
In this experiment model resnet50 is trained during 100 epochs 5 times using *5 image classes*: globules, milia_like_cyst, negative_network, pigment_network, streaks.
1. Prepare data as it stated in "Data preparation" section
2. To repeat this experiment run command:
~~~~
python main.py --model resnet50 --pretrained --N 5 --lr 0.0001 --batch_size 5 --attribute attribute_globules attribute_milia_like_cyst attribute_negative_network attribute_pigment_network attribute_streaks  --n_epochs 200 --workers 1 --optimizer adam --image_path /path/to/images/
~~~~


During experiments you can observe metrics and losses using tensorboard as: (curently disabled)
~~~~
tensorboard --logdir runs
~~~~
All hyperparameters and log during experiments are saving in the *'./runs/debug' folder*. <br>
Also all results will be stored as csv file in the *'./Results/<start_day&time_of_the_experiment>/results.csv'*

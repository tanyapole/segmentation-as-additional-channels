# segmentation-as-additional-channels
Using segmentationmasks as additional channels for classification

This algorithm compares neural networks in multi-classification task using segmentation masks as additional chanels with using just image information.

## Data preparation
my_preprocess.py module transforms original images and masks to h5py format

1. Download skin images and their masks from [ISIC2018 lession attribute detection challenge](https://challenge.kitware.com/#phase/5abcbb6256357d0139260e5f)
2. Run shell prompt from root of the project.
3. Execute command to make the script runnable for you
~~~~
chmod u+x bash_scripts/prepare.sh
~~~~
4. run prepare.sh with following parameters:
   1. */path/to/images/*
   2. */path/to/masks/*
   3. */path/to/save/*
   4. *resize_size*
   5. *number_of_parallel_jobs*
~~~~
. Bash_scripts/prepare.sh Data/skin_images/ Data/skin_masks/ Data/h5/ 224 12
~~~~
## Experiments

1. Prepare data as it stated in "Data preporation" section
2. Run shell prompt from root of the project.
3. Execute command to make the script runnable for you
~~~~
chmod u+x bash_scripts/run_exp.sh
~~~~
4. To repeat experiment run *run_exp.sh* with following parameters:
   1. *path/to/h5/files/
~~~~
. Bash_scripts/run_exp.sh Data/h5/
~~~~
During experiments you can observe metrics and losses using tensorboard as:
~~~~
tensorboard --logdir runs
~~~~
All hyperparameters and log during experiments are saving in the './runs/debug' folder.
Also all results will be stored as csv file in the './Results/<start_day&time_of_the_experiment>/results.csv'

## Tasks
- [x] Refactor code
- [x] Add bash scripts
- [x] Add readmi with explanations
- [ ] define thesis 

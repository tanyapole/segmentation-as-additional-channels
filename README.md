# segmentation-as-additional-channels
Using segmentationmasks as additional channels for classification

This algorithm compares vanila neural networks in multi-classification task using segmentation masks as additional chanels with using just image information.

## Data preparation
my_preprocess.py module transforms original images and masks to h5py format

1. Download skin images and their masks from [ISIC2018 lession attribute detection challenge](https://challenge.kitware.com/#phase/5abcbb6256357d0139260e5f)
2. Run shell prompt from root of the project.
3. Execute command to make the script runnable for you
~~~~
chmod u+x bash_scripts/prepare.sh
~~~~
4. run prepare sh with following parameters:
   1. */path/to/images/*
   2. */path/to/masks/*
   3. */path/to/save/*
   4. *resize_size*
   5. *number_of_parallel_jobs*
~~~~
. bash_scripts/prepare.sh /home/irek/My_work/train/data/ /home/irek/My_work/train/binary/ /home/irek/My_work/train/ttt/ 224 12
~~~~
## Experiments

1. Run shell prompt from root of the project.
2. Execute command to make the script runnable for you
~~~~
chmod u+x bash_scripts/run_exp.sh
~~~~
3. To repeat experiment run *run_exp.sh*
~~~~
. bash_scripts/run_exp.sh
~~~~

## Tasks
- [x] Refactor code
- [x] Add bash scripts
- [x] Add readmi with explanations
- [ ] define thesis 

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

### First experiment. 2 classes without masks train.
In this experiment model resnet50 is trained during 400 epochs 10 times using *2 image classes*: milia_like_cyst and pigment_network. After 200 epoch each time model weights are saved.
1. Prepare data as it stated in "Data preporation" section
2. Run shell prompt from root of the project.
3. Execute command to make the script runnable for you
~~~~
chmod u+x bash_scripts/run_exp_base_train_2_class.sh
~~~~
4. To repeat first experiment run *run_exp_base_train_2_class.sh* with following parameters:
   1. *path/to/h5/files/*
   2. *path/where/to/save/model/*
~~~~
. Bash_scripts/run_exp_base_train_2_class.sh Data/h5/ model/
~~~~
### Second experiment. Zeroing squares or whole masks.
In this experiment model resnet50 is retrained during 200 epochs 10 times using 2 image classes: milia_like_cyst and pigment_network. After 200 epoch each time model weights are saved. Model uses saved weights from first experiment, so you can repeat it or download from https://github.com/CaptainFest/Trained_models to */model_folder_name/* folder. Retraining is running with fixed different parameters of cell or not zeroing, different cell sizes & different probabilities.
1. Prepare data as it stated in "Data preporation" section
2. Run shell prompt from root of the project.
3. Execute command to make the script runnable for you
~~~~
chmod u+x bash_scripts/run_exp_resume_general.sh
~~~~
4. To repeat second experiment run *run_exp_resume_general.sh* with following parameters:
   1. *path/to/h5/files/*
   2. *path/to/saved/model/weights/*
~~~~
. Bash_scripts/run_exp_resume_general.sh Data/h5/ model/
~~~~
### Third experiment. Aux retrain
In this experiment model resnet50 is trained during 200 epochs 10 times using 2 image classes: milia_like_cyst and pigment_network. After 200 epoch each time model weights are saved. Model uses saved weights from first experiment, so you can repeat it or download from https://github.com/CaptainFest/Trained_models to */model_folder_name/* folder. Model resnet50 modificated so after output_size==28 auxiliary exit is added. And for all outputs from this exit auxiliary loss is computed as sum of std of each unique image in batch. 
1. Prepare data as it stated in "Data preporation" section
2. Run shell prompt from root of the project.
3. Execute command to make the script runnable for you
~~~~
chmod u+x bash_scripts/run_exp_resume_aux.sh
~~~~
4. To repeat third experiment run *run_exp_resume_aux.sh* with following parameters:
   1. *path/to/h5/files/*
   2. *path/to/saved/model/weights/*
~~~~
. Bash_scripts/run_exp_base_train_2_class.sh Data/h5/ model/
~~~~
### Forth experiment. All classes without masks train.
In this experiment model resnet50 is trained during 400 epochs 10 times using *ALL* image classes. After 200 epoch each time model weights are saved.
1. Prepare data as it stated in "Data preporation" section
2. Run shell prompt from root of the project.
3. Execute command to make the script runnable for you
~~~~
chmod u+x bash_scripts/run_exp_base_train_all.sh
~~~~
4. To repeat forth experiment run *run_exp_base_train_all.sh* with following parameters:
   1. *path/to/h5/files/*
   2. *path/where/to/save/model/*
~~~~
. Bash_scripts/run_exp_base_train_2_class.sh Data/h5/ model_all/
~~~~

During experiments you can observe metrics and losses using tensorboard as: (curently disabled)
~~~~
tensorboard --logdir runs
~~~~
All hyperparameters and log during experiments are saving in the *'./runs/debug' folder*. <br>
Also all results will be stored as csv file in the *'./Results/<start_day&time_of_the_experiment>/results.csv'*

## Tasks
- [x] Refactor code
- [x] Add bash scripts
- [x] Add readmi with explanations
- [ ] define thesis 

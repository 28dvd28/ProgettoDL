# Self-supervised Lerning for Endoscopic Video Analysis -- PyTorch implementation

This repository contains the translated code in PyTorch for the paper "Self-supervised Learning for Endoscopic Video 
Analysis" (https://arxiv.org/pdf/2308.12394v1). The original code is available in [this repository](https://github.com/royhirsch/endossl).

All the implementations details can be found in the code where each file it has been carefully documented. Here is
just described the main structure of the project and the folders.

## Environment

The code has been executed on a linux machine with _Ubuntu 20.04_. The execution was done in a 
anaconda environment with python 3.12.4. The main requirements are:

- torch 2.4.0 with cuda 12.1
- torchvision 0.19.0 
- torchmetrics 1.4.0.post0
- tensorboard 2.17.0
- tqdm 4.66.4
- transformers 4.43.3
- numpy 2.0.1

Make sure to use an environment with this dependencies installed.



## Preparation of the dataset

First you need to download the Cholec80 dataset, it can be done just executing the _prepare.py_ file that can
be fownd in the _data_ folder. For a correct execution you can run the following command

    python prepare.py --data_rootdir YOUR_LOCATION

where _YOUR_LOCATION_ is the location where you want to download the dataset. 
The script will download the dataset in a .zip file and will then extract it.

## Structure

### Data folder
In the data folder, other than the _prepare.py_ file, there is the _cholec80_images.py_ file
that implement the dataloader for the Cholec80 dataset. Its execution can be tested using

    python data/cholec80_images.py

be sure to have your terminal running in the endossl-main folder. Opening the file, at the beginning 
there is some commented line that must be uncommented based on which training you want to test. In the
code you will find all the instructions.

### Downstream folder

In the downstream folder there are the files that implement the training parts: _cholec80_classifier.py_ for the 
frame phase classifier, while _ViT_pretraining.py_ for the MSN pretraining. The execution of the files can be done with

    python downstream/<script>.py

before each execution be sure to create, in the *exps* direcotry, the folders where the checkpoints and training info will be saved, that can be
changed in the _Config_ class defined at the beginning of each of the above files, overwriting the
_exp_dir_ values. In the experiment directory must be present two folders, _checkpoints_ and _tb_logs_.
In the first one will be saved all the model checkpoints, while in the second the tensorboard logs.

If your directory is called for instance _dir_experiment1_, the structure should be like this:
    
    exps
    ├── dir_experiment1
    │   ├── checkpoints
    │   └── tb_logs
Make sure also to have your terminal in the endossl-main folder before launching the scripts.

For the _cholec80_classifier.py_ script, for the testing part you must uncomment the last line of the code,
while for using a pre-trained model, it must be corrected set the flag _pretrained_ in the Config class to
True and it must be set the path to the pre-trained model in the _pretrained_path_ and _model_name_ variables.

### Models folder
In the models folder there are the implementation of the models used. _MyViTMSN.py_ contains the definition class 
that implements the model for the classifier, while _MyViTMSN_pretraining.py_ contains the definition 
of the class that implements the model for the self-supervised training.

### Exps folder
After the download of the project folder, in the exps folder there are only the tensorboard logs of the 
pretraining part done using MSN, that is the main part of this project, that can be saw launching the command
    
        tensorboard --logdir=<path-to-exps>/pretraining/tb_logs

where _<path-to-exps>_ is the path to the exps folder in the endossl-main folder.

It is also possible to see the tensorboard logs of the classifier training that was done in the beginning
using a pre-trained model over the ImageNet-1k dataset. The logs shows a convergence of the model to
poor loss and F1 score values. The scores can be saw launching the command

        tensorboard --logdir=<path-to-exps>/cholec80_classifier/tb_logs

where _<path-to-exps>_ is the path to the exps folder in the endossl-main folder.

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import softmax

sys.path.append(os.path.realpath(__file__ + '/../../'))

from data import cholec80_images
from models.MyViTMSN import MyViTMSNModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Config:

    # experiment directory
    exp_dir = os.path.join('exps', 'pretraining')

    # dataset info
    dataset_name = 'cholec80'
    data_root = os.path.join('cholec80')

    # metrics
    task_type = 'multi_class'
    monitor_metric = 'val_macro_f1'

    # optimization
    optimize_name = 'adamw'
    learning_rate = 1e-4
    weight_decay = 1e-5

    # training
    num_epochs = 5
    batch_size = 150
    validation_freq = 1

def training_loop():


    datasets = cholec80_images.get_pytorch_dataloaders(
        data_root=Config.data_root,
        batch_size=Config.batch_size
    )



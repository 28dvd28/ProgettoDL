"""A compact module for running down-stream experiments."""
import glob
import sys
import os
from traceback import print_exc

from networkx.algorithms.bipartite.cluster import modes
from sympy.physics.vector import outer
from triton.ops.blocksparse import softmax

sys.path.append(os.path.realpath(__file__ + '/../../'))

import torch
import torch.nn as nn
from torch.nn.functional import softmax
from torchvision import models
from tqdm import tqdm

from data import cholec80_images
from train import eval_lib
from train import train_lib
from pretraining_ViT import MyViTMSNModel
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def verbose_print(msg, verbose=True, is_title=False):
    if verbose:
        print(msg)
    if is_title:
        print('#' * 40)


def run_experiment(config, verbose=True):
    """Stand-alone function for running an experiment."""

    verbose_print('Config:', verbose)
    attr_names = [i for i in dir(config) if not i.startswith('__')]
    for a in attr_names:
        verbose_print('{} = {}'.format(a, getattr(config, a, None)), verbose)
    print('\n\n')

    if not os.path.exists(config.exp_dir):
        os.makedirs(config.exp_dir)

    ##############################################################################
    # Datasets & Model
    ##############################################################################
    verbose_print('Create datasets and model', verbose, True)
    datasets = cholec80_images.get_pytorch_dataloaders(
        data_root=config.data_root,
        batch_size=config.batch_size,
        train_transformation=config.train_transformation,
    )

    if config.is_linear_evaluation:
        model = train_lib.get_linear_model(
            input_dim=config.input_dim, output_dim=config.num_classes
        )
    elif config.model == 'resnet50':
        # TODO (1): controllare se c'è bisogno di cambiare o aggiungere il Global Average Pooling 2D
        # TODO (2): se il training viene fatto sull'intero modello resnet50 o solo sull'ultimo layer
        model = models.resnet50()
        model.fc = nn.Linear(model.fc.in_features, config.num_classes)
    elif 'vit' in config.model:
        model = MyViTMSNModel()
        # model.load_state_dict(torch.load(config.saved_model_dir))
        for param in model.vitMsn.parameters():
            param.requires_grad = False
    else:
        raise ValueError('Invalid model name: {}'.format(config.model))

    model = model.to(device)
    optimizer = train_lib.get_optimizer(
        model,
        config.optimize_name,
        config.learning_rate,
        config.momentum,
        config.weight_decay,
    )
    loss = train_lib.get_loss(config.task_type)
    saved_model_metrics = []
    new_model_metrics = train_lib.get_metrics(config.task_type, config.num_classes)
    callbacks = train_lib.get_callbacks(config.callbacks_names, optimizer, config.exp_dir, config.monitor_metric, config.learning_rate)

    ##############################################################################
    # Train
    ##############################################################################
    history = {'train_loss': [], 'val_loss': [], 'val_metrics': []}
    verbose_print('Begin training', verbose, True)
    for epoch in range(config.num_epochs):

        # training for the fist epoch
        model.train()
        running_loss = 0.0
        bar = tqdm(datasets['train'], total=len(datasets['train']), desc=f'Train of epoch: {epoch}')
        for inputs, labels in bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = softmax(model(inputs), dim=1)
            loss_value = loss(outputs, labels)
            loss_value.backward()
            optimizer.step()
            running_loss += loss_value.item()

            bar.set_postfix(loss=loss_value.item() / (bar.n + 1))
        bar.close()

        epoch_train_loss = running_loss / len(datasets['train'])

        for metric in new_model_metrics:
            metric.to(device)

        # validation
        if epoch % config.validation_freq == 0:
            model.eval()
            running_loss = 0.0
            for metric in new_model_metrics:
                metric.reset_val()
            with torch.no_grad():
                bar_eval = tqdm(datasets['validation'], total=len(datasets['validation']), desc=f'Test of epoch: {epoch}')
                for inputs, labels in bar_eval:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss_value = loss(outputs, labels)
                    running_loss += loss_value.item()
                    for metric in new_model_metrics:
                        metric.update_val(outputs, labels)
                    bar_eval.set_postfix(loss=loss_value.item() / (bar_eval.n + 1))

            bar_eval.close()
            epoch_val_loss = running_loss / len(datasets['validation'])

            if 'checkpoints' in config.callbacks_names:
                # save the model if the validation metric is better than the previous one
                print('-->Saving checkpoints')
                for i in range(len(new_model_metrics)):
                    callbacks['checkpoints'].on_save_checkpoint(new_model_metrics[i], model)

            if 'reduce_lr_plateau' in config.callbacks_names:
                print('-->Check reduction on plateau')
                callbacks['reduce_lr_plateau'].step(epoch_val_loss)

            if 'early_stopping' in config.callbacks_names:
                print('-->Check early stopping')
                callbacks['early_stopping'].step(epoch_val_loss)
                if callbacks['early_stopping'].end_patience():
                    break

            if 'tensorboard' in config.callbacks_names:
                print('-->Saving tensorboard validation')
                callbacks['tensorboard'].add_scalar('Validation/Loss', epoch_val_loss, epoch)
                for metric in new_model_metrics:
                    callbacks['tensorboard'].add_scalar(f'Validation/{metric.name}', metric.value, epoch)

        if 'step_scheduler' in config.callbacks_names:
            print('-->Step scheduler')
            callbacks['step_scheduler'].step()
        if 'tensorboard' in config.callbacks_names:
            print('-->Saving tensorboard train')
            callbacks['tensorboard'].add_scalar('Train/Loss', epoch_train_loss, epoch)

        # creation of the history
        history['train_loss'].append(epoch_train_loss)
        if 'epoch_val_loss' not in locals():
            epoch_val_loss = 0
        history['val_loss'].append(epoch_val_loss)

        mod_metric = []
        for metric in new_model_metrics:
            mod_metric.append(metric.value)
        history['val_metrics'].append(mod_metric)

    if 'tensorboard' in config.callbacks_names:
        callbacks['tensorboard'].flush()
        callbacks['tensorboard'].close()

    #############################################################################
    # End of train evaluation
    #############################################################################
    if config.manually_load_best_checkpoint:
        checkpoints_dir = os.path.join(config.exp_dir, 'checkpoints')
        checkpoints = glob.glob(os.path.join(checkpoints_dir, '*'))
        if checkpoints:
            latest = checkpoints[-1]
            print(f'Load latest checkpoints: {latest}')
            model.load_state_dict(torch.load(latest))
        else:
            print('Haven\'t loaded a saved checkpoints')

    # For the special case of phases, re-extract the dataset with the 'with_image_path'
    # attribute for calculating video-level metrics
    verbose_print('Start end of train evaluation', verbose, True)
    datasets = cholec80_images.get_pytorch_dataloaders(
        data_root=config.data_root,
        batch_size=config.batch_size,
        train_transformation=config.train_transformation,
        with_image_path=True,
    )

    mets = eval_lib.end_of_training_evaluation(
        model,
        datasets['validation'],
        datasets['test'],
        label_key=config.label_key,
        exp_dir=config.exp_dir,
        epoch=config.num_epochs)
    return mets, history

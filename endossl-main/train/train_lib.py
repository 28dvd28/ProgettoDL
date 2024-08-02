"""Helper training functions."""
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from torcheval.metrics import MultilabelAUPRC
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from torch.utils.tensorboard import SummaryWriter


from abc import ABC, abstractmethod


def cross_entropy(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return_val = torch.tensor(0).to(target.device).to(target.dtype)
    if target.shape[0] == 1:
        return - target.dot(torch.log(prediction))
    for i in range(target.shape[0]):
        return_val += target[i].dot(torch.log(prediction[i]))
    return -return_val / target.shape[0]


def get_linear_model(input_dim: int, output_dim: int):
    return nn.Sequential(
        nn.Flatten(start_dim=1),
        nn.Linear(10000, 5)
    )


class LinearFineTuneModel(nn.Module):
    def __init__(self, backbone, output_dim):
        super(LinearFineTuneModel, self).__init__()
        self.backbone = backbone
        self.projection = nn.Linear(list(backbone.modules())[-1].out_features, output_dim)

    def forward(self, x):
        return self.projection(self.backbone(x)[1])


def get_loss(task_type: str):
    if task_type == 'multi_class':
        return cross_entropy
    elif task_type == 'multi_label':
        return nn.BCEWithLogitsLoss()
    else:
        raise ValueError


def get_optimizer(
        neural_net: nn.Module,
        optimizer_name: str,
        learning_rate: float = 0.001,
        momentum: float = 0.9,
        weight_decay: float = 0.00001) -> optim:
    """Initialize optimizer by its name."""

    optimizer_name = optimizer_name.lower()
    if optimizer_name == 'adam':
        return optim.Adam(neural_net.parameters(), lr=learning_rate)
    elif optimizer_name == 'adamw':
        return optim.AdamW(neural_net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'rmsprop':
        return optim.RMSprop(neural_net.parameters(), lr=learning_rate, momentum=momentum)
    elif optimizer_name == 'momentum':
        return optim.SGD(
            neural_net.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        return optim.SGD(
            neural_net.parameters(),
            lr=learning_rate,
            momentum=0,
            weight_decay=weight_decay)
    else:
        raise ValueError('Optimizer %s not supported' % optimizer_name)


class MyMetrics(ABC):

    def __init__(self, name):
        self.name = name
        self.value = 0
        self.counter = 0

    @abstractmethod
    def update_val(self, predictions, ground_truths):
        pass

    def reset_val(self):
        self.value = 0
        self.counter = 0


class MyF1Score(MyMetrics):
    def __init__(self, num_classes, average, name):
        super(MyF1Score, self).__init__(name)
        self.f1_score = torchmetrics.F1Score(task='multiclass', num_classes=num_classes, average=average)

    def update_val(self, predictions, ground_truths):
        self.counter += 1
        self.value = (self.value * (self.counter - 1))
        self.value += self.f1_score(predictions, ground_truths).item()
        self.value = self.value / self.counter

    def to(self, device):
        self.f1_score = self.f1_score.to(device)


class MyAccuracy(MyMetrics):

    def __init__(self, num_classes, name):
        super(MyAccuracy, self).__init__(name)
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes, average='macro', top_k=2)

    def update_val(self, predictions, ground_truths):
        self.counter += 1
        self.value = (self.value * (self.counter - 1))
        self.value = self.accuracy(predictions, ground_truths).item()
        self.value = self.value / self.counter

    def to(self, device):
        self.accuracy = self.accuracy.to(device)


class MyPrecision(MyMetrics):

    def __init__(self, name):
        super(MyPrecision, self).__init__(name)
        self.precision = torchmetrics.Precision(task='multilabel')

    def update_val(self, predictions, ground_truths):
        self.counter += 1
        self.value = (self.value * (self.counter - 1))
        self.value = self.precision(predictions, ground_truths).item()
        self.value = self.value / self.counter

    def to(self, device):
        self.precision = self.precision.to(device)


class MyAUC(MyMetrics):

    def __init__(self, num_labels, name):
        super(MyAUC, self).__init__(name)
        self.auc = MultilabelAUPRC(num_labels=num_labels)

    def update_val(self, predictions, ground_truths):
        self.auc.update(predictions, ground_truths)
        self.counter += 1
        self.value = (self.value * (self.counter - 1))
        self.value = self.auc.compute()
        self.value = self.value / self.counter
        self.auc.reset()

    def to(self, device):
        self.auc = self.auc.to(device)


class MyIntersectionOverUnion(MyMetrics):

    def __init__(self, name):
        super(MyIntersectionOverUnion, self).__init__(name)

    def update_val(self, predictions, ground_truths):

        self.counter += 1
        if predictions[0] < ground_truths[0] and predictions[0] + predictions[2] < ground_truths[0]:
            self.value += 0
        elif predictions[1] < ground_truths[1] and predictions[1] + predictions[3] < ground_truths[1]:
            self.value += 0
        elif ground_truths[0] < predictions[0] and ground_truths[0] + ground_truths[2] < predictions[0]:
            self.value += 0
        elif ground_truths[1] < predictions[1] and ground_truths[1] + ground_truths[3] < predictions[1]:
            self.value += 0
        else:
            pred_area = predictions[2] * predictions[3]
            gt_area = ground_truths[2] * ground_truths[3]
            union_area = pred_area + gt_area

            xA = max(predictions[0], ground_truths[0])
            yA = max(predictions[1], ground_truths[1])
            xB = min(predictions[0] + predictions[2], ground_truths[0] + ground_truths[2])
            yB = min(predictions[1] + predictions[3], ground_truths[1] + ground_truths[3])
            intersection_area = abs(xA - xB) * abs(yA - yB)

            self.value += intersection_area / union_area
            self.value = self.value / self.counter



def get_metrics(task_type: str, num_classes: int):
    if task_type == 'multi_class':
        return [MyF1Score(num_classes=num_classes,
                          average='micro',
                          name='micro_f1'),
                MyF1Score(num_classes=num_classes,
                          average='macro',
                          name='macro_f1'),
                MyAccuracy(num_classes=num_classes, name='accuracy')
                ]
    elif task_type == 'multi_label':
        return [MyPrecision(name='precision'),
                MyAUC(num_labels=num_classes, name='pr_auc')]
    else:
        raise ValueError


class EarlyStopping:
    def __init__(self, monitor, patience, mode):
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.wait_count = 0

        self.value = None

    def step(self, loss):
        if self.value is None:
            self.value = loss
        elif (self.mode == 'min' and loss < self.value) or (self.mode == 'max' and loss > self.value):
            self.value = loss
            self.wait_count = 0
        else:
            self.wait_count += 1

    def end_patience(self):
        return self.wait_count > self.patience



def get_early_stopping_callback(
        monitor_metric='val_accuracy',
        patience=5,
        mode='min'):
    return EarlyStopping(
        monitor=monitor_metric,
        patience=patience,
        mode=mode,
    )


class ModelCheckpoint:
    def __init__(self, dirpath, filename, save_best_only, monitor_metric, mode):
        self.dirpath = dirpath
        self.filename = filename
        if not filename.endswith('.pth'):
            self.filename += '.pth'
        self.save_best_only = save_best_only
        self.monitor_metric = monitor_metric
        self.mode = mode
        self.progressive_number = 1

        self.best_model_score = None
        self.best_model_path = None

    def on_save_checkpoint(self, metric, model):
        if 'val_' + metric.name == self.monitor_metric:
            if (self.best_model_score is None
                    or (metric.value > self.best_model_score and self.mode == 'max')
                    or (metric.value < self.best_model_score and self.mode == 'min')):
                self.best_model_score = metric.value

                if self.save_best_only:
                    self.best_model_path = os.path.join(self.dirpath, self.filename.format(epoch=1))
                else:
                    self.best_model_path = os.path.join(self.dirpath, self.filename.format(epoch=self.progressive_number))
                    self.progressive_number += 1

                self.best_model_path = os.path.abspath(self.best_model_path)

                torch.save(model.state_dict(), self.best_model_path)


def get_checkpoint_callback(
        exp_dir,
        save_best_only=True,
        monitor_metric='val_accuracy',
        mode='min'
):
    checkpoint_callback = ModelCheckpoint(
        dirpath=exp_dir + '/checkpoints',
        filename='epoch_{epoch:02d}',
        save_best_only=save_best_only,
        monitor_metric=monitor_metric,
        mode=mode
    )

    return checkpoint_callback


def get_tensorboard_callback(exp_dir):
    log_dir = os.path.join(exp_dir, "tb_logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)
    return writer


def get_reduce_lr_plateau_callback(
        optimizer,
        factor=0.3,
        patience=10,
        mode='min',
        min_lr=1e-5,
):
    return ReduceLROnPlateau(
        optimizer=optimizer,
        factor=factor,
        patience=patience,
        mode=mode,
        min_lr=min_lr
    )


def get_learning_rate_step_scheduler_callback(
        optimizer,
        factor=0.3,
        milestones=[30]
):
    return MultiStepLR(
        optimizer=optimizer,
        gamma=factor,
        milestones=milestones
    )


def get_callbacks(callbacks_names, optimizer, exp_dir, monitor_metric, gamma):
    callbacks = {}
    if 'checkpoints' in callbacks_names:
        callbacks['checkpoints'] = get_checkpoint_callback(exp_dir, monitor_metric=monitor_metric)
    if 'reduce_lr_plateau' in callbacks_names:
        callbacks['reduce_lr_plateau'] = get_reduce_lr_plateau_callback(optimizer, factor=gamma)
    if 'step_scheduler' in callbacks_names:
        callbacks['step_scheduler'] = get_learning_rate_step_scheduler_callback(
            optimizer=optimizer, factor=gamma)
    if 'early_stopping' in callbacks_names:
        callbacks['early_stopping'] = get_early_stopping_callback()
    if 'tensorboard' in callbacks_names:
        callbacks['tensorboard'] = get_tensorboard_callback(exp_dir)
    return callbacks

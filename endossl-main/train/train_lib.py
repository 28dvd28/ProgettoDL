"""Helper training functions."""

import torch.nn as nn
import torch.optim as optim
import torchmetrics
from torcheval.metrics import MultilabelAUPRC
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from abc import ABC, abstractmethod


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
        return nn.CrossEntropyLoss()
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

    @abstractmethod
    def update_val(self, predictions, ground_truths):
        pass

    def reset_val(self):
        self.value = 0


class MyF1Score(MyMetrics):
    def __init__(self, num_classes, average, name):
        super(MyF1Score, self).__init__(name)
        self.f1_score = torchmetrics.F1Score(task='multiclass', num_classes=num_classes, average=average)

    def update_val(self, predictions, ground_truths):
        self.value = self.f1_score(predictions, ground_truths).item()

    def to(self, device):
        self.f1_score = self.f1_score.to(device)


class MyAccuracy(MyMetrics):

    def __init__(self, num_classes, name):
        super(MyAccuracy, self).__init__(name)
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes, average='macro', top_k=2)

    def update_val(self, predictions, ground_truths):
        self.value = self.accuracy(predictions, ground_truths).item()

    def to(self, device):
        self.accuracy = self.accuracy.to(device)


class MyPrecision(MyMetrics):

    def __init__(self, name):
        super(MyPrecision, self).__init__(name)
        self.precision = torchmetrics.Precision(task='multilabel')

    def update_val(self, predictions, ground_truths):
        self.value = self.precision(predictions, ground_truths).item()

    def to(self, device):
        self.precision = self.precision.to(device)


class MyAUC(MyMetrics):

    def __init__(self, num_labels, name):
        super(MyAUC, self).__init__(name)
        self.auc = MultilabelAUPRC(num_labels=num_labels)

    def update_val(self, predictions, ground_truths):
        self.auc.update(predictions, ground_truths)
        self.value = self.auc.compute()
        self.auc.reset()

    def to(self, device):
        self.auc = self.auc.to(device)

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


def get_early_stopping_callback(
        monitor_metric='val_loss',
        patience=5,
        verbose=True,
        mode='auto'):
    return EarlyStopping(
        monitor=monitor_metric,
        patience=patience,
        verbose=verbose,
        mode=mode,
    )


def get_checkpoint_callback(
        exp_dir,
        monitor='val_loss',
        verbose=True,
        save_best_only=True,
        save_weights_only=False,
        mode='auto',
        save_freq='epoch',
):
    checkpoint_callback = ModelCheckpoint(
        dirpath=exp_dir + '/checkpoints',
        filename='epoch_{epoch:02d}',
        monitor=monitor,
        save_top_k=1 if save_best_only else -1,
        save_weights_only=save_weights_only,
        mode=mode,
        every_n_epochs=1 if save_freq == 'epoch' else None,
        verbose=verbose
    )
    return checkpoint_callback


def get_tensorboard_callback(exp_dir):
    logger = TensorBoardLogger(
        save_dir=exp_dir,
        name='logs',
        log_graph=False,
        log_every_n_steps=50
    )
    return logger


class ReduceLROnPlateauCallback(pl.Callback):
    def __init__(self, monitor='val_loss', factor=0.3, patience=10, verbose=1, mode='auto', min_lr=1e-5):
        self.scheduler = None
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.verbose = verbose
        self.mode = mode
        self.min_lr = min_lr

    def on_fit_start(self, trainer, pl_module):
        self.scheduler = ReduceLROnPlateau(
            trainer.optimizers[0],
            mode=self.mode,
            factor=self.factor,
            patience=self.patience,
            min_lr=self.min_lr
        )

    def on_validation_end(self, trainer, pl_module):
        logs = trainer.callback_metrics
        current = logs.get(self.monitor)
        if current is None:
            if self.verbose:
                print(f"ReduceLROnPlateau: {self.monitor} is not available.")
            return

        self.scheduler.step(current)


def get_reduce_lr_plateau_callback(
        monitor='val_loss',
        factor=0.3,
        patience=10,
        verbose=1,
        mode='auto',
        min_lr=1e-5,
):
    return ReduceLROnPlateauCallback(
        monitor=monitor,
        factor=factor,
        patience=patience,
        verbose=verbose,
        mode=mode,
        min_lr=min_lr
    )


class LearningRateStepSchedulerCallback(pl.Callback):
    def __init__(self, learning_rate=1e-4, factor=0.3, milestones=[30], verbose=1):
        self.scheduler = None
        self.learning_rate = learning_rate
        self.factor = factor
        self.milestones = milestones
        self.verbose = verbose

    def on_fit_start(self, trainer, pl_module):
        optimizer = optim.Adam(pl_module.parameters(), lr=self.learning_rate)
        self.scheduler = MultiStepLR(
            optimizer,
            milestones=self.milestones,
            gamma=self.factor
        )
        trainer.optimizers = [optimizer]
        trainer.lr_schedulers = [self.scheduler]

    def on_epoch_end(self):
        self.scheduler.step()


def get_learning_rate_step_scheduler_callback(
        learning_rate=1e-4,
        factor=0.3,
        milestones=[30],
        verbose=1,
):
    return LearningRateStepSchedulerCallback(
        learning_rate=learning_rate,
        factor=factor,
        milestones=milestones,
        verbose=verbose
    )


def get_callbacks(callbacks_names, exp_dir, monitor_metric, learning_rate):
    callbacks = []
    if 'checkpoint' in callbacks_names:
        callbacks.append(get_checkpoint_callback(exp_dir, monitor_metric))
    if 'reduce_lr_plateau' in callbacks_names:
        callbacks.append(get_reduce_lr_plateau_callback(monitor_metric))
    if 'step_scheduler' in callbacks_names:
        callbacks.append(get_learning_rate_step_scheduler_callback(
            learning_rate=learning_rate))
    if 'early_stopping' in callbacks_names:
        callbacks.append(get_early_stopping_callback())
    if 'tensorboard' in callbacks_names:
        callbacks.append(get_tensorboard_callback(exp_dir))
    return callbacks

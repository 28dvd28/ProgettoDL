import sys
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.classification import MulticlassF1Score
from torch.nn.functional import softmax
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from tqdm import tqdm

sys.path.append(os.path.realpath(__file__ + '/../../'))

from data import cholec80_images
from models.MyViTMSN import MyViTMSNModel
from models.MyViTMSN_pretraining import MyViTMSNModel_pretraining


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Config:

    # experiment directory
    exp_dir = os.path.join('exps', 'cholec80_classifier_pretrained')

    # model
    model = 'vit'
    pretrained = False
    model_name = 'model_29.pth'
    pretrained_path = os.path.join('exps', 'pretraining', 'checkpoints', model_name)

    # dataset info
    dataset_name = 'cholec80'
    data_root = os.path.join('cholec80')

    # metrics
    task_type = 'multi_class'
    monitor_metric = 'val_macro_f1'

    # optimization
    optimize_name = 'adam'
    learning_rate =  1e-4
    weight_decay = 1e-5

    # training
    num_epochs = 20
    num_classes = 7
    batch_size = 150
    validation_freq = 1

def train_loop():

    datasets = cholec80_images.get_pytorch_dataloaders(
        data_root=Config.data_root,
        batch_size=Config.batch_size
    )

    if Config.model == 'resnet50':
        model = models.resnet50()
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, Config.num_classes)
    elif 'vit' == Config.model:
        model = MyViTMSNModel(device=device)

        if Config.pretrained:
            pretrained_model = MyViTMSNModel_pretraining(ipe=len(datasets['train']), num_epochs=Config.num_epochs, device=device)
            model_path = Config.pretrained_path
            pretrained_model.load_state_dict(torch.load(model_path))

            model.vitMsn.load_state_dict(pretrained_model.vitMsn_anchor.state_dict())

        model.classifier = nn.Linear(model.classifier.in_features, Config.num_classes)
        for param in model.vitMsn.parameters():
            param.requires_grad = False
    else:
        raise ValueError('Invalid model name: {}'.format(Config.model))

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)
    criterion = nn.CrossEntropyLoss()
    metric_f1 = MulticlassF1Score(num_classes=Config.num_classes, average='macro').to(device)
    writer = SummaryWriter(log_dir=os.path.join(Config.exp_dir, 'tb_logs'))

    for epoch in range(Config.num_epochs):

        '''Train loop'''
        running_train_loss = 0.0
        bar = tqdm(total=len(datasets['train']), desc=f'Train of epoch {epoch + 1}', ncols=100)
        model.train()

        for i, (inputs, labels) in enumerate(datasets['train'], 0):
            inputs, labels = inputs.to(device).to(torch.float), labels.to(device)

            optimizer.zero_grad()
            output = model(inputs)
            output = softmax(output, dim=1)

            loss_value = criterion(output, labels)
            loss_value.backward()
            optimizer.step()
            running_train_loss += loss_value.item()

            writer.add_scalar(f'TrainLoop/epoch_{epoch}_loss', loss_value.item(), i)
            bar.set_postfix(loss=f'{loss_value.item()}')
            bar.update(1)

        bar.close()
        epoch_train_loss = running_train_loss / len(datasets['train'])


        '''Validation loop'''
        running_macroF1_score = 0.0
        bar = tqdm(total=len(datasets['validation']), desc=f'Validation of epoch {epoch + 1}', ncols=100)
        model.eval()

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(datasets['validation'], 0):
                optimizer.zero_grad()
                inputs, labels = inputs.to(device), labels.to(device)

                output = softmax(model(inputs), dim=1)

                metric_value = metric_f1(output, labels)
                running_macroF1_score += metric_value.item()

                writer.add_scalar(f'ValidationLoop/epoch_{epoch}_macrof1score', metric_value.item(), i)
                bar.set_postfix(f1score=f'{metric_value.item()}')
                bar.update(1)

            bar.close()

        epoch_macroF1_score = running_macroF1_score / len(datasets['validation'])

        writer.add_scalar(f'Averaged losses for epoch/train', epoch_train_loss, epoch)
        writer.add_scalar(f'Averaged macro f1 score for epoch/validation', epoch_macroF1_score, epoch)

        torch.save(model.state_dict(), os.path.join(Config.exp_dir, 'checkpoints', f'model_{epoch}.pth'))

        filename = os.path.join(Config.exp_dir, 'checkpoints', 'models_details.txt')
        with open(filename, 'a') as file:
            concatenated_string = f'Epoch: {epoch} - Train loss: {epoch_train_loss} - Macro f1 score: {epoch_macroF1_score}\n'
            file.write(concatenated_string)

    writer.flush()
    writer.close()


def test_loop(model_path : str):
    """Function for executing a test loop over the cholec80 dataset.
    The results will be saved in tensorflow in the experiment direcotry that can be found in the Config class at the
    top of the current file.

    The metric that has been used and that it is monitored is the multiclass macro F1 score.

    Args:
        model_path: a string containing the path for the trained model over which execute the testing part
    """

    datasets = cholec80_images.get_pytorch_dataloaders(
        data_root=Config.data_root,
        batch_size=Config.batch_size
    )

    model = MyViTMSNModel(device=device)
    model.classifier = nn.Linear(model.classifier.in_features, Config.num_classes)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    metric_f1 = MulticlassF1Score(num_classes=Config.num_classes, average='macro').to(device)
    writer = SummaryWriter(log_dir=os.path.join(Config.exp_dir, 'tb_logs'))

    running_test_mascrof1 = 0.0
    bar = tqdm(total=len(datasets['test']), desc=f'Test', ncols=100)

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(datasets['test'], 0):

            inputs, labels = inputs.to(device), labels.to(device)
            outputs = softmax(model(inputs), dim=1)

            f1score = metric_f1(outputs, labels)
            running_test_mascrof1 += f1score.item()

            writer.add_scalar(f'TestLoop/MacroF1score', f1score, i)
            bar.set_postfix(loss=f1score.item())
            bar.update(1)

    bar.close()
    writer.add_scalar(f'TestLoop/FinaMacroF1', running_test_mascrof1 / len(datasets["test"]), 1)
    print(f'Average MacroF1 for test: {running_test_mascrof1 / len(datasets["test"])}')

    writer.flush()
    writer.close()


if __name__ == '__main__':
    train_loop()

    # example of usage of the test loop function
    # test_loop('exps/cholec80_classifier/checkpoints/model_19.pth')

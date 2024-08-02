import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import softmax
from tqdm import tqdm

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


def mask_generator(batch_size : int, patch_numbers : int)->list:
    """ Generate a mask for the input tensor, such that half of the patches will be masked.

    Args:
        batch_size (int): The size of the batch
        patch_numbers (int): The number of patches in the input tensor
    """

    mask = torch.zeros(batch_size, patch_numbers)
    for i in range(batch_size):
        idx = torch.randperm(patch_numbers)[:patch_numbers // 2]
        mask[i, idx] = 1
    mask = mask == 1
    return mask

def training_loop():

    datasets = cholec80_images.get_pytorch_dataloaders(
        data_root=Config.data_root,
        batch_size=Config.batch_size,
        double_img=True
    )

    model = MyViTMSNModel()
    model.to(device)

    patch_size = model.vitMsn.config.patch_size
    patch_numbers = (224 * 224) // (patch_size * patch_size)

    optimizer = optim.AdamW(model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter(log_dir=os.path.join(Config.exp_dir, 'tb_logs'))

    for epoch in range(Config.num_epochs):

        '''Train loop'''
        running_train_loss = 0.0
        bar = tqdm(datasets['train'], total=len(datasets['train']), desc=f'Train of epoch {epoch+1}', ncols=100)
        model.train()
        i=0

        for inputs_target, inputs_anchor, _ in bar:

            optimizer.zero_grad()
            inputs_anchor, inputs_target = inputs_anchor.to(device), inputs_target.to(device)
            mask = mask_generator(inputs_target.shape[0], patch_numbers)

            output_target = softmax(model(inputs_target), dim=1)
            output_anchor = softmax(model(inputs_anchor, bool_masked_pos=mask), dim=1)

            loss_value = criterion(output_anchor, output_target)
            loss_value.backward()
            optimizer.step()
            running_train_loss += loss_value.item()

            writer.add_scalar(f'TrainLoop/epoch_{epoch}_loss', loss_value,i)
            bar.set_postfix(loss=loss_value.item())
            i += 1

        bar.close()
        epoch_train_loss = running_train_loss / len(datasets['train'])


        ''' Validation loop'''
        running_validation_loss = 0.0
        bar = tqdm(datasets['validation'], total=len(datasets['validation']), desc=f'Train of epoch {epoch + 1}', ncols=100)
        model.eval()
        i = 0

        with torch.no_grad:
            for inputs_target, inputs_anchor, _ in bar:
                inputs_anchor, inputs_target = inputs_anchor.to(device), inputs_target.to(device)
                mask = mask_generator(inputs_target.shape[0], patch_numbers)

                output_target = softmax(model(inputs_target), dim=1)
                output_anchor = softmax(model(inputs_anchor, bool_masked_pos=mask), dim=1)

                loss_value = criterion(output_anchor, output_target)
                running_validation_loss += loss_value.item()

                writer.add_scalar(f'TestLoop/epoch_{epoch}_loss', loss_value, i)
                bar.set_postfix(loss=loss_value.item())
                i += 1

            bar.close()
            epoch_test_loss = running_validation_loss / len(datasets['train'])

        writer.add_scalar(f'Averaged losses for epoch/train', epoch_train_loss, epoch)
        writer.add_scalar(f'Averaged losses for epoch/validation', epoch_test_loss, epoch)

        torch.save(model.state_dict(), os.path.join(Config.exp_dir, 'checkpoints', f'model_{epoch}.pth'))

        filename = os.path.join(Config.exp_dir, 'checkpoints', 'models_details.txt')
        with open(filename, 'a') as file:
            concatenated_string = f'Epoch: {epoch} - Train loss: {epoch_train_loss} - Validation loss: {epoch_test_loss}\n'
            file.write(concatenated_string)

    writer.close()

def test_loop(model_path: str):

    datasets = cholec80_images.get_pytorch_dataloaders(
        data_root=Config.data_root,
        batch_size=Config.batch_size,
        double_img=True
    )

    model = MyViTMSNModel()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    patch_size = model.vitMsn.config.patch_size
    patch_numbers = (224 * 224) // (patch_size * patch_size)

    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter(log_dir=os.path.join(Config.exp_dir, 'tb_logs'))

    running_test_loss = 0.0
    bar = tqdm(datasets['test'], total=len(datasets['test']), desc=f'Test', ncols=100)
    model.train()
    i = 0

    for inputs_target, inputs_anchor, _ in bar:
        inputs_anchor, inputs_target = inputs_anchor.to(device), inputs_target.to(device)
        mask = mask_generator(inputs_target.shape[0], patch_numbers)

        output_target = softmax(model(inputs_target), dim=1)
        output_anchor = softmax(model(inputs_anchor, bool_masked_pos=mask), dim=1)

        loss_value = criterion(output_anchor, output_target)
        loss_value.backward()
        running_test_loss += loss_value.item()

        writer.add_scalar(f'TestLoop/loss', loss_value, i)
        bar.set_postfix(loss=loss_value.item())
        i += 1

    bar.close()
    print(f'Average loss for test: {running_test_loss / len(datasets["test"])}')


if __name__ == '__main__':
    training_loop()








from traceback import print_exc

import torch
from numpy.ma.core import count
from torch import optim
from torch import nn
from torch.nn.functional import softmax
from transformers import ViTMSNModel, AutoImageProcessor
from tqdm import tqdm
import numpy as np
import sys, os

sys.path.append(os.path.realpath(__file__ + '/../../'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from data import cholec80_images
from train import train_lib
from config import Config


class MyViTMSNModel(nn.Module):
    def __init__(self):
        super(MyViTMSNModel, self).__init__()
        self.image_processor = AutoImageProcessor.from_pretrained("facebook/vit-msn-small") # Preprocessing the image
        self.vitMsn = ViTMSNModel.from_pretrained('facebook/vit-msn-small')
        self.classifier = nn.Linear(self.vitMsn.config.hidden_size, 1024)

        if self.vitMsn.embeddings.mask_token is None:
            self.vitMsn.embeddings.mask_token = nn.Parameter(torch.zeros(1, 1, self.vitMsn.config.hidden_size))

    def forward(self, inputs, bool_masked_pos = None):
        inputs = torch.tensor(np.array(self.image_processor(inputs, do_rescale=False)['pixel_values'])).to(device)
        if bool_masked_pos is not None:
            output = self.vitMsn(inputs, bool_masked_pos=bool_masked_pos)[0]
        else:
            output = self.vitMsn(inputs)[0]
        output = self.classifier(output[:, 0, :])
        return output


def training_loop(config):

    config.batch_size = 150

    # Load the dataloader for the cholec80 dataset
    # datasets = cholec80_images.get_pytorch_dataloaders(
    #     data_root=config.data_root,
    #     batch_size=config.batch_size,
    #     train_transformation=config.train_transformation,
    # )

    model = MyViTMSNModel() # Load the model
    model.to(device)
    patch_size = model.vitMsn.config.patch_size # Get the patch size of the model
    patch_numbers = (224 * 224) // (patch_size * patch_size) # Calculate the number of patches

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = train_lib.cross_entropy

    config.callbacks_names = ['reduce_lr_plateau', 'tensorboard']
    callbacks = train_lib.get_callbacks(
        config.callbacks_names,
        optimizer,
        os.path.join('..','exps', 'pretraining', 'tmp'),
        config.monitor_metric,
        0.01)

    best_val_loss = float('inf')
    counter = 0

    for epoch in range(2):

        running_loss = 0.0

        bar = tqdm(datasets['train'], total=len(datasets['train']), desc=f'Epoch {epoch+1}', ncols=100)
        model.train()
        for inputs, _ in bar:
            optimizer.zero_grad()

            output_target = softmax(model(inputs), dim=1)
            # generation of a mask of half of the patches
            mask = torch.zeros(inputs.shape[0], patch_numbers)
            for i in range(inputs.shape[0]):
                idx = torch.randperm(patch_numbers)[:patch_numbers // 2]
                mask[i, idx] = 1
            mask = mask == 1
            output_anchor = softmax(model(inputs, bool_masked_pos=mask), dim=1)

            loss_value = criterion(output_anchor, output_target)
            loss_value.backward()
            optimizer.step()
            running_loss += loss_value.item()

            bar.set_postfix(loss=loss_value.item())

        bar.close()
        epoch_train_loss = running_loss / len(datasets['train'])
        print(epoch_train_loss)

        # validation
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            bar_eval = tqdm(datasets['validation'], total=len(datasets['validation']), desc=f'Epoch {epoch + 1}', ncols=100)
            for inputs, _ in bar_eval:
                output_target = softmax(model(inputs), dim=1)
                mask = torch.zeros(inputs.shape[0], patch_numbers)
                for i in range(inputs.shape[0]):
                    idx = torch.randperm(patch_numbers)[:patch_numbers // 2]
                    mask[i, idx] = 1
                mask = mask == 1
                output_anchor = softmax(model(inputs, bool_masked_pos=mask), dim=1)
                loss_value = criterion(output_anchor, output_target)

                bar_eval.set_postfix(loss=loss_value.item())
                running_loss += loss_value.item()

        bar_eval.close()
        epoch_val_loss = running_loss / len(datasets['validation'])
        print(epoch_val_loss)

        if epoch_val_loss < best_val_loss:
            print('-->Saving model')
            best_val_loss = epoch_val_loss
            counter += 1
            torch.save(model.state_dict(), os.path.join('..', 'exps', 'pretraining', 'tmp', 'checkpoints', f'best_model{counter}.pth'))

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

        if 'step_scheduler' in config.callbacks_names:
            print('-->Step scheduler')
            callbacks['step_scheduler'].step()
        if 'tensorboard' in config.callbacks_names:
            print('-->Saving tensorboard train')
            callbacks['tensorboard'].add_scalar('Train/Loss', epoch_train_loss, epoch)


def test(config):

    device = 'cpu'
    config.batch_size = 1

    datasets = cholec80_images.get_pytorch_dataloaders(
        data_root=config.data_root,
        batch_size=config.batch_size,
        train_transformation=config.train_transformation,
    )

    image_processor = AutoImageProcessor.from_pretrained("facebook/vit-msn-small") # Preprocessing the image
    model = MyViTMSNModel() # Load the model
    model.load_state_dict(torch.load(os.path.join('../exps/pretraining', 'best_model1.pth')))

    for param in model.parameters():
        param.requires_grad = False

    patch_size = model.vitMsn.config.patch_size # Get the patch size of the model
    patch_numbers = (224 * 224) // (patch_size * patch_size) # Calculate the number of patches

    model.to(device)
    model.eval()
    criterion = train_lib.cross_entropy

    bar_eval = tqdm(datasets['test'], total=len(datasets['test']), desc='Test', ncols=100)
    running_loss = 0.0
    with torch.no_grad():
        for inputs, _ in bar_eval:
            inputs = torch.tensor(image_processor(inputs)['pixel_values']).to(device)

            output_target = softmax(model(inputs), dim=1)
            mask = torch.randint(0, 2, (inputs.shape[0], patch_numbers))
            output_anchor = softmax(model(inputs, bool_masked_pos=mask), dim=1)

            loss_value = criterion(output_anchor, output_target)
            running_loss += loss_value.item()

            bar_eval.set_postfix(loss=loss_value.item())
        bar_eval.close()


if __name__ == '__main__':
    training_loop(Config())

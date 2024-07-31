from traceback import print_exc

import torch
from torch import optim
from torch import nn
from torch.nn import Softmax
from transformers import ViTMSNModel, AutoImageProcessor
from tqdm import tqdm

import sys, os

from triton.ops.blocksparse import softmax

sys.path.append(os.path.realpath(__file__ + '/../../'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from data import cholec80_images
from config import Config


class MyViTMSNModel(nn.Module):
    def __init__(self):
        super(MyViTMSNModel, self).__init__()
        self.vitMsn = ViTMSNModel.from_pretrained('facebook/vit-msn-small')
        self.classifier = nn.Linear(self.vitMsn.config.hidden_size, 7)
        self.bool_masked_pos  = None

        if self.vitMsn.embeddings.mask_token is None:
            self.vitMsn.embeddings.mask_token = nn.Parameter(torch.zeros(1, 1, self.vitMsn.config.hidden_size))

    def forward(self, inputs, bool_masked_pos = None):
        if bool_masked_pos is not None:
            output = self.vitMsn(inputs, bool_masked_pos=bool_masked_pos)[0]
        else:
            output = self.vitMsn(inputs)[0]
        output = self.classifier(output[:, 0, :])
        return output


def training_loop(config):

    config.batch_size = 5

    # Load the dataloader for the cholec80 dataset
    datasets = cholec80_images.get_pytorch_dataloaders(
        data_root=config.data_root,
        batch_size=config.batch_size,
        train_transformation=config.train_transformation,
    )

    image_processor = AutoImageProcessor.from_pretrained("facebook/vit-msn-small") # Preprocessing the image
    model = MyViTMSNModel() # Load the model
    model.to(device)
    patch_size = model.vitMsn.config.patch_size # Get the patch size of the model
    patch_numbers = (224 * 224) // (patch_size * patch_size) # Calculate the number of patches

    optimizer = optim.Adam(model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()
    softmax = nn.Softmax()

    best_val_loss = float('inf')
    counter = 0

    for epoch in range(2):

        running_loss = 0.0

        bar = tqdm(datasets['train'], total=len(datasets['train']), desc=f'Epoch {epoch+1}')
        model.train()
        for inputs, _ in bar:

            inputs = torch.tensor(image_processor(inputs)['pixel_values']).to(device)
            optimizer.zero_grad()

            output_target = softmax(model(inputs))
            mask = torch.randint(0, 2, (inputs.shape[0], patch_numbers)) == 1
            output_anchor = softmax(model(inputs, bool_masked_pos=mask))

            loss_value = criterion(output_anchor, output_target)
            loss_value.backward()
            optimizer.step()
            running_loss += loss_value.item()

            bar.set_postfix(loss=loss_value.item() / (bar.n + 1))

        bar.close()
        epoch_train_loss = running_loss / len(datasets['train'])
        print(epoch_train_loss)

        # validation

        if epoch % 1 == 0:
            model.eval()
            running_loss = 0.0
            with torch.no_grad():
                bar_eval = tqdm(datasets['validation'], total=len(datasets['train']), desc=f'Epoch {epoch + 1}')
                for inputs, _ in bar_eval:
                    inputs = torch.tensor(image_processor(inputs)['pixel_values']).to(device)

                    output_target = softmax(model(inputs))
                    mask = torch.randint(0, 2, (inputs.shape[0], patch_numbers)) == 1
                    output_anchor = softmax(model(inputs, bool_masked_pos=mask))

                    loss_value = criterion(output_anchor, output_target)
                    running_loss += loss_value.item()

                    bar_eval.set_postfix(loss=loss_value.item() / (bar.n + 1))
            bar_eval.close()

            epoch_val_loss = running_loss / len(datasets['validation'])
            print(epoch_val_loss)

            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                counter += 1
                torch.save(model.state_dict(), os.path.join('../exps/PretrainedModels', f'best_model{counter}.pth'))


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
    model.load_state_dict(torch.load(os.path.join('../exps/PretrainedModels', 'best_model1.pth')))

    for param in model.parameters():
        param.requires_grad = False

    patch_size = model.vitMsn.config.patch_size # Get the patch size of the model
    patch_numbers = (224 * 224) // (patch_size * patch_size) # Calculate the number of patches

    model.to(device)
    model.eval()
    softmax = nn.Softmax()
    criterion = nn.CrossEntropyLoss()

    bar_eval = tqdm(datasets['test'], total=len(datasets['test']), desc='Test')
    running_loss = 0.0
    with torch.no_grad():
        for inputs, _ in bar_eval:
            inputs = torch.tensor(image_processor(inputs)['pixel_values']).to(device)

            output_target = softmax(model(inputs))
            mask = torch.randint(0, 2, (inputs.shape[0], patch_numbers))
            output_anchor = softmax(model(inputs, bool_masked_pos=mask))

            loss_value = criterion(output_anchor, output_target)
            running_loss += loss_value.item()

            bar_eval.set_postfix(loss=loss_value.item() / (bar_eval.n + 1))
        bar_eval.close()


if __name__ == '__main__':
    training_loop(Config())

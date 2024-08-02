import torch
import torch.nn as nn
import numpy as np
from transformers import ViTMSNModel, AutoImageProcessor

class MyViTMSNModel(nn.Module):
    """This class implement the ViT-MSN model for the pre-training part. It will upload the model from the Hugging Face
    model hub and add a classifier layer on top of the model. Default model is the 'facebook/vit-msn-small' model and
    the classifier layer is a linear layer with 1024 output neurons. This FC layer can be accessed by model.classifier

    Args:
        pretrained_model_name_or_path (str): The name or path of the model to be loaded. Default is 'facebook/vit-msn-small'
        device: a string that contain the device to be used. Default is 'cpu', but can be changed to 'cuda'
        if GPU is available
        """
    def __init__(self, pretrained_model_name_or_path : str = 'facebook/vit-msn-small', device : str = 'cpu'):
        super(MyViTMSNModel, self).__init__()
        self.image_processor = AutoImageProcessor.from_pretrained('facebook/vit-msn-small')
        self.vitMsn = ViTMSNModel.from_pretrained(pretrained_model_name_or_path)
        self.classifier = nn.Linear(self.vitMsn.config.hidden_size, 1024)
        self.device = device

        if self.vitMsn.embeddings.mask_token is None:
            self.vitMsn.embeddings.mask_token = nn.Parameter(torch.zeros(1, 1, self.vitMsn.config.hidden_size))

    def forward(self, inputs, bool_masked_pos = None):
        inputs = torch.tensor(np.array(self.image_processor(inputs, do_rescale=False)['pixel_values'])).to(self.device)
        if bool_masked_pos is not None:
            output = self.vitMsn(inputs, bool_masked_pos=bool_masked_pos)[0]
        else:
            output = self.vitMsn(inputs)[0]
        output = self.classifier(output[:, 0, :])
        return output
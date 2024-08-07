import torch
import torch.nn as nn
from transformers import ViTMSNModel, AutoImageProcessor

class MyViTMSNModel(nn.Module):
    """This class implement the ViT-MSN model for the transfer learning part. It will upload the model from the Hugging Face
    model hub and add a classifier layer on top of the model. Default model is the 'facebook/vit-msn-small' model and
    the classifier layer is a linear layer with 1024 output neurons. This FC layer can be accessed by model.classifier

    Args:
        pretrained_model_name_or_path (str): The name or path of the model to be loaded. Default is 'facebook/vit-msn-small'
        device: a string that contain the device to be used. Default is 'cpu', but can be changed to 'cuda'
        if GPU is available
        """
    def __init__(self, pretrained_model_name_or_path : str = 'facebook/vit-msn-small', device : str = 'cpu'):
        super(MyViTMSNModel, self).__init__()
        self.image_processor = AutoImageProcessor.from_pretrained("facebook/vit-msn-small")
        self.vitMsn = ViTMSNModel.from_pretrained(pretrained_model_name_or_path)
        self.classifier = nn.Linear(self.vitMsn.config.hidden_size, 1024, bias=False)
        self.device = device

        if self.vitMsn.embeddings.mask_token is None:
            self.vitMsn.embeddings.mask_token = nn.Parameter(torch.zeros(1, 1, self.vitMsn.config.hidden_size))

    def forward(self, inputs):
        inputs = torch.Tensor(self.image_processor(inputs, do_rescale=False, return_tensors="np")['pixel_values']).to(self.device)
        output = self.vitMsn(inputs)[0]
        output = self.classifier(output[:, 0, :])
        return output
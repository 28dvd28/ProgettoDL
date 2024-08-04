import torch
import torch.nn as nn
from transformers import ViTMSNModel, AutoImageProcessor

class MyViTMSNModel_pretraining(nn.Module):
    """This class implement the ViT-MSN model for the pre-training part. It will upload the model from the Hugging Face
    model hub and add a classifier layer on top of the model. Default model is the 'facebook/vit-msn-small' model and
    the classifier layer is a linear layer with 1024 output neurons. This FC layer can be accessed by model.classifier

    Args:
        pretrained_model_name_or_path (str): The name or path of the model to be loaded. Default is 'facebook/vit-msn-small'
        device: a string that contain the device to be used. Default is 'cpu', but can be changed to 'cuda'
        if GPU is available
        """
    def __init__(self, pretrained_model_name_or_path : str = 'facebook/vit-msn-small', device : str = 'cpu'):
        super(MyViTMSNModel_pretraining, self).__init__()
        self.image_processor = AutoImageProcessor.from_pretrained("facebook/vit-msn-small")
        self.vitMsn_target = ViTMSNModel.from_pretrained(pretrained_model_name_or_path)
        self.vitMsn_anchor = ViTMSNModel.from_pretrained(pretrained_model_name_or_path)
        self.classifier = nn.Linear(self.vitMsn_anchor.config.hidden_size, 1024, bias=False)
        self.device = device

        if self.vitMsn_anchor.embeddings.mask_token is None:
            self.vitMsn_anchor.embeddings.mask_token = nn.Parameter(torch.zeros(1, 1, self.vitMsn_anchor.config.hidden_size))

        for param in self.vitMsn_target.parameters():
            param.requires_grad = False

        patch_size = self.vitMsn_anchor.config.patch_size
        self.patch_numbers = (224 * 224) // (patch_size * patch_size)

    def forward(self, img_anchor, img_target):
        if self.train:

            bool_masked_pos = self.mask_generator(img_anchor.shape[0], self.patch_numbers)

            img_anchor = torch.Tensor(self.image_processor(img_anchor, do_rescale=False, return_tensors="np")['pixel_values']).to(self.device)
            img_target = torch.Tensor(self.image_processor(img_target, do_rescale=False, return_tensors="np")['pixel_values']).to(self.device)

            output_anchor = self.vitMsn_anchor(img_anchor, bool_masked_pos=bool_masked_pos)[0]
            output_anchor = self.classifier(output_anchor[:, 0, :])

            output_target = self.vitMsn_target(img_target)[0]
            output_target = self.classifier(output_target[:, 0, :])
        else:
            output_anchor = self.vitMsn_anchor(img_anchor)[0]
            output_anchor = self.classifier(output_anchor[:, 0, :])

            output_target = self.vitMsn_target(img_target)[0]
            output_target = self.classifier(output_target[:, 0, :])


        return output_anchor, output_target

    def parameters(self):

        params = [
            *list(self.vitMsn_anchor.parameters()),
            *list(self.classifier.parameters()),
        ]

        return params


    def mask_generator(self, batch_size: int, patch_numbers: int) -> torch.Tensor:
        """ Generate a mask for the input tensor, such that half of the patches will be masked.

        Args:
            batch_size (int): The size of the batch
            patch_numbers (int): The number of patches in the input tensor
        """

        mask = torch.zeros(batch_size, patch_numbers)
        for i in range(batch_size):
            idx = torch.randperm(patch_numbers)[:patch_numbers  // 2]
            mask[i, idx] = 1
        return mask

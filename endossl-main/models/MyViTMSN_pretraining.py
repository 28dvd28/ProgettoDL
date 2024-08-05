from idlelib.pyshell import MyRPCClient

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
    def __init__(self, ipe, num_epochs, pretrained_model_name_or_path : str = 'facebook/vit-msn-small', device : str = 'cpu'):
        super(MyViTMSNModel_pretraining, self).__init__()
        self.image_processor = AutoImageProcessor.from_pretrained("facebook/vit-msn-small")
        self.vitMsn_target = ViTMSNModel.from_pretrained(pretrained_model_name_or_path)
        self.vitMsn_anchor = ViTMSNModel.from_pretrained(pretrained_model_name_or_path)
        self.device = device
        self.train_phase = True
        self.tau = 0.1
        self.prototypes, self.prototypes_labels = self.prototypes_init(1024, self.vitMsn_anchor.config.hidden_size)
        self.momentum_scheduler = self.momentum_scheduler_init(ipe, num_epochs)

        if self.vitMsn_anchor.embeddings.mask_token is None:
            self.vitMsn_anchor.embeddings.mask_token = nn.Parameter(torch.zeros(1, 1, self.vitMsn_anchor.config.hidden_size))

        for param in self.vitMsn_target.parameters():
            param.requires_grad = False

        patch_size = self.vitMsn_anchor.config.patch_size
        self.patch_numbers = (224 * 224) // (patch_size * patch_size)

    def forward(self, img_anchor, img_target):

        bool_masked_pos = self.mask_generator(img_anchor.shape[0], self.patch_numbers)

        img_anchor = torch.Tensor(self.image_processor(img_anchor, do_rescale=False, return_tensors="np")['pixel_values']).to(self.device)
        img_target = torch.Tensor(self.image_processor(img_target, do_rescale=False, return_tensors="np")['pixel_values']).to(self.device)

        output_anchor = self.vitMsn_anchor(img_anchor, bool_masked_pos=bool_masked_pos)[0]
        output_anchor = nn.functional.normalize(output_anchor[:, 0, :])
        output_anchor = nn.functional.softmax(output_anchor @ self.prototypes.T / self.tau) @ self.prototypes_labels

        output_target = self.vitMsn_target(img_target)[0]
        output_target = nn.functional.normalize(output_target[:, 0, :])
        output_target = nn.functional.softmax(output_target @ self.prototypes.T / self.tau) @ self.prototypes_labels

        return output_anchor, output_target


    def parameters(self):

        """
        Function for the creation of the list of parameters of the model, will return only the parameters of the anchor
        vision transformer and the learnable prototypes
        """

        params = [*list(self.vitMsn_anchor.parameters()), {
            'params': [self.prototypes],
            'LARS_exclude': True,
            'WD_exclude': True,
            'weight_decay': 0
        }]

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


    def prototypes_init(self, num_proto, output_dim):

        """Function for the initialization of the matrix (K, d) for the K prototypes of dimension d

        Args:
            num_proto: int value for number K of different prototypes
            output_dim: int value for dimension d of each prototype
        Return:
            prototypes: matrix of prototypes
            proto_labels: matrix of one hot encoding proto labels
        """

        with torch.no_grad():
            prototypes = torch.empty(num_proto, output_dim)
            _sqrt_k = (1. / output_dim) ** 0.5
            torch.nn.init.uniform_(prototypes, -_sqrt_k, _sqrt_k)
            prototypes = torch.nn.parameter.Parameter(prototypes).to(self.device)

            proto_labels = nn.functional.one_hot(torch.tensor([i for i in range(num_proto)]), num_proto)
            prototypes.requires_grad = True

        return prototypes, proto_labels


    def momentum_scheduler_init(self, ipe, num_epochs):

        """
        Initializing the momentum scheduler that is needed for the exponential moving average for updating the
        target ViT during training, using the learned parameters of the anchor ViT
        Args:
              ipe: iterations per epoch during training loop
              num_epochs: total number of epochs in trianing loop
        Return:
            momentum scheduler
        """

        _start_m, _final_m = 0.996, 1.0
        _increment = (_final_m - _start_m) / (ipe * num_epochs * 1.25)
        momentum_scheduler = (_start_m + (_increment * i) for i in range(int(ipe * num_epochs * 1.25) + 1))
        return momentum_scheduler


    def exponential_moving_average(self):

        """
        Function for updating the target visual transformer parameters during training based on the learned parameters
        of the anchor visual transformer. This function uses the exponential moving average for doing this update.
        """

        with torch.no_grad():
            m = next(self.momentum_scheduler)
            for param_q, param_k in zip(self.vitMsn_anchor.parameters(), self.vitMsn_target.parameters()):
                param_k.data.mul_(m).add_((1. - m) * param_q.detach().data)

from idlelib.pyshell import MyRPCClient
from typing import Any, Generator

import torch
import torch.nn as nn
from transformers import ViTMSNModel, AutoImageProcessor, ViTConfig


class MyViTMSNModel_pretraining(nn.Module):
    """This class implement the ViT-MSN model for the pre-training part. It will upload the image processor from the Hugging Face
    model hub and will set two different Vit-MSN model, with same configuration with a total of 12 hidden layers, 384 as hidden size,
    6 attention heads and a intermediate size (fully connected after the attention) of 1536.

    Are then saved some parameters by default such as tau and then are initialized the learnable prototypes.
    The momentum scheduler is also initialized for the exponential moving average for updating the target ViT during training,
    which parameter will not be trained using backpropagation.

    If necessary will be initialized also the embeddings mask token and is set the patch numbers for the generation of the
    random masking.

    Parameters:
        ipe (int): the number of iterations per epoch during training loop
        num_epochs (int): the total number of epochs in training loop
        device (str): a string that contain the device to be used. Default is 'cpu', but can be changed to 'cuda'
        if GPU is available
        """
    def __init__(self, ipe, num_epochs, device : str = 'cpu'):
        super(MyViTMSNModel_pretraining, self).__init__()
        self.image_processor = AutoImageProcessor.from_pretrained("facebook/vit-msn-small")
        config = ViTConfig(num_hidden_layers=12, hidden_size=384, num_attention_heads=6, intermediate_size=1536)
        self.vitMsn_target = ViTMSNModel(config)
        self.vitMsn_anchor = ViTMSNModel(config)
        self.device = device
        self.train_phase = True
        self.tau = 0.1
        self.prototypes = self.prototypes_init(1024, self.vitMsn_anchor.config.hidden_size)
        self.momentum_scheduler = self.momentum_scheduler_init(ipe, num_epochs)

        if self.vitMsn_anchor.embeddings.mask_token is None:
            self.vitMsn_anchor.embeddings.mask_token = nn.Parameter(torch.zeros(1, 1, self.vitMsn_anchor.config.hidden_size))
        if self.vitMsn_target.embeddings.mask_token is None:
            self.vitMsn_target.embeddings.mask_token = nn.Parameter(torch.zeros(1, 1, self.vitMsn_anchor.config.hidden_size))

        for param in self.vitMsn_target.parameters():
            param.requires_grad = False

        patch_size = self.vitMsn_anchor.config.patch_size
        self.patch_numbers = (224 * 224) // (patch_size * patch_size)

    def forward(self, img_anchor, img_target) -> (torch.Tensor, torch.Tensor):

        """For the forward part the two inputs, that must be the same image with different random data augmentation operations
         will follow two different paths, one for the anchor image and one for the target image.

         Both of them are first given in input to the image processor, that prepare the image for the ViT-MSN model applying
         transformations such as resizing, normalization and tensor conversion, if necessary.

         Then both images are given in inputs to the ViT-MSN models; remember that the anchor branch will use a random
         mask that is previously computed. After the ViT is then computed the dot product with the prototypes matrix,
         scaled by the tau value and applied the softmax for obtaining two probability distributions.

        Parameters:
            img_anchor (torch.Tensor) : containing the anchor image
            img_target (torch.Tensor) : containing the target image

        Returns:
            (torch.Tensor, torch.Tensor) : two tensors containing the probabilities of the anchor and target branches
         """

        bool_masked_pos = self.mask_generator(img_anchor.shape[0], self.patch_numbers)

        img_anchor = torch.Tensor(self.image_processor(img_anchor, do_rescale=False, return_tensors="np")['pixel_values']).to(self.device)
        img_target = torch.Tensor(self.image_processor(img_target, do_rescale=False, return_tensors="np")['pixel_values']).to(self.device)

        output_anchor = self.vitMsn_anchor(img_anchor, bool_masked_pos=bool_masked_pos)[0]
        output_anchor = nn.functional.normalize(output_anchor[:, 0, :])
        output_anchor = nn.functional.softmax(output_anchor @ self.prototypes.T / self.tau, dim=1)

        output_target = self.vitMsn_target(img_target)[0]
        output_target = nn.functional.normalize(output_target[:, 0, :])
        output_target = nn.functional.softmax(output_target @ self.prototypes.T / self.tau, dim=1)

        return output_anchor, output_target


    def mask_generator(self, batch_size: int, patch_numbers: int) -> torch.Tensor:

        """ Generate a random mask for the input tensor, such that half of the patches will be masked.

        Parameters:
            batch_size (int): The size of the batch
            patch_numbers (int): The number of patches in the input tensor

        Returns:
            torch.Tensor: A tensor of shape (batch_size, patch_numbers) containing the random mask, where each position that
            contains a 1 will not mask the corresponding patch, if contains 0 it will.
        """

        mask = torch.zeros(batch_size, patch_numbers)
        for i in range(batch_size):
            idx = torch.randperm(patch_numbers)[:patch_numbers  // 2]
            mask[i, idx] = 1
        return mask


    def prototypes_init(self, num_proto: int, output_dim: int) -> torch.Tensor:

        """Function for the initialization of the matrix (K, d) for the K prototypes of dimension d

        Parameters:
            num_proto (int): int value for number K of different prototypes
            output_dim (int): int value for dimension d of each prototype
        Return:
            torch.Tensor: tensor matrix of prototypes
        """

        with torch.no_grad():
            prototypes = torch.empty(num_proto, output_dim)
            _sqrt_k = (1. / output_dim) ** 0.5
            torch.nn.init.uniform_(prototypes, -_sqrt_k, _sqrt_k)
            prototypes = torch.nn.Parameter(prototypes.to(self.device))

        return prototypes


    def momentum_scheduler_init(self, ipe: int, num_epochs: int) -> Generator[float, Any, None]:

        """
        Initializing the momentum scheduler that is needed for the exponential moving average for updating the
        target ViT during training, using the learned parameters of the anchor ViT
        Parameters:
              ipe (int): iterations per epoch during training loop
              num_epochs  (int): total number of epochs in trianing loop
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

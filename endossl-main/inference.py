"""Simple script for loading pre-trained model and extracting hidden representations from it."""

import torch
import torch.nn as nn
from torchvision import transforms
import argparse
from PIL import Image


def get_image(image_path, size=224):
    img = Image.open(image_path).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])
    img_tensor = transform(img)

    return img_tensor


class ModelWrapper(nn.Module):
    def __init__(self, model_path):
        super(ModelWrapper, self).__init__()
        self.backbone = torch.load(model_path)
        self.backbone.eval()

    def forward(self, x):
        with torch.no_grad():
            return self.backbone(x)[1]

    def get_config(self):
        return {
            'model_path': self.model_path
        }

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model_path",
    type=str,
    default='/root/vits_lapro_private/saved_model_inference',
    help="Path to a pre-trained model",
)
parser.add_argument(
    "--image_path",
    type=str,
    default='/root/cholec80/samples/video01_000001.png',
    help="Path to a image file",
)

def main(args):
    model = ModelWrapper(args.model_path)
    img = get_image(args.image_path)
    embed = model(img.unsqueeze(0)).squeeze(0)
    print('Hidden vector shape is: {}'.format(embed.shape))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

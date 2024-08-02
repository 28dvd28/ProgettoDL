"""Module for creating TF datasets for Cholec80 dataset"""

import os
import xml.etree.ElementTree as ET
import pickle

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

_SUBSAMPLE_RATE = 1

curr_dir = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(curr_dir, 'config.json')

resize = transforms.Resize((224, 224))

resize_and_center_crop = transforms.Resize((224, 224))

_RAND_AUGMENT = transforms.RandAugment(num_ops=3, magnitude=7)


def randaug(image):
    image = resize(image)
    image = image * 255.0
    image = image.to(torch.uint8)
    return _RAND_AUGMENT.forward(image) / 255.0


def get_train_image_transformation(name):
    if name == 'randaug':
        return randaug
    else:
        return resize


class CustomPolypsSetDataset(Dataset):
    def __init__(self, data_root, transform=resize, with_image_path=False):
        self.data_root = data_root
        self.transform = transform
        self.with_image_path = with_image_path

        self.all_frame_names, self.all_labels = self.prebuild()

    def __len__(self):
        return len(self.all_labels)

    def __getitem__(self, idx):
        img_path = self.all_frame_names[idx * _SUBSAMPLE_RATE]
        label = self.all_labels[idx]
        if self.with_image_path:
            return self.parse_example_image_path(img_path, label, img_path)
        else:
            return self.parse_example(img_path, label)

    def prebuild(self):
        all_labels = pickle.load(open(os.path.join(self.data_root, 'labels.pkl'), 'rb'))
        all_frame_names = pickle.load(open(os.path.join(self.data_root, 'frames_path.pkl'), 'rb'))
        return all_frame_names, torch.tensor(all_labels)

    def parse_image(self, image_path):
        img = torchvision.io.read_file(image_path)
        img = torchvision.io.decode_jpeg(img)
        return self.transform(img)

    def parse_label(self, label):
        return label

    def parse_example(self, image_path, label):
        return (self.parse_image(image_path),
                self.parse_label(label))

    def parse_example_image_path(self, image_path, label, image_path_):
        return (self.parse_image(image_path),
                self.parse_label(label),
                image_path_)


def get_pytorch_dataloaders(data_root, batch_size, train_transformation='randaug', with_image_path=False):
    dataloaders = {'train': None, 'validation': None, 'test': None}
    for key in dataloaders.keys():
        dataset = CustomPolypsSetDataset(
            os.path.join(data_root, key+'2019'),
            transform=get_train_image_transformation(train_transformation),
            with_image_path=with_image_path
        )
        if key == 'train':
            dataloaders[key] = DataLoader(dataset, shuffle=True, batch_size=batch_size)
        else:
            dataloaders[key] = DataLoader(dataset, shuffle=False, batch_size=batch_size)

    return dataloaders


if __name__ == '__main__':
    par_dir = os.path.realpath(__file__ + '/../../')
    data_root = os.path.join(par_dir, 'PolypsSet')
    dataloaders = get_pytorch_dataloaders(data_root, 100)

    for (data_batch, labels_batch) in dataloaders['validation']:
        print(data_batch.shape)
        print(labels_batch.shape)

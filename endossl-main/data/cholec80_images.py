"""Module for creating TF datasets for Cholec80 dataset"""

import os

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

_LABEL_NUM_MAPPING = {
    'GallbladderPackaging': 0,
    'CleaningCoagulation': 1,
    'GallbladderDissection': 2,
    'GallbladderRetraction': 3,
    'Preparation': 4,
    'ClippingCutting': 5,
    'CalotTriangleDissection': 6
}

_CHOLEC80_SPLIT = {'train': range(1, 41),
                   'validation': range(41, 49),
                   'test': range(49, 81)}

curr_dir = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(curr_dir, 'config.json')

resize = transforms.Resize((224, 224))
_RAND_AUGMENT = transforms.RandAugment(num_ops=3, magnitude=7)


def randaug(image: torch.Tensor)->torch.Tensor:
    """Apply random augmentation using RandAugment to the image in input, following
    data augmentation method based on `"RandAugment: Practical automated data augmentation
    with a reduced search space".

    Args:
        image (torch.Tensor): Image to augment.
    Returns:
        torch.Tensor: Augmented image.
    """
    image = resize(image)
    image = image * 255.0
    image = image.to(torch.uint8)
    return _RAND_AUGMENT.forward(image) / 255.0


def get_train_image_transformation(name: str):
    """Get the transformation function for the training images.

    Args:
        name (str): Name of the transformation to apply. Can be 'randaug' or 'resize'.
    Returns:
        function: Transformation function
    """
    if name == 'randaug':
        return randaug
    else:
        return resize


class CustomCholec80Dataset(Dataset):
    """Custom dataset for Cholec80 dataset. During the initialization it save just tha paths of the images, uploading
    them only during the __getitem__ call called during the forward process applied by the dataloader.

    Args:
        data_root (str): Path to the root folder of the dataset. Make sure that in the root folder there are
        two separate folders, one for the frames and one for the phase annotations. The frames folder should contain
        a folder for each video, numerated from 01 tov 80. The phase annotations folder should contain a txt file for
        each video, with the same numeration, and the labels separated by a tabulation.
        video_ids (list): List of the video ids to use. It will consider only the frames in the folders corresponding to
        this ids.
        transform (function): Transformation function to apply to the images.
        with_image_path (bool): If True, the __getitem__ method will return also the path of the image.

    """
    def __init__(self, data_root, video_ids, transform=resize, with_image_path=False):
        self.video_ids = video_ids
        self.data_root = data_root
        self.transform = transform
        self.with_image_path = with_image_path

        self.all_frame_names, self.all_labels = self.prebuild(video_ids)

    def __len__(self):
        return len(self.all_labels)

    def __getitem__(self, idx):
        img_path = self.all_frame_names[idx]
        label = self.all_labels[idx]
        if self.with_image_path:
            return self.parse_example_image_path(img_path, label, img_path)
        else:
            return self.parse_example(img_path, label)

    def prebuild(self, video_ids):
        frames_dir = os.path.join(self.data_root, 'frames')
        annos_dir = os.path.join(self.data_root, 'phase_annotations')

        all_labels = []
        all_frame_names = []
        for video_id in video_ids:
            video_frames_dir = os.path.join(frames_dir, video_id)
            frames = [os.path.join(video_frames_dir, f) for f in os.listdir(video_frames_dir)]
            with open(os.path.join(annos_dir, video_id + '-phase.txt'), 'r') as f:
                labels = f.readlines()[1:]
            labels = [l.split('\t')[1][:-1] for l in labels]
            labels = [_LABEL_NUM_MAPPING[l] for l in labels[::]][:len(frames)]
            all_frame_names += frames
            all_labels += labels
        return all_frame_names, all_labels

    def parse_image(self, image_path : str)->torch.Tensor:
        """Parse the image in input, applying the transformation function."""
        img = torchvision.io.read_file(image_path)
        img = torchvision.io.decode_png(img)
        return self.transform(img)

    def parse_label(self, label):
        """Parse the label in input, converting it to a tensor."""
        return label

    def parse_example(self, image_path, label):
        """Parse the example in input, returning the image and the label."""
        return (self.parse_image(image_path),
                self.parse_label(label))

    def parse_example_image_path(self, image_path, label, image_path_):
        """Parse the example in input, returning the image, the label and the path of the image."""
        return (self.parse_image(image_path),
                self.parse_label(label),
                image_path_)


def get_pytorch_dataloaders(data_root, batch_size, with_image_path=False)->dict:
    """Function that return a dictionary with the dataloaders for the Cholec80 dataset. Will contain a dataloader for
    train, test and validation set. For the training set, the images will be augmented and it is applied the shuffle.
    The validation and test dataloaders will only apply resize of the images and there will be no shuffle.

    Args:
        data_root (str): Path to the root folder of the dataset. Make sure that in the root folder there are
        two separate folders, one for the frames and one for the phase annotations. The frames folder should contain
        a folder for each video, numerated from 01 tov 80. The phase annotations folder should contain a txt file for
        each video, with the same numeration, and the labels separated by a tabulation.
        batch_size (int): Batch size for the dataload
        with_image_path (bool): If True, the __getitem__ method will return also the path of the image.
    """

    dataloaders = {}
    for split, ids_range in _CHOLEC80_SPLIT.items():

        if split == 'train':
            train_transformation = 'randaug'
        else:
            train_transformation = 'resize'

        dataset = CustomCholec80Dataset(
            data_root,
            [f'video{i:02}' for i in ids_range],
            transform=get_train_image_transformation(train_transformation),
            with_image_path=with_image_path
        )
        if split == 'train':
            dataloaders[split] = DataLoader(dataset, shuffle=True, batch_size=batch_size)
        else:
            dataloaders[split] = DataLoader(dataset, shuffle=False, batch_size=batch_size)

    return dataloaders


if __name__ == '__main__':
    par_dir = os.path.realpath(__file__ + '/../../')
    data_root = os.path.join(par_dir, 'cholec80', 'cholec80')
    dataloaders = get_pytorch_dataloaders(data_root, 8)
    validation_dataloader = dataloaders['validation']

    for (data_batch, labels_batch) in validation_dataloader:
        print(data_batch.shape)
        print(labels_batch.shape)
        break

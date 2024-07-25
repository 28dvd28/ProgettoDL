"""Module for creating TF datasets for Cholec80 dataset"""

import os

import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

_CHOLEC80_PHASES_WEIGHTS = {
    0: 1.9219914802981897,
    1: 0.19571110990619747,
    2: 0.9849911311229362,
    3: 0.2993075998175712,
    4: 1.942680301399354,
    5: 1.0,
    6: 2.2015858493443123
}

_LABEL_NUM_MAPPING = {
    'GallbladderPackaging': 0,
    'CleaningCoagulation': 1,
    'GallbladderDissection': 2,
    'GallbladderRetraction': 3,
    'Preparation': 4,
    'ClippingCutting': 5,
    'CalotTriangleDissection': 6
}

_SUBSAMPLE_RATE = 25

_CHOLEC80_SPLIT = {'train': range(1, 41),
                   'validation': range(41, 49),
                   'test': range(49, 81)}

curr_dir = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(curr_dir, 'config.json')

resize = transforms.Resize((224, 224))

resize_and_center_crop = transforms.Resize((224, 224))

_RAND_AUGMENT = transforms.RandAugment(num_ops=3, magnitude=7)


def randaug(image):
    image = resize(image)
    return _RAND_AUGMENT.forward(image * 255.0) / 255.0


def get_train_image_transformation(name):
    if name == 'randaug':
        return randaug
    else:
        return resize


class CustomCholec80Dataset(Dataset):
    def __init__(self, data_root, video_ids, transform=resize, with_image_path=False):
        self.video_ids = video_ids
        self.data_root = data_root
        self.transform = transform
        self.with_image_path = with_image_path

        self.all_frame_names, self.all_labels = self.prebuild(video_ids)

    def __len__(self):
        return len(self.all_frame_names)

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
            labels = [_LABEL_NUM_MAPPING[l] for l in labels[::_SUBSAMPLE_RATE]][:len(frames)]
            all_frame_names += frames
            all_labels += labels
        return all_frame_names, all_labels

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
    dataloaders = {}
    for split, ids_range in _CHOLEC80_SPLIT.items():
        dataset = CustomCholec80Dataset(
            data_root,
            [f'video{i:02}' for i in ids_range],
            transform=get_train_image_transformation(train_transformation),
            with_image_path=with_image_path
        )
        dataloaders[split] = DataLoader(dataset, batch_size=batch_size)
    return dataloaders


if __name__ == '__main__':
    par_dir = os.path.realpath(__file__ + '/../../')
    data_root = os.path.join(par_dir, 'cholec80', 'cholec80')
    dataloaders = get_pytorch_dataloaders(data_root, 8)
    validation_dataloader = dataloaders['validation']

    for (data_batch, labels_batch) in enumerate(validation_dataloader):
        print(data_batch.shape)
        print(labels_batch.shape)
        break

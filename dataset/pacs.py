import os.path

from torch.utils.data import Dataset
import numpy as np

from PIL import Image

from torchvision.transforms import transforms


class PACSDataset(Dataset):

    def __init__(self, base_path, name, train=True, transformer=None):
        if train:
            self.paths, self.text_labels = np.load("../data/pacs/{}_train.pkl".format(name), allow_pickle=True)
        else:
            self.paths, self.text_labels = np.load("../data/pacs/{}_test.pkl".format(name), allow_pickle=True)

        label_dict = {'dog': 0, 'elephant': 1, 'giraffe': 2, 'guitar': 3, 'horse': 4, 'house': 5, 'person': 6}

        self.labels = [label_dict[text] for text in self.text_labels]
        self.transformer = transformer
        self.base_path = base_path if base_path is not None else "./data"

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # img_path = os.path.join(self.base_path, self.paths[idx].replace('office_caltech_10', 'office'))
        label = self.labels[idx]
        image = Image.open(self.paths[idx])

        if len(image.split()) != 3:
            image = transforms.Grayscale(num_output_channels=3)(image)

        if self.transformer is not None:
            image = self.transformer(image)

        return image, label

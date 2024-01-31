import torch
import numpy as np
from torch.utils.data import Dataset


class CafeDataset(Dataset):
    def __init__(self, t_file: str, i_file: str):
        """
        Args:
            t_file (str): text file,including text and label
            i_file (str): image file
        """
        text = np.load(t_file)
        self.text = torch.from_numpy(text["data"]).float()
        img = np.load(i_file)
        self.image = torch.from_numpy(img["data"]).squeeze().float()
        self.label = torch.from_numpy(text["label"]).long()

    def __len__(self):
        return self.text.shape[0]

    def __getitem__(self, index: int):
        return {
            'text': self.text[index],
            'image': self.image[index],
            'label': self.label[index]
        }

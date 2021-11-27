import torch
from torch.utils.data import Dataset

from get_dev import get_dev


class AutoencoderDataset(Dataset):

    def __init__(self, M, N):
        # generate labels
        self.labels = torch.zeros(N, dtype=torch.int64).random_(M)

        # generate one hot vectors
        self.samples = torch.zeros(N, M, dtype=torch.float)
        counter = 0
        for i in self.labels:
            self.samples[counter, i] = 1
            counter += 1

        dev = get_dev()
        self.labels = self.labels.to(dev)
        self.samples = self.samples.to(dev)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]

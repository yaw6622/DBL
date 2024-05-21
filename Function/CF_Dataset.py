#CFDataset function to package the data and labels 
#so that it can be conveniently called in the main function.
import os
import numpy as np
import torch
from torch.utils.data import Dataset

class CFDataset(Dataset):
    def __init__(self, data_dir, label_dir, transform=None, norm=True):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.transform = transform
        self.norm = norm

        # Get the paths of all data and label files.
        self.data_files = sorted(os.listdir(self.data_dir))
        self.label_files = sorted(os.listdir(self.label_dir))

        # Calculate the mean and standard deviation for each channel.
        if self.norm:
            self.mean, self.std = self._compute_channel_stats()

    def __getitem__(self, idx):
        data_file = os.path.join(self.data_dir, self.data_files[idx])
        label_file = os.path.join(self.label_dir, self.label_files[idx])

        # Read data and label files.
        data = np.load(data_file).astype(np.float16)
        label = np.load(label_file).astype(np.uint8)

        # Normalize each channel.
        if self.norm:
            data = self._standardize_channels(data)

        # Apply data augmentation transformations.
        if self.transform is not None:
            data, label = self.transform(data, label)

        return data, label

    def __len__(self):
        return len(self.data_files)

    def _compute_channel_stats(self):
        channel_stats = []
        for channel in range(10):
            channel_data = []
            for data_file in self.data_files:
                data = np.load(os.path.join(self.data_dir, data_file)).astype(np.float64)
                channel_data.append(data[:, channel].flatten())
            channel_data = np.concatenate(channel_data, axis=0)
            channel_mean = np.mean(channel_data)
            channel_std = np.std(channel_data)
            channel_stats.append((channel_mean, channel_std))

        mean = [stat[0] for stat in channel_stats]
        std = [stat[1] for stat in channel_stats]

        return mean, std

    def _standardize_channels(self, data):
        standardized_data = np.zeros_like(data, dtype=np.float32)
        for channel in range(10):
            standardized_data[:, channel] = (data[:, channel] - self.mean[channel]) / self.std[channel]

        return standardized_data

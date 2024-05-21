#我有一些.npy格式的数据文件和一些.npy格式的标注文件，数据文件分别存储在CF_Train和CF_Test文件夹中，
#标注分别存储在Label_Train和Label_Test文件夹中。帮我写一个CFDataset函数，将训练和验证的分别的数据和标注打包为pytorch的dataset格式，方便在主函数中调用
import os
import numpy as np
import torch
from torch.utils.data import Dataset

# class CFDataset(Dataset):
#     def __init__(self, data_dir, label_dir, transform=None, norm=True):
#         self.data_dir = data_dir
#         self.label_dir = label_dir
#         self.transform = transform
#         self.norm = norm
#
#         # 获取所有数据和标签文件的路径
#         self.data_files = os.listdir(self.data_dir)
#         self.label_files = os.listdir(self.label_dir)
#
#         # 计算数据集的均值和标准差
#         if self.norm:
#             self.mean, self.std = self._compute_norm()
#
#     def __len__(self):
#         return len(self.data_files)
#
#     def __getitem__(self, idx):
#         data_file = os.path.join(self.data_dir, self.data_files[idx])
#         label_file = os.path.join(self.label_dir, self.label_files[idx])
#
#         # 读取数据和标签文件
#         data = np.load(data_file).astype(np.float16)
#         label = np.load(label_file).astype(np.uint8)
#
#         # 归一化数据和标签
#         if self.norm:
#             data = self._normalize(data, self.mean, self.std)
#
#         # 应用数据增强变换
#         if self.transform is not None:
#             data, label = self.transform(data, label)
#
#         return data, label
#
#     def set_batch_size(self, batch_size):
#         self.batch_size = batch_size
#
#     def get_batch_size(self):
#         return self.batch_size
#
#     def _compute_norm(self):
#         mean_list = []
#         std_list = []
#         for data_file in self.data_files:
#             data = np.load(os.path.join(self.data_dir, data_file)).astype(np.float64)
#             mean = np.mean(data)
#             std = np.std(data)
#             mean_list.append(mean)
#             std_list.append(std)
#
#         mean = np.mean(mean_list)
#         print(mean)
#         std = np.mean(std_list)
#         print(std)
#
#         return mean, std
#
#     def _normalize(self, data, mean, std):
#         return (data - mean) / std
class CFDataset(Dataset):
    def __init__(self, data_dir, label_dir, transform=None, norm=True):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.transform = transform
        self.norm = norm

        # 获取所有数据和标签文件的路径
        self.data_files = sorted(os.listdir(self.data_dir))
        self.label_files = sorted(os.listdir(self.label_dir))

        # 计算每个通道的均值和标准差
        if self.norm:
            self.mean, self.std = self._compute_channel_stats()

    def __getitem__(self, idx):
        data_file = os.path.join(self.data_dir, self.data_files[idx])
        label_file = os.path.join(self.label_dir, self.label_files[idx])

        # 读取数据和标签文件
        data = np.load(data_file).astype(np.float16)
        label = np.load(label_file).astype(np.uint8)

        # 对每个通道进行标准化
        if self.norm:
            data = self._standardize_channels(data)

        # 应用数据增强变换
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

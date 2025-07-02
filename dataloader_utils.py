# dataloader_utils.py

import torch
import numpy as np
from torch.utils.data import Dataset

class PixelTimeSeriesDataset(Dataset):
    def __init__(self, X, y, x_static):
        self.X = torch.tensor(X, dtype=torch.float32).transpose(1, 0)
        self.y = torch.tensor(y, dtype=torch.float32).transpose(1, 0)
        self.x_static = torch.tensor(x_static, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.x_static[idx]

def split_data(X, static, y, coords=None, splits=(0.4, 0.3, 0.3), downsample_ratio=1.0, seed=42):
    assert np.isclose(sum(splits), 1.0)
    T, P, _ = X.shape
    total_samples = int(P * downsample_ratio)
    np.random.seed(seed)
    all_indices = np.random.permutation(P)[:total_samples]
    n_train = int(splits[0] * total_samples)
    n_val = int(splits[1] * total_samples)
    idx_train, idx_val, idx_test = all_indices[:n_train], all_indices[n_train:n_train+n_val], all_indices[n_train+n_val:]

    def slice(idx): return (X[:, idx], static[idx], y[:, idx], coords[idx] if coords is not None else None)
    return {
        'train': slice(idx_train),
        'val': slice(idx_val),
        'test': slice(idx_test),
    }

def load_data(data_dir="prepared_data"):
    X = np.load(f"{data_dir}/X.npy")
    static = np.load(f"{data_dir}/static.npy")
    y = np.load(f"{data_dir}/y.npy")
    coords = np.load(f"{data_dir}/coords_lan_lon.npy")
    return X, static, y, coords

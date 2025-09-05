import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Callable

class GenericEEGDataset(Dataset):
    """
    A generic EEG dataset class suitable for both HMC and TUSZ datasets.
    
    Assuming the dataset has the following structure: 
    - root_path/
        - train/
            - file1.npy
            - file2.npy
            - ...
        - dev/
        - eval/
        - train.json
        - dev.json
        - eval.json
    """
    def __init__(self, root_path: str, split: str, transform: Optional[Callable] = None):
        """
        Args:
            root_path (str): root path to the dataset.
            split (str): split to load, one of 'train', 'dev' and 'eval'.
            transform (Optional[Callable]): a function/transform to apply to the data.
        """
        self.root_path = root_path
        self.split = split
        self.transform = transform

        if split not in ['train', 'dev', 'eval']:
            raise ValueError(f"Invalid split '{split}'. Must be one of 'train', 'dev', 'eval'.")

        self.data_dir = os.path.join(self.root_path, self.split)
        manifest_path = os.path.join(self.root_path, f"{self.split}.json")

        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(f"Data directory not found at: {self.data_dir}")
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(f"Manifest JSON not found at: {manifest_path}")

        print(f"Loading manifest from: {manifest_path}")
        with open(manifest_path, 'r') as f:
            self.manifest = json.load(f)
            
    def __len__(self) -> int:
        """Length of the dataset."""
        return len(self.manifest)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.

        Returns:
            tuple: (data, label) where data is a Tensor of shape (C, T) and label is a Tensor.
        """
        sample_info = self.manifest[idx]
        file_id = sample_info['id']
        label = sample_info['label']

        # Load .npy
        npy_path = os.path.join(self.data_dir, f"{file_id}.npy")
        try:
            data = np.load(npy_path).astype(np.float32)
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found for id '{file_id}' at: {npy_path}")
        
        # To PyTorch Tensor
        data_tensor = torch.from_numpy(data)
        label_tensor = torch.tensor(label, dtype=torch.long)

        # Apply transform if any
        if self.transform:
            data_tensor = self.transform(data_tensor)

        return data_tensor, label_tensor
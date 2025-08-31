import json
import torch
import numpy as np
from torch.utils.data import Dataset

class SimpleDataset(Dataset):
    """
    A dataset used for EEG recognition, loading data from a JSON file.
    JSON format: [{"id": "sample_id_1", "label": 0}, {"id": "sample_id_2", "label": 1}, ...]
    """
    def __init__(self, json_path, eeg_folder):
        """
        Args:
            json_path (string): path to the JSON file.
            eeg_folder (string): directory containing EEG .npy files.
        """
        self.json_path = json_path
        self.eeg_folder = eeg_folder
        with open(self.json_path, 'r', encoding='utf-8') as f:
            self.samples = json.load(f)

    def __len__(self):
        """Return the total number of samples."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): index of the sample to retrieve.
        
        Returns:
            tuple: (eeg_tensor, label_tensor) where eeg_tensor is the EEG data
        """
        sample_record = self.samples[idx]
        
        sample_id = sample_record['id']
        label = sample_record['label']
        
        eeg_tensor = torch.from_numpy(
            np.load(f"{self.eeg_folder}/{sample_id}.npy")
        ).float()

        label_tensor = torch.tensor(label, dtype=torch.float32)
        
        return eeg_tensor, label_tensor

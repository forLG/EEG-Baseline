import torch
from torch.utils.data import DataLoader
from addict import Dict

from .base_dataset import GenericEEGDataset

def build_dataloader(config: Dict) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build DataLoaders for training, validation, and testing based on the provided configuration.

    Args:
        config (Dict): Configuration dictionary.

    Returns:
        A tuple containing (train_loader, val_loader, test_loader).
    """
    dataset_name = config.name.lower()
    
    if dataset_name in ['hmc', 'tusz']:
        DatasetClass = GenericEEGDataset
    else:
        raise ValueError(f"Unknown dataset name: '{dataset_name}'. Supported datasets are 'hmc', 'tusz'.")

    # print(f"Building DataLoaders for '{dataset_name}' dataset...")

    # TODO: Apply data transforms according to config
    train_dataset = DatasetClass(root_path=config.path, split='train', transform=None)
    val_dataset = DatasetClass(root_path=config.path, split='dev', transform=None)
    test_dataset = DatasetClass(root_path=config.path, split='eval', transform=None)
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,  # shuffle for training
        num_workers=config.get('num_workers', 0),
        pin_memory=config.get('pin_memory', True)
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.get('num_workers', 0),
        pin_memory=config.get('pin_memory', True)
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.get('num_workers', 0),
        pin_memory=config.get('pin_memory', True)
    )

    # print("DataLoaders created successfully.")
    return train_loader, val_loader, test_loader
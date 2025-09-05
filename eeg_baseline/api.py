from typing import Dict, Any
import torch

from .utils.config_parser import load_and_merge_config
from .datasets.builder import build_dataloader
from .models.builder import build_model
from .pipelines import train_pipeline, eval_pipeline

def train(config_path: str, **kwargs) -> Dict[str, Any]:
    """
    Train a model based on the provided configuration file and additional parameters.

    Args:
        config_path (str): Path to the configuration file.
        **kwargs: Additional parameters to override configuration settings.

    Returns:
        Dict[str, Any]: A dictionary containing training results and metrics.
    """
    # print("--- Starting Training Pipeline ---")

    config = load_and_merge_config(config_path, **kwargs)
    
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    
    print(f"Building dataset: {config.dataset.name}")
    train_loader, val_loader, _ = build_dataloader(config.dataset)

    print(f"Building model: {config.model.name}")
    model = build_model(config.model)
    model.to(device)

    results = train_pipeline.run(config, model, train_loader, val_loader)

    # print("--- Training Finished ---")
    print(f"Results: {results}")
    
    return results

def evaluate(ckpt_path: str, config_path: str, **kwargs) -> Dict[str, Any]:
    """
    Evaluate a model based on the provided checkpoint and configuration file.

    Args:
        ckpt_path (str): Path to the model checkpoint.
        config_path (str): Path to the configuration file.
        **kwargs: Additional parameters to override configuration settings.

    Returns:
        Dict[str, Any]: A dictionary containing evaluation results and metrics.
    """
    # print("--- Starting Evaluation Pipeline ---")

    tmp_dict = {'model': {'params': {'pretrain_model_path': ckpt_path}}}
    config = load_and_merge_config(config_path, **tmp_dict, **kwargs)

    torch.manual_seed(config.seed)
    device = torch.device(config.device)

    print(f"Building dataset: {config.dataset.name}")
    _, _, test_loader = build_dataloader(config.dataset)

    print(f"Building model: {config.model.name}")
    model = build_model(config.model)
    model.to(device)

    results = eval_pipeline.run(config, model, test_loader)
    # print("--- Evaluation Finished ---")
    print(f"Results: {results}")

    return results
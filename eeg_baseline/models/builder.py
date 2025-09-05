import importlib
from typing import Any
import torch.nn as nn
from addict import Dict

def build_model(model_config: Dict) -> nn.Module:
    """
    Dynamically builds a model from a config object.
    Args:
        config (object): A config object with 'name' and other model parameters.
            Example: config.name = 'eegnet', config.params = {'param1': val1}
    """
    model_name = model_config.name.lower()
    module_path = f"eeg_baseline.models.{model_name}.model"
    
    try:
        model_module = importlib.import_module(module_path)
        model_class = getattr(model_module, model_config.class_name) # e.g. "EEGNet" or "GramWrapper"
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Model {model_config.name} or class {model_config.class_name} not found at {module_path}") from e

    model = model_class(**model_config.params)
    return model
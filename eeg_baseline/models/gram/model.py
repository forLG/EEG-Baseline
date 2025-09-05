from types import SimpleNamespace
from einops import rearrange
import torch

from .original.modeling_Gram_finetune import Gram
from ..base_model import BaseModel

class GramWrapper(BaseModel):
    """
    A Wrapper class for the Gram model to integrate with the EEG baseline framework.
    """
    def __init__(
        self, 
        num_classes: int, 
        dataset_name: str, 
        base_model_path: str, 
        vqgan_model_path: str, 
        **kwargs):
        """
        Args:
            num_classes (int): Number of output classes for classification.
            dataset_name (str): Name of the dataset to determine channel configuration.
            base_model_path (str): Path to the base model checkpoint.
            **kwargs: Additional keyword arguments for Gram model configuration.
        """
        super().__init__()

        # Load base model configuration
        base_model_ckpt = torch.load(base_model_path, map_location='cpu')
        pre_cf = base_model_ckpt['cf']
        pre_cf.if_scratch = False
        pre_cf.if_finetune = True
        pre_cf.n_class = num_classes
        pre_cf.vqgan_model_path = vqgan_model_path
        if 'layer_scale_init_values' not in pre_cf: 
            pre_cf.layer_scale_init_values = 0.1
        if 'drop_path_rate' not in pre_cf:
            pre_cf.drop_path_rate = 0.1

        # Determine channel list based on dataset
        self.ch_list = self.get_ch_list_for_dataset(dataset_name)
        if not self.ch_list:
            raise ValueError(f"Channel list for dataset '{dataset_name}' not found.")

        self.model = Gram(pre_cf).to(device='cuda' if torch.cuda.is_available() else 'cpu')

        # Load base model weights
        state_dict = self.model.state_dict()
        for k,v in state_dict.items():
            if k in base_model_ckpt['model'] and 'vqgan' not in k:
                state_dict[k] = base_model_ckpt['model'][k]
            self.model.load_state_dict(state_dict)  

        # Load pretrained weights
        pretrain_model_path = kwargs.get('pretrain_model_path', None)
        if pretrain_model_path:
            print(f"Loading pretrained weights from {pretrain_model_path}")
            checkpoint = torch.load(pretrain_model_path, map_location='cpu')
            self.load_state_dict(checkpoint)
            
            # pre_cf = checkpoint.get('cf', SimpleNamespace())
            # self.model.cf.if_finetune = self.cf.if_finetune
            # self.model.cf.if_scratch = self.cf.if_scratch
            # self.model.cf.n_class = self.cf.n_class
            # self.model.cf.vqgan_model_path = self.cf.vqgan_model_path
            # if not hasattr(pre_cf, 'layer_scale_init_values'): 
            #     self.model.cf.layer_scale_init_values = self.cf.layer_scale_init_values
            # if not hasattr(pre_cf, 'drop_path_rate'):
            #     self.model.cf.drop_path_rate = self.cf.drop_path_rate

            # state_dict = self.model.state_dict()
            # # filter out unnecessary keys
            # pretrained_dict = {k: v for k, v in checkpoint['model'].items() if k in state_dict and 'cls_head' not in k}
            # state_dict.update(pretrained_dict)
            # self.model.load_state_dict(state_dict)
            # print(f"Loaded {len(pretrained_dict)} matching keys from checkpoint.")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass after wrapping, automatically determines channels based on dataset.
        """
        x = x.float() / 100.0  # Normalize to avoid gradient explosion
        x = rearrange(x, 'B N (A T) -> B N A T', T=200) # 200 is the sampling rate
        logits, _ = self.model(x, self.ch_list)

        return logits

    def get_ch_list_for_dataset(self, dataset_name: str):
        """Return channel list for a given dataset name."""
        ch_lists = {
            'HMC': ['F4', 'C3', 'C4', 'O2'], 
            'TUSZ': ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'A1', 'A2', 'CZ']
        }
        return ch_lists.get(dataset_name.upper())
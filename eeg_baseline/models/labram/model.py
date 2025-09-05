import torch
from collections import OrderedDict
from einops import rearrange

from ..base_model import BaseModel
from .original.modeling_finetune import NeuralTransformer, labram_base_patch200_200

class LaBraMWrapper(BaseModel):
    def __init__(
        self,
        num_classes: int,
        dataset_name: str,
        base_model_path: str,
        drop_rate: float = 0.0, 
        drop_path_rate: float = 0.1,
        attn_drop_rate: float = 0.0,
        drop_block_rate: float = 0.0,
        use_mean_pooling: bool = True,
        init_scale: float = 0.001,
        use_rel_pos_bias: bool = False,
        use_abs_pos_emb: bool = True,
        init_values: float = 0.1,
        qkv_bias: bool = False,
        **kwargs
    ):
        """
        Wrapper for LaBraM model.

        Args:
            num_classes (int): Number of output classes.
            dataset_name (str): Name of the dataset to determine channel configration.
            base_model_path (str): Path to the base model checkpoint.
            **kwargs: Additional keyword arguments.

            For details of other parameters, please refer to the original code.
        """
        super().__init__()

        self.model = labram_base_patch200_200(
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            attn_drop_rate=attn_drop_rate,
            drop_block_rate=drop_block_rate,
            use_mean_pooling=use_mean_pooling,
            init_scale=init_scale,
            use_rel_pos_bias=use_rel_pos_bias,
            use_abs_pos_emb=use_abs_pos_emb,
            init_values=init_values,
            qkv_bias=qkv_bias,
            num_classes=num_classes
        )

        self.ch_list = self.get_ch_list_for_dataset(dataset_name)

        # Load base model weights, copied from original code
        model_key = 'model|module'
        model_prefix = ''
        model_filter_name = 'gzp'

        base_model_ckpt = torch.load(base_model_path, map_location='cpu')
        checkpoint_model = None
        for model_key in model_key.split('|'):
            if model_key in base_model_ckpt:
                checkpoint_model = base_model_ckpt[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = base_model_ckpt
        if (checkpoint_model is not None) and (model_filter_name != ''):
            all_keys = list(checkpoint_model.keys())
            new_dict = OrderedDict()
            for key in all_keys:
                if key.startswith('student.'):
                    new_dict[key[8:]] = checkpoint_model[key]
                else:
                    pass
            checkpoint_model = new_dict

        state_dict = self.model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        all_keys = list(checkpoint_model.keys())
        for key in all_keys:
            if "relative_position_index" in key:
                checkpoint_model.pop(key)

        self._load_state_dict(checkpoint_model, prefix=model_prefix)

        # Load pretrained weights if provided
        pretrain_model_path = kwargs.get('pretrain_model_path', None)
        if pretrain_model_path:
            print(f"Loading pretrained weights from {pretrain_model_path}")
            checkpoint = torch.load(pretrain_model_path, map_location='cpu')
            self.load_state_dict(checkpoint)

    def forward(self, x):
        """
        Forward pass after wrapping, automatically determines channels based on dataset.
        """

        x = x.float() / 100.0 # Normalize to avoid gradient explosion
        x = rearrange(x, 'B N (A T) -> B N A T', T=200) # 200 is the sampling rate

        return self.model(x, self.ch_list)

    def get_ch_list_for_dataset(self, dataset_name: str):
        """Return channel list for a given dataset name."""
        # Transferred to index defined in original code, 0 is the cls token
        ch_lists = {
            'HMC': [0, 22, 40, 44, 81],
            # ['F4', 'C3', 'C4', 'O2'], 
            'TUSZ': [0, 1, 3, 18, 22, 42, 46, 62, 66, 79, 81, 16, 24, 87, 89, 88, 90, 93, 94, 44]
            # ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'A1', 'A2', 'CZ']
        }
        return ch_lists.get(dataset_name.upper())

    # Copied from original code
    def _load_state_dict(self, state_dict, prefix='', ignore_missing="relative_position_index"):
        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(
                prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(self.model, prefix=prefix)

        warn_missing_keys = []
        ignore_missing_keys = []
        for key in missing_keys:
            keep_flag = True
            for ignore_key in ignore_missing.split('|'):
                if ignore_key in key:
                    keep_flag = False
                    break
            if keep_flag:
                warn_missing_keys.append(key)
            else:
                ignore_missing_keys.append(key)

        missing_keys = warn_missing_keys

        if len(missing_keys) > 0:
            print("Weights of {} not initialized from pretrained model: {}".format(
                self.model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            print("Weights from pretrained model not used in {}: {}".format(
                self.model.__class__.__name__, unexpected_keys))
        if len(ignore_missing_keys) > 0:
            print("Ignored weights of {} not initialized from pretrained model: {}".format(
                self.model.__class__.__name__, ignore_missing_keys))
        if len(error_msgs) > 0:
            print('\n'.join(error_msgs))
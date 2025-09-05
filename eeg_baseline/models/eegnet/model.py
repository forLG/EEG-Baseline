# The code is adapted from https://github.com/abhishekmshr956/EEGNet

import torch
from torch import nn

class EEGNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_channels: int,
        window_length: int = 2000,
        num_temporal_filts: int = 8,
        num_spatial_filts: int = 2,
        p_dropout: float = 0.5,
        avgpool_factor: int = 4,
        **kwargs
    ):
        """
        Args:
            num_classes (int): Number of output classes for classification
            num_channels (int): Number of channels in the input data
            window_length: Length of the input data in samples
            num_temporal_filts (int): Number of temporal filters in the first convolutional layer
            num_spatial_filts (int): Number of spatial filters in the second convolutional layer
            p_dropout (int): Probability of dropout
            avgpool_factor (int): Factor for the first average pooling layer
            **kwargs: Additional keyword arguments for EEGNet configuration
        """
        super(EEGNet, self).__init__()
        self.F1 = num_temporal_filts
        self.D = num_spatial_filts
        self.C = num_channels
        self.F2 = self.D * self.F1

        self.p = p_dropout
        self.T = window_length
        self.avgpool_factor1 = avgpool_factor
        self.num_classes = num_classes

        self.block1 = nn.Sequential(
            nn.Conv2d(1, self.F1, (1, 9), padding='same'),
            nn.Conv2d(self.F1, self.D * self.F1, (self.C, 1), groups=self.F1),
            nn.ELU(),
            nn.AvgPool2d((1, self.avgpool_factor1)),
            nn.Dropout(self.p)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(self.F2,
                      self.F2,
                      (1, 2 * (self.T // (self.avgpool_factor1 * 2)) + 1),
                      groups=self.F2,
                      padding='same'),
            nn.Conv2d(self.F2, self.F2, (1, 1)),
            nn.ELU(),
            nn.Flatten(start_dim=1),  # Flatten along dimension 1
            nn.Linear((self.T // self.avgpool_factor1) * self.F2, self.num_classes)  # Fully connected layer
       
            # nn.Flatten(start_dim=2, end_dim=-1),
            # nn.Linear(self.T // self.avgpool_factor1, 1)
        )

        pretrain_model_path = kwargs.get('pretrain_model_path', None)
        if pretrain_model_path:
            print(f"Loading pretrained weights from {pretrain_model_path}")
            checkpoint = torch.load(pretrain_model_path, map_location='cpu')
            self.load_state_dict(checkpoint)

    def forward(self, x):
        x = x.unsqueeze(1)
        block1 = self.block1(x)
        block2 = self.block2(block1)
        return block2

if __name__ == "__main__":
    num_timesteps = 240
    batch_size = 16
    num_channels = 32

    net = EEGNet(num_classes=2,
                 num_chans=num_channels,
                 window_length=num_timesteps,
                 avgpool_factor=2)

    test_data = torch.rand(batch_size, num_channels, num_timesteps)
    print("input shape", test_data.shape)
    print("output shape", net(test_data).shape)
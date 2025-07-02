import torch
import torch.nn as nn

class TestModel(nn.Module):
    def __init__(self, config, device):
        super(TestModel, self).__init__()
        self.conv = nn.Conv2d(in_channels=2, out_channels=config["num_classes"], kernel_size=(1, 1))
        self.device = device

    def forward(self, local_features):
        return self.conv(local_features["soil_moisture_one_week"].to(self.device))

import torch
import torch.nn as nn
import torch.nn.functional as F

######## TestModel

class TestModel(nn.Module):
    def __init__(self, config, device):
        super(TestModel, self).__init__()
        self.conv = nn.Conv2d(in_channels=2, out_channels=config["num_classes"], kernel_size=(1, 1))
        self.device = device

    def forward(self, local_features):
        return self.conv(local_features["soil_moisture_one_week"].to(self.device))

######## BasicUNet -- https://github.com/jaxony/unet-pytorch

def conv3x3(in_channels, out_channels, stride=1, padding=1, bias=True, groups=1):    
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=bias, groups=groups)
def upconv2x2(in_channels, out_channels, mode='transpose'):
    if mode == 'transpose': return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    else: return nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2), conv1x1(in_channels, out_channels))
def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=groups, stride=1)
class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, pooling=True):
        super(DownConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling
        self.conv1 = conv3x3(self.in_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)
        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return x, before_pool
class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, up_mode="transpose"):
        super(UpConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up_mode = up_mode
        self.upconv = upconv2x2(self.in_channels, self.out_channels, mode=self.up_mode)
        self.conv1 = conv3x3(2*self.out_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)
    def forward(self, from_down, from_up):
        from_up = self.upconv(from_up)
        x = torch.cat((from_up, from_down), 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x
class BasicUNet(nn.Module):
    def __init__(self, config, device, depth=5, start_filts=64, up_mode="transpose"):
        super(BasicUNet, self).__init__()
        self.up_mode = up_mode
        self.num_classes = config["num_classes"]
        self.channels_in_features = {"dem":1, "permanent_water":1, "soil_moisture_one_week":2, "soil_moisture_one_day":2, "soil_class":1}
        self.in_channels = sum([self.channels_in_features[feature] for feature in config["features"]])
        self.start_filts = start_filts
        self.depth = depth
        self.down_convs = []
        self.up_convs = []
        self.device = device
        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts*(2**i)
            pooling = True if i < depth-1 else False
            down_conv = DownConv(ins, outs, pooling=pooling)
            self.down_convs.append(down_conv)
        for i in range(depth-1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs, up_mode=up_mode)
            self.up_convs.append(up_conv)
        self.conv_final = conv1x1(outs, self.num_classes)
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)
    def forward(self, local_features):
        encoder_outs = []
        x = torch.concat(list(local_features.values()), dim=1).to(self.device)
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)
        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i+2)]
            x = module(before_pool, x)
        x = self.conv_final(x)
        return x
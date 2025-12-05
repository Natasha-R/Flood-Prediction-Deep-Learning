import torch
import torch.nn as nn
import torch.nn.functional as F
import modelling.utils as utils

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
    def __init__(self, config, depth=5, start_filts=64, up_mode="transpose"):
        super(BasicUNet, self).__init__()

        self.up_mode = up_mode
        self.num_classes = config["num_classes"]
        self.in_channels, self.in_embeddings = utils.find_num_channels(config)
        self.in_channels = self.in_channels * len(config["scales"])
        self.in_embeddings = self.in_embeddings * len(config["scales"])
        self.start_filts = start_filts
        self.depth = depth
        self.down_convs = []
        self.up_convs = []
        self.feature_classes_exist = False
        if self.in_embeddings > 0:
             self.feature_classes_exist = True
             self.embedding = nn.Embedding(num_embeddings=31, embedding_dim=self.in_embeddings)
        self.scales = config["scales"]

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
        self.local_final = conv1x1(outs, self.num_classes)
        self.context_final = conv1x1(outs, self.num_classes)
        self.basin_final = conv1x1(outs, self.num_classes)
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)

    def forward(self, data):
        encoder_outs = []
        
        if self.feature_classes_exist:
            x = torch.concat([torch.concat([data[f"{scale}_features"], 
                                            self.embedding(torch.clip(data[f"{scale}_classes"], 0, 30).int()).squeeze(1).permute(0, 3, 1, 2)],
                                            dim=1) for scale in self.scales], dim=1)
        else:
            x = torch.concat([data[f"{scale}_features"] for scale in self.scales], dim=1)

        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)
        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i+2)]
            x = module(before_pool, x)

        predictions = {}
        predictions["local_pred"] = self.local_final(x)
        if "context" in self.scales:
            predictions["context_pred"] = self.context_final(x)
        if "basin" in self.scales:
            predictions["basin_pred"] = self.basin_final(x)
        return predictions
    
class ChainedUNet(nn.Module):
    def __init__(self, config, depth=5, start_filts=64, up_mode="transpose"):
        super(ChainedUNet, self).__init__()

        self.up_mode = up_mode
        self.num_classes = config["num_classes"]
        self.in_channels, self.in_embeddings = utils.find_num_channels(config)
        self.start_filts = start_filts
        self.depth = depth

        self.local_down_convs = []
        self.local_up_convs = []
        self.context_down_convs = []
        self.context_up_convs = []
        self.basin_down_convs = []
        self.basin_up_convs = []

        self.feature_classes_exist = False
        if self.in_embeddings > 0:
             self.feature_classes_exist = True
             self.embedding = nn.Embedding(num_embeddings=31, embedding_dim=self.in_embeddings)
        self.scales = config["scales"]

        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts*(2**i)
            pooling = True if i < depth-1 else False
            down_conv = DownConv(ins, outs, pooling=pooling)
            self.basin_down_convs.append(down_conv)
        for i in range(depth-1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs, up_mode=up_mode)
            self.basin_up_convs.append(up_conv)      
        self.basin_final = conv1x1(outs, self.num_classes) #BclassesHW 

        for i in range(depth):
            ins = self.in_channels + self.num_classes if i == 0 else outs
            outs = self.start_filts*(2**i)
            pooling = True if i < depth-1 else False
            down_conv = DownConv(ins, outs, pooling=pooling)
            self.context_down_convs.append(down_conv)
        for i in range(depth-1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs, up_mode=up_mode)
            self.context_up_convs.append(up_conv)
        self.context_final = conv1x1(outs, self.num_classes) #BclassesHW

        for i in range(depth):
            ins = self.in_channels + self.num_classes if i == 0 else outs
            outs = self.start_filts*(2**i)
            pooling = True if i < depth-1 else False
            down_conv = DownConv(ins, outs, pooling=pooling)
            self.local_down_convs.append(down_conv)
        for i in range(depth-1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs, up_mode=up_mode)
            self.local_up_convs.append(up_conv)
        self.local_final = conv1x1(outs, self.num_classes) #BclassesHW

        self.local_down_convs = nn.ModuleList(self.local_down_convs)
        self.local_up_convs = nn.ModuleList(self.local_up_convs)
        self.context_down_convs = nn.ModuleList(self.context_down_convs)
        self.context_up_convs = nn.ModuleList(self.context_up_convs)
        self.basin_down_convs = nn.ModuleList(self.basin_down_convs)
        self.basin_up_convs = nn.ModuleList(self.basin_up_convs)

    def forward(self, data):

        basin_encoder_outs = []
        if self.feature_classes_exist:
            x = torch.concat([data[f"basin_features"], self.embedding(torch.clip(data[f"basin_classes"], 0, 30).int()).squeeze(1).permute(0, 3, 1, 2)], dim=1)
        else:
            x = data[f"basin_features"]
        for i, module in enumerate(self.basin_down_convs):
            x, basin_before_pool = module(x)
            basin_encoder_outs.append(basin_before_pool)
        for i, module in enumerate(self.basin_up_convs):
            basin_before_pool = basin_encoder_outs[-(i+2)]
            x = module(basin_before_pool, x)
        basin_pred = self.basin_final(x)

        context_encoder_outs = []
        if not self.feature_classes_exist:
            x = torch.concat([F.softmax(basin_pred, dim=1), data[f"context_features"]], dim=1)
        else:
            x = torch.concat([basin_pred, data[f"context_features"], self.embedding(torch.clip(data[f"context_classes"], 0, 30).int()).squeeze(1).permute(0, 3, 1, 2)], dim=1)
        for i, module in enumerate(self.context_down_convs):
            x, context_before_pool = module(x)
            context_encoder_outs.append(context_before_pool)
        for i, module in enumerate(self.context_up_convs):
            context_before_pool = context_encoder_outs[-(i+2)]
            x = module(context_before_pool, x)
        context_pred = self.context_final(x)

        local_encoder_outs = []
        if not self.feature_classes_exist:
            x = torch.concat([F.softmax(context_pred, dim=1), data[f"local_features"]], dim=1)
        else:
            x = torch.concat([context_pred, data[f"local_features"], self.embedding(torch.clip(data[f"local_classes"], 0, 30).int()).squeeze(1).permute(0, 3, 1, 2)], dim=1)
        for i, module in enumerate(self.local_down_convs):
            x, local_before_pool = module(x)
            local_encoder_outs.append(local_before_pool)
        for i, module in enumerate(self.local_up_convs):
            local_before_pool = local_encoder_outs[-(i+2)]
            x = module(local_before_pool, x)
        local_pred = self.local_final(x)

        return {"local_pred": local_pred, "context_pred": context_pred, "basin_pred": basin_pred}
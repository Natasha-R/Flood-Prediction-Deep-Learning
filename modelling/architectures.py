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
    def __init__(self, in_channels, out_channels, num_scales=1, up_mode="transpose"):
        super(UpConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up_mode = up_mode
        self.upconv = upconv2x2(self.in_channels, self.out_channels, mode=self.up_mode)
        self.conv1 = conv3x3((num_scales+1)*self.out_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)
    def forward(self, from_down, from_up):
        from_up = self.upconv(from_up)
        x = torch.cat((from_up, from_down), 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x
def create_embeddings(config):
    num_classes = {"soil_class":31, "land_cover":12}
    class_features = ["soil_class", "land_cover"]
    return nn.ModuleList([nn.Embedding(num_embeddings=num_classes[feature], embedding_dim=3) for feature in config["features"] if feature in class_features])
    
class BasicUNet(nn.Module):
    def __init__(self, config, depth=5, start_filts=64, up_mode="transpose"):
        super(BasicUNet, self).__init__()

        self.up_mode = up_mode
        self.num_classes = config["num_classes"]
        self.in_channels = utils.find_num_channels(config)
        self.in_channels = self.in_channels * len(config["scales"])
        self.start_filts = start_filts
        self.depth = depth
        self.down_convs = []
        self.up_convs = []
        self.feature_classes_exist = False
        if config["class_features_exist"]:
             self.feature_classes_exist = True
             self.num_class_features = config["num_class_features"]
             self.embeddings = create_embeddings(config)
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
        if "context" in self.scales:
            self.context_final = conv1x1(outs, self.num_classes)
        if "basin" in self.scales:
            self.basin_final = conv1x1(outs, self.num_classes)
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)

    def forward(self, data):
        encoder_outs = []
        
        if self.feature_classes_exist:
            x = torch.concat([torch.concat([data[f"{scale}_features"]] + [self.embeddings[index](torch.clip(data[f"{scale}_classes"][:, index, :, :], 0, 30).int()).squeeze(1).permute(0, 3, 1, 2) for index in range(self.num_class_features)],
                                            dim=1) for scale in self.scales], dim=1)
        else:
            x = torch.concat([data[f"{scale}_features"] for scale in self.scales], dim=1)

        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)
        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i+2)]
            x = module(before_pool, x)

        predictions = {"local_pred": self.local_final(x)}
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
        self.in_channels = utils.find_num_channels(config)
        self.start_filts = start_filts
        self.depth = depth
        self.feature_classes_exist = False
        if config["class_features_exist"]:
             self.feature_classes_exist = True
             self.num_class_features = config["num_class_features"]
             self.embeddings = create_embeddings(config)
        self.scales = config["scales"]

        if "basin" in self.scales:
            self.basin_down_convs = []
            self.basin_up_convs = []
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
            self.basin_down_convs = nn.ModuleList(self.basin_down_convs)
            self.basin_up_convs = nn.ModuleList(self.basin_up_convs)

        if "context" in self.scales:
            self.context_down_convs = []
            self.context_up_convs = []
            for i in range(depth):
                if "basin" in self.scales:
                    ins = self.in_channels + self.num_classes if i == 0 else outs
                else:
                    ins = self.in_channels if i == 0 else outs
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
            self.context_down_convs = nn.ModuleList(self.context_down_convs)
            self.context_up_convs = nn.ModuleList(self.context_up_convs)

        self.local_down_convs = []
        self.local_up_convs = []
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

    def forward(self, data):

        if "basin" in self.scales:
            basin_encoder_outs = []
            if self.feature_classes_exist:
                x = torch.concat([data[f"basin_features"]] + [self.embeddings[index](torch.clip(data[f"basin_classes"][:, index, :, :], 0, 30).int()).squeeze(1).permute(0, 3, 1, 2) for index in range(self.num_class_features)], dim=1)
            else:
                x = data[f"basin_features"]
            for i, module in enumerate(self.basin_down_convs):
                x, basin_before_pool = module(x)
                basin_encoder_outs.append(basin_before_pool)
            for i, module in enumerate(self.basin_up_convs):
                basin_before_pool = basin_encoder_outs[-(i+2)]
                x = module(basin_before_pool, x)
            basin_pred = self.basin_final(x)

        if "context" in self.scales:
            context_encoder_outs = []
            if not self.feature_classes_exist:
                if "basin" in self.scales:
                    x = torch.concat([F.softmax(basin_pred, dim=1), data[f"context_features"]], dim=1)
                else:
                    x = data[f"context_features"]
            else:
                if "basin" in self.scales:
                    x = torch.concat([basin_pred, data[f"context_features"]] + [self.embeddings[index](torch.clip(data[f"context_classes"][:, index, :, :], 0, 30).int()).squeeze(1).permute(0, 3, 1, 2) for index in range(self.num_class_features)], dim=1)

                else:
                    x = torch.concat([data[f"context_features"]] + [self.embeddings[index](torch.clip(data[f"context_classes"][:, index, :, :], 0, 30).int()).squeeze(1).permute(0, 3, 1, 2) for index in range(self.num_class_features)], dim=1)

            for i, module in enumerate(self.context_down_convs):
                x, context_before_pool = module(x)
                context_encoder_outs.append(context_before_pool)
            for i, module in enumerate(self.context_up_convs):
                context_before_pool = context_encoder_outs[-(i+2)]
                x = module(context_before_pool, x)
            context_pred = self.context_final(x)

        local_encoder_outs = []
        if not self.feature_classes_exist:
            if "context" in self.scales:
                x = torch.concat([F.softmax(context_pred, dim=1), data[f"local_features"]], dim=1)
            else:
                x = torch.concat([F.softmax(basin_pred, dim=1), data[f"local_features"]], dim=1)
        else:
            if "context" in self.scales:
                x = torch.concat([context_pred, data[f"local_features"]] + [self.embeddings[index](torch.clip(data[f"local_classes"][:, index, :, :], 0, 30).int()).squeeze(1).permute(0, 3, 1, 2) for index in range(self.num_class_features)], dim=1)
            else:
                x = torch.concat([basin_pred, data[f"local_features"]] + [self.embeddings[index](torch.clip(data[f"local_classes"][:, index, :, :], 0, 30).int()).squeeze(1).permute(0, 3, 1, 2) for index in range(self.num_class_features)], dim=1)
        for i, module in enumerate(self.local_down_convs):
            x, local_before_pool = module(x)
            local_encoder_outs.append(local_before_pool)
        for i, module in enumerate(self.local_up_convs):
            local_before_pool = local_encoder_outs[-(i+2)]
            x = module(local_before_pool, x)
        local_pred = self.local_final(x)

        predictions = {"local_pred": local_pred}
        if "context" in self.scales:
            predictions["context_pred"] = context_pred
        if "basin" in self.scales:
            predictions["basin_pred"] = basin_pred
        return predictions
    
class FusedUNet(nn.Module):
    def __init__(self, config, depth=5, start_filts=64, up_mode="transpose"):
        super(FusedUNet, self).__init__()

        self.up_mode = up_mode
        self.num_classes = config["num_classes"]
        self.in_channels = utils.find_num_channels(config)
        self.start_filts = start_filts
        self.depth = depth

        self.feature_classes_exist = False
        if config["class_features_exist"]:
             self.feature_classes_exist = True
             self.num_class_features = config["num_class_features"]
             self.embeddings = create_embeddings(config)
        self.scales = config["scales"]

        if "basin" in self.scales:
            self.basin_down_convs = []
            for i in range(depth):
                ins = self.in_channels if i == 0 else outs
                outs = self.start_filts*(2**i)
                pooling = True if i < depth-1 else False
                down_conv = DownConv(ins, outs, pooling=pooling)
                self.basin_down_convs.append(down_conv)
            self.basin_down_convs = nn.ModuleList(self.basin_down_convs)

        if "context" in self.scales:
            self.context_down_convs = []
            for i in range(depth):
                ins = self.in_channels if i == 0 else outs
                outs = self.start_filts*(2**i)
                pooling = True if i < depth-1 else False
                down_conv = DownConv(ins, outs, pooling=pooling)
                self.context_down_convs.append(down_conv)
            self.context_down_convs = nn.ModuleList(self.context_down_convs)

        self.local_down_convs = []
        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts*(2**i)
            pooling = True if i < depth-1 else False
            down_conv = DownConv(ins, outs, pooling=pooling)
            self.local_down_convs.append(down_conv)
        self.local_down_convs = nn.ModuleList(self.local_down_convs)
        
        self.fusion = nn.Conv2d(start_filts*(2**(depth-1)) * len(self.scales), start_filts*(2**(depth-1)), kernel_size=1)

        self.up_convs = []
        for i in range(depth-1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs, num_scales=len(self.scales), up_mode=up_mode)
            self.up_convs.append(up_conv)      
        self.up_convs = nn.ModuleList(self.up_convs)

        self.basin_final = conv1x1(outs, self.num_classes) #BclassesHW 
        self.context_final = conv1x1(outs, self.num_classes) #BclassesHW
        self.local_final = conv1x1(outs, self.num_classes) #BclassesHW

    def forward(self, data):

        final_features = []

        if "basin" in self.scales:
            basin_encoder_outs = []
            if self.feature_classes_exist:
                basin = torch.concat([data[f"basin_features"]] + [self.embeddings[index](torch.clip(data[f"basin_classes"][:, index, :, :], 0, 30).int()).squeeze(1).permute(0, 3, 1, 2) for index in range(self.num_class_features)], dim=1)
            else:
                basin = data[f"basin_features"]
            for i, module in enumerate(self.basin_down_convs):
                basin, basin_before_pool = module(basin)
                basin_encoder_outs.append(basin_before_pool)
            final_features.append(basin)

        if "context" in self.scales:
            context_encoder_outs = []
            if self.feature_classes_exist:
                context = torch.concat([data[f"context_features"]] + [self.embeddings[index](torch.clip(data[f"context_classes"][:, index, :, :], 0, 30).int()).squeeze(1).permute(0, 3, 1, 2) for index in range(self.num_class_features)], dim=1)
            else:
                context = data[f"context_features"]
            for i, module in enumerate(self.context_down_convs):
                context, context_before_pool = module(context)
                context_encoder_outs.append(context_before_pool)
            final_features.append(context)

        local_encoder_outs = []
        if self.feature_classes_exist:
            local = torch.concat([data[f"local_features"]] + [self.embeddings[index](torch.clip(data[f"local_classes"][:, index, :, :], 0, 30).int()).squeeze(1).permute(0, 3, 1, 2) for index in range(self.num_class_features)], dim=1)
        else:
            local = data[f"local_features"]
        for i, module in enumerate(self.local_down_convs):
            local, local_before_pool = module(local)
            local_encoder_outs.append(local_before_pool)
        final_features.append(local)

        concat = torch.cat(final_features, dim=1)

        x = self.fusion(concat)

        for i, module in enumerate(self.up_convs):
            up_features = []
            if "basin" in self.scales:
                basin_before_pool = basin_encoder_outs[-(i+2)]
                up_features.append(basin_before_pool)
            if "context" in self.scales:
                context_before_pool = context_encoder_outs[-(i+2)]
                up_features.append(context_before_pool)
            local_before_pool = local_encoder_outs[-(i+2)]
            up_features.append(local_before_pool)
            before_pool = torch.cat(up_features, dim=1)
            x = module(before_pool, x)

        predictions = {"local_pred": self.local_final(x)}
        if "context" in self.scales:
            predictions["context_pred"] = self.context_final(x)
        if "basin" in self.scales:
            predictions["basin_pred"] = self.basin_final(x)
        return predictions
    
class FusedBranchedUNet(nn.Module):
    def __init__(self, config, depth=5, start_filts=64, up_mode="transpose"):
        super(FusedBranchedUNet, self).__init__()

        self.up_mode = up_mode
        self.num_classes = config["num_classes"]
        self.in_channels = utils.find_num_channels(config)
        self.start_filts = start_filts
        self.depth = depth

        self.feature_classes_exist = False
        if config["class_features_exist"]:
             self.feature_classes_exist = True
             self.num_class_features = config["num_class_features"]
             self.embeddings = create_embeddings(config)
        self.scales = config["scales"]

        if "basin" in self.scales:
            self.basin_down_convs = []
            self.basin_up_convs = []
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
            self.basin_down_convs = nn.ModuleList(self.basin_down_convs)
            self.basin_up_convs = nn.ModuleList(self.basin_up_convs)
            self.basin_final = conv1x1(outs, self.num_classes) #BclassesHW 

        if "context" in self.scales:
            self.context_down_convs = []
            self.context_up_convs = []
            for i in range(depth):
                ins = self.in_channels if i == 0 else outs
                outs = self.start_filts*(2**i)
                pooling = True if i < depth-1 else False
                down_conv = DownConv(ins, outs, pooling=pooling)
                self.context_down_convs.append(down_conv)
            for i in range(depth-1):
                ins = outs
                outs = ins // 2
                up_conv = UpConv(ins, outs, up_mode=up_mode)
                self.context_up_convs.append(up_conv)
            self.context_down_convs = nn.ModuleList(self.context_down_convs)
            self.context_up_convs = nn.ModuleList(self.context_up_convs)
            self.context_final = conv1x1(outs, self.num_classes) #BclassesHW

        self.local_down_convs = []
        self.local_up_convs = []
        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts*(2**i)
            pooling = True if i < depth-1 else False
            down_conv = DownConv(ins, outs, pooling=pooling)
            self.local_down_convs.append(down_conv)
        for i in range(depth-1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs, up_mode=up_mode)
            self.local_up_convs.append(up_conv)
        self.local_down_convs = nn.ModuleList(self.local_down_convs)
        self.local_up_convs = nn.ModuleList(self.local_up_convs)
        self.local_final = conv1x1(outs, self.num_classes) #BclassesHW
        
        self.fusion = nn.Conv2d(start_filts*(2**(depth-1)) * len(self.scales), start_filts*(2**(depth-1)), kernel_size=1)

    def forward(self, data):

        final_features = []

        if "basin" in self.scales:
            basin_encoder_outs = []
            if self.feature_classes_exist:
                basin = torch.concat([data[f"basin_features"]] + [self.embeddings[index](torch.clip(data[f"basin_classes"][:, index, :, :], 0, 30).int()).squeeze(1).permute(0, 3, 1, 2) for index in range(self.num_class_features)], dim=1)
            else:
                basin = data[f"basin_features"]
            for i, module in enumerate(self.basin_down_convs):
                basin, basin_before_pool = module(basin)
                basin_encoder_outs.append(basin_before_pool)
            final_features.append(basin)

        if "context" in self.scales:
            context_encoder_outs = []
            if self.feature_classes_exist:
                context = torch.concat([data[f"context_features"]] + [self.embeddings[index](torch.clip(data[f"context_classes"][:, index, :, :], 0, 30).int()).squeeze(1).permute(0, 3, 1, 2) for index in range(self.num_class_features)], dim=1)
            else:
                context = data[f"context_features"]
            for i, module in enumerate(self.context_down_convs):
                context, context_before_pool = module(context)
                context_encoder_outs.append(context_before_pool)
            final_features.append(context)

        local_encoder_outs = []
        if self.feature_classes_exist:
            local = torch.concat([data[f"local_features"]] + [self.embeddings[index](torch.clip(data[f"local_classes"][:, index, :, :], 0, 30).int()).squeeze(1).permute(0, 3, 1, 2) for index in range(self.num_class_features)], dim=1)
        else:
            local = data[f"local_features"]
        for i, module in enumerate(self.local_down_convs):
            local, local_before_pool = module(local)
            local_encoder_outs.append(local_before_pool)
        final_features.append(local)

        concat = torch.cat(final_features, dim=1)
        concat = self.fusion(concat)

        if "basin" in self.scales:
            x = concat
            for i, module in enumerate(self.basin_up_convs):
                basin_before_pool = basin_encoder_outs[-(i+2)]
                x = module(basin_before_pool, x)
            basin_pred = self.basin_final(x)

        if "context" in self.scales:
            x = concat
            for i, module in enumerate(self.context_up_convs):
                context_before_pool = context_encoder_outs[-(i+2)]
                x = module(context_before_pool, x)
            context_pred = self.context_final(x)

        x = concat
        for i, module in enumerate(self.local_up_convs):
            local_before_pool = local_encoder_outs[-(i+2)]
            x = module(local_before_pool, x)
        local_pred = self.local_final(x)

        predictions = {"local_pred": local_pred}
        if "context" in self.scales:
            predictions["context_pred"] = context_pred
        if "basin" in self.scales:
            predictions["basin_pred"] = basin_pred
        return predictions
    
class MultiFusedBranchedUNet(nn.Module):
    def __init__(self, config, depth=5, start_filts=64, up_mode="transpose"):
        super(MultiFusedBranchedUNet, self).__init__()

        self.up_mode = up_mode
        self.num_classes = config["num_classes"]
        self.in_channels = utils.find_num_channels(config)
        self.start_filts = start_filts
        self.depth = depth

        self.feature_classes_exist = False
        if config["class_features_exist"]:
             self.feature_classes_exist = True
             self.num_class_features = config["num_class_features"]
             self.embeddings = create_embeddings(config)
        self.scales = config["scales"]

        if "basin" in self.scales:
            self.basin_down_convs = []
            self.basin_up_convs = []
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
            self.basin_down_convs = nn.ModuleList(self.basin_down_convs)
            self.basin_up_convs = nn.ModuleList(self.basin_up_convs)
            self.basin_final = conv1x1(outs, self.num_classes) #BclassesHW 

        if "context" in self.scales:
            self.context_down_convs = []
            self.context_up_convs = []
            for i in range(depth):
                ins = self.in_channels if i == 0 else outs
                outs = self.start_filts*(2**i)
                pooling = True if i < depth-1 else False
                down_conv = DownConv(ins, outs, pooling=pooling)
                self.context_down_convs.append(down_conv)
            for i in range(depth-1):
                ins = outs
                outs = ins // 2
                up_conv = UpConv(ins, outs, up_mode=up_mode)
                self.context_up_convs.append(up_conv)
            self.context_down_convs = nn.ModuleList(self.context_down_convs)
            self.context_up_convs = nn.ModuleList(self.context_up_convs)
            self.context_final = conv1x1(outs, self.num_classes) #BclassesHW

        self.local_down_convs = []
        self.local_up_convs = []
        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts*(2**i)
            pooling = True if i < depth-1 else False
            down_conv = DownConv(ins, outs, pooling=pooling)
            self.local_down_convs.append(down_conv)
        for i in range(depth-1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs, up_mode=up_mode)
            self.local_up_convs.append(up_conv)
        self.local_down_convs = nn.ModuleList(self.local_down_convs)
        self.local_up_convs = nn.ModuleList(self.local_up_convs)
        self.local_final = conv1x1(outs, self.num_classes) #BclassesHW
        
        self.basin_context_fusion = nn.Conv2d(start_filts*(2**(depth-1)) * 2, start_filts*(2**(depth-1)), kernel_size=1)
        self.local_scale_fusion = nn.Conv2d(start_filts*(2**(depth-1)) * {3: 4, 2: 2}.get(len(self.scales)), start_filts*(2**(depth-1)), kernel_size=1)
    
    def forward(self, data):

        bottleneck_for_concat = []

        if "basin" in self.scales:
            basin_encoder_outs = []
            if self.feature_classes_exist:
                basin = torch.concat([data[f"basin_features"]] + [self.embeddings[index](torch.clip(data[f"basin_classes"][:, index, :, :], 0, 30).int()).squeeze(1).permute(0, 3, 1, 2) for index in range(self.num_class_features)], dim=1)
            else:
                basin = data[f"basin_features"]
            for i, module in enumerate(self.basin_down_convs):
                basin, basin_before_pool = module(basin)
                basin_encoder_outs.append(basin_before_pool)
            bottleneck_for_concat.append(basin)
            for i, module in enumerate(self.basin_up_convs):
                basin_before_pool = basin_encoder_outs[-(i+2)]
                basin = module(basin_before_pool, basin)
            basin_pred = self.basin_final(basin)

        if "context" in self.scales:
            context_encoder_outs = []
            if self.feature_classes_exist:
                context = torch.concat([data[f"context_features"]] + [self.embeddings[index](torch.clip(data[f"context_classes"][:, index, :, :], 0, 30).int()).squeeze(1).permute(0, 3, 1, 2) for index in range(self.num_class_features)], dim=1)
            else:
                context = data[f"context_features"]
            for i, module in enumerate(self.context_down_convs):
                context, context_before_pool = module(context)
                context_encoder_outs.append(context_before_pool)
            bottleneck_for_concat.append(context)

            if "basin" in self.scales:
                context = torch.cat(bottleneck_for_concat, dim=1)
                context = self.basin_context_fusion(context)
                bottleneck_for_concat.append(context)

            for i, module in enumerate(self.context_up_convs):
                context_before_pool = context_encoder_outs[-(i+2)]
                context = module(context_before_pool, context)
            context_pred = self.context_final(context)

        local_encoder_outs = []
        if self.feature_classes_exist:
            local = torch.concat([data[f"local_features"]] + [self.embeddings[index](torch.clip(data[f"local_classes"][:, index, :, :], 0, 30).int()).squeeze(1).permute(0, 3, 1, 2) for index in range(self.num_class_features)], dim=1)
        else:
            local = data[f"local_features"]
        for i, module in enumerate(self.local_down_convs):
            local, local_before_pool = module(local)
            local_encoder_outs.append(local_before_pool)
        bottleneck_for_concat.append(local)

        local = torch.cat(bottleneck_for_concat, dim=1)
        local = self.local_scale_fusion(local)

        for i, module in enumerate(self.local_up_convs):
            local_before_pool = local_encoder_outs[-(i+2)]
            local = module(local_before_pool, local)
        local_pred = self.local_final(local)

        predictions = {"local_pred": local_pred}
        if "context" in self.scales:
            predictions["context_pred"] = context_pred
        if "basin" in self.scales:
            predictions["basin_pred"] = basin_pred
        return predictions
import torch
import torch.nn as nn
import torch.nn.functional as F
import modelling.utils as utils
import math

######## BasicUNet -- https://github.com/jaxony/unet-pytorch

def conv3x3(in_channels, out_channels, stride=1, bias=True, groups=1, kernel_size=3):    
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, bias=bias, groups=groups)

def upconv2x2(in_channels, out_channels, mode='transpose'):
    if mode == 'transpose': return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    else: return nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2), conv1x1(in_channels, out_channels))

def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=groups, stride=1)

class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0, pooling=True, kernel_size=3):
        super(DownConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout_rate = dropout
        self.pooling = pooling
        self.conv1 = conv3x3(self.in_channels, self.out_channels, kernel_size=kernel_size)
        self.conv2 = conv3x3(self.out_channels, self.out_channels, kernel_size=kernel_size)
        if self.dropout_rate > 0:
            self.dropout = nn.Dropout2d(self.dropout_rate)
        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        before_pool = x
        if self.pooling:
            x = self.pool(x)
            if self.dropout_rate > 0:
                x = self.dropout(x)
        return x, before_pool
    
class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0, num_scales=1, up_mode="transpose", kernel_size=3):
        super(UpConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up_mode = up_mode
        self.dropout_rate = dropout
        self.upconv = upconv2x2(self.in_channels, self.out_channels, mode=self.up_mode)
        self.conv1 = conv3x3((num_scales+1)*self.out_channels, self.out_channels, kernel_size=kernel_size)
        self.conv2 = conv3x3(self.out_channels, self.out_channels, kernel_size=kernel_size)
        if self.dropout_rate > 0:
            self.dropout = nn.Dropout2d(self.dropout_rate)
    def forward(self, from_down, from_up):
        from_up = self.upconv(from_up)
        x = torch.cat((from_up, from_down), 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        if self.dropout_rate > 0:
            x = self.dropout(x)
        return x
    
def create_embeddings(config):
    num_classes = {"soil_class":31, "land_cover":12}
    class_features = ["soil_class", "land_cover"]
    return nn.ModuleList([nn.Embedding(num_embeddings=num_classes[feature], embedding_dim=3) for feature in config["features"] if feature in class_features])

class ConvFusion(nn.Module):
    def __init__(self, fusion_channels):
        super().__init__()
        self.fusion = nn.Conv2d(fusion_channels * 2, fusion_channels, kernel_size=1)
    def forward(self, feature_map_1, feature_map_2):
        return self.fusion(torch.cat([feature_map_1, feature_map_2], dim=1))

class CrossScaleAttention(nn.Module):
    def __init__(self, config, embed_dim=512, num_heads=8, depth=5):
        super().__init__()

        self.cross_attention = nn.MultiheadAttention(embed_dim=embed_dim,
                                                     num_heads=num_heads,
                                                     batch_first=True,
                                                     dropout=config["dropout"])
        self.norm = nn.LayerNorm(embed_dim)
        self.conv = nn.Conv2d(embed_dim, embed_dim, 1)
        self.relu = nn.ReLU()
        
        size = int(256/(2**(depth-1)))
        x_encoding = torch.linspace(-1, 1, size)
        y_encoding = torch.linspace(-1, 1, size)
        x_encoding = x_encoding.view(1, 1, 1, size).expand(1, 1, size, size)
        y_encoding = y_encoding.view(1, 1, size, 1).expand(1, 1, size, size)
        self.register_buffer("positional_encoding", torch.cat([x_encoding, y_encoding], dim=1), persistent=True)

        self.query_conv = nn.Conv2d(embed_dim + 2, embed_dim, 1)
        self.key_conv = nn.Conv2d(embed_dim + 2, embed_dim, 1)

    def forward(self, query, key):
        B, C, H, W = query.shape
        query = self.query_conv(torch.cat([query, self.positional_encoding.expand(B, 2, H, W)], dim=1))
        key = self.key_conv(torch.cat([key, self.positional_encoding.expand(B, 2, H, W)], dim=1))
        query = query.flatten(2).transpose(1, 2).contiguous()
        key = key.flatten(2).transpose(1, 2).contiguous()
        attended, _ = self.cross_attention(query=query, key=key, value=key)
        attended = query + attended
        attended = self.norm(attended)
        attended = attended.transpose(1, 2).contiguous().reshape(B, C, H, W)
        attended = self.conv(attended)
        attended = self.relu(attended)
        return attended

class BasicUNet(nn.Module):
    def __init__(self, config, depth=5, start_filts=64, up_mode="transpose"):
        super(BasicUNet, self).__init__()

        self.up_mode = up_mode
        self.num_classes = config["num_classes"]
        self.kernel_size = config.get("kernel_size", 3)
        self.in_channels = utils.find_num_channels(config)
        self.in_channels = self.in_channels * len(config["scales"])
        self.dropout = config["dropout"]
        self.start_filts = start_filts
        self.depth = config.get("depth", 5)
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
            down_conv = DownConv(ins, outs, pooling=pooling, dropout=self.dropout, kernel_size=self.kernel_size)
            self.down_convs.append(down_conv)
        for i in range(depth-1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs, up_mode=up_mode, dropout=self.dropout, kernel_size=self.kernel_size)
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
        self.kernel_size = config.get("kernel_size", 3)
        self.in_channels = utils.find_num_channels(config)
        self.dropout = config["dropout"]
        self.start_filts = start_filts
        self.depth = config.get("depth", 5)
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
                down_conv = DownConv(ins, outs, pooling=pooling, dropout=self.dropout, kernel_size=self.kernel_size)
                self.basin_down_convs.append(down_conv)
            for i in range(depth-1):
                ins = outs
                outs = ins // 2
                up_conv = UpConv(ins, outs, up_mode=up_mode, dropout=self.dropout, kernel_size=self.kernel_size)
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
                down_conv = DownConv(ins, outs, pooling=pooling, dropout=self.dropout, kernel_size=self.kernel_size)
                self.context_down_convs.append(down_conv)
            for i in range(depth-1):
                ins = outs
                outs = ins // 2
                up_conv = UpConv(ins, outs, up_mode=up_mode, dropout=self.dropout, kernel_size=self.kernel_size)
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
            down_conv = DownConv(ins, outs, pooling=pooling, dropout=self.dropout, kernel_size=self.kernel_size)
            self.local_down_convs.append(down_conv)
        for i in range(depth-1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs, up_mode=up_mode, dropout=self.dropout, kernel_size=self.kernel_size)
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
    
class BranchedUNet(nn.Module):
    def __init__(self, config, depth=5, start_filts=64, up_mode="transpose"):
        super(BranchedUNet, self).__init__()

        self.up_mode = up_mode
        self.num_classes = config["num_classes"]
        self.use_attention = config.get("use_attention", True)
        self.kernel_size = config.get("kernel_size", 3)
        self.dropout = config["dropout"]
        self.in_channels = utils.find_num_channels(config)
        self.start_filts = start_filts
        self.depth = config.get("depth", 5)
        bottleneck_channels = start_filts * (2 ** (depth - 1))

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
                down_conv = DownConv(ins, outs, pooling=pooling, dropout=self.dropout, kernel_size=self.kernel_size)
                self.basin_down_convs.append(down_conv)
            for i in range(depth-1):
                ins = outs
                outs = ins // 2
                up_conv = UpConv(ins, outs, up_mode=up_mode, dropout=self.dropout, kernel_size=self.kernel_size)
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
                down_conv = DownConv(ins, outs, pooling=pooling, dropout=self.dropout, kernel_size=self.kernel_size)
                self.context_down_convs.append(down_conv)
            for i in range(depth-1):
                ins = outs
                outs = ins // 2
                up_conv = UpConv(ins, outs, up_mode=up_mode, dropout=self.dropout, kernel_size=self.kernel_size)
                self.context_up_convs.append(up_conv)
            self.context_down_convs = nn.ModuleList(self.context_down_convs)
            self.context_up_convs = nn.ModuleList(self.context_up_convs)
            self.context_final = conv1x1(outs, self.num_classes) #BclassesHW
            if "basin" in self.scales:
                self.context_attends_basin = CrossScaleAttention(config, bottleneck_channels) if self.use_attention else ConvFusion(bottleneck_channels)
                
        self.local_down_convs = []
        self.local_up_convs = []
        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts*(2**i)
            pooling = True if i < depth-1 else False
            down_conv = DownConv(ins, outs, pooling=pooling, dropout=self.dropout, kernel_size=self.kernel_size)
            self.local_down_convs.append(down_conv)
        for i in range(depth-1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs, up_mode=up_mode, dropout=self.dropout, kernel_size=self.kernel_size)
            self.local_up_convs.append(up_conv)
        self.local_down_convs = nn.ModuleList(self.local_down_convs)
        self.local_up_convs = nn.ModuleList(self.local_up_convs)
        self.local_final = conv1x1(outs, self.num_classes) #BclassesHW
        self.local_attends_higher = CrossScaleAttention(config, bottleneck_channels) if self.use_attention else ConvFusion(bottleneck_channels)

    def forward(self, data):

        final_features = {}

        if "basin" in self.scales:
            basin_encoder_outs = []
            if self.feature_classes_exist:
                basin = torch.concat([data[f"basin_features"]] + [self.embeddings[index](torch.clip(data[f"basin_classes"][:, index, :, :], 0, 30).int()).squeeze(1).permute(0, 3, 1, 2) for index in range(self.num_class_features)], dim=1)
            else:
                basin = data[f"basin_features"]
            for i, module in enumerate(self.basin_down_convs):
                basin, basin_before_pool = module(basin)
                basin_encoder_outs.append(basin_before_pool)
            final_features["basin"] = basin

        if "context" in self.scales:
            context_encoder_outs = []
            if self.feature_classes_exist:
                context = torch.concat([data[f"context_features"]] + [self.embeddings[index](torch.clip(data[f"context_classes"][:, index, :, :], 0, 30).int()).squeeze(1).permute(0, 3, 1, 2) for index in range(self.num_class_features)], dim=1)
            else:
                context = data[f"context_features"]
            for i, module in enumerate(self.context_down_convs):
                context, context_before_pool = module(context)
                context_encoder_outs.append(context_before_pool)
            final_features["context"] = context

        local_encoder_outs = []
        if self.feature_classes_exist:
            local = torch.concat([data[f"local_features"]] + [self.embeddings[index](torch.clip(data[f"local_classes"][:, index, :, :], 0, 30).int()).squeeze(1).permute(0, 3, 1, 2) for index in range(self.num_class_features)], dim=1)
        else:
            local = data[f"local_features"]
        for i, module in enumerate(self.local_down_convs):
            local, local_before_pool = module(local)
            local_encoder_outs.append(local_before_pool)
        final_features["local"] = local

        if "basin" in self.scales:
            if "context" in self.scales:
                context_attended_basin = self.context_attends_basin(final_features["context"], final_features["basin"])
                local_attended_higher = self.local_attends_higher(final_features["local"], context_attended_basin)
            else:
                local_attended_higher = self.local_attends_higher(final_features["local"], final_features["basin"])
        else:
            local_attended_higher = self.local_attends_higher(final_features["local"], final_features["context"])

        if "basin" in self.scales:
            x = final_features["basin"]
            for i, module in enumerate(self.basin_up_convs):
                basin_before_pool = basin_encoder_outs[-(i+2)]
                x = module(basin_before_pool, x)
            basin_pred = self.basin_final(x)

        if "context" in self.scales:
            if "basin" in self.scales:
                x = context_attended_basin
            else:
                x = final_features["context"]
            for i, module in enumerate(self.context_up_convs):
                context_before_pool = context_encoder_outs[-(i+2)]
                x = module(context_before_pool, x)
            context_pred = self.context_final(x)

        x = local_attended_higher
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
    
class BranchedLocalUNet(nn.Module):
    def __init__(self, config, depth=5, start_filts=64, up_mode="transpose"):
        super(BranchedLocalUNet, self).__init__()

        self.up_mode = up_mode
        self.num_classes = config["num_classes"]
        self.use_attention = config.get("use_attention", True)
        self.kernel_size = config.get("kernel_size", 3)
        self.dropout = config["dropout"]
        self.in_channels = utils.find_num_channels(config)
        self.start_filts = start_filts
        self.depth = config.get("depth", 5)
        bottleneck_channels = start_filts * (2 ** (depth - 1))

        self.feature_classes_exist = False
        if config["class_features_exist"]:
             self.feature_classes_exist = True
             self.num_class_features = config["num_class_features"]
             self.embeddings = create_embeddings(config)
        self.scales = config["scales"]

        if "basin" in self.scales:
            self.basin_weight = config["basin_weight"]
            self.basin_down_convs = []
            for i in range(depth):
                ins = self.in_channels if i == 0 else outs
                outs = self.start_filts*(2**i)
                pooling = True if i < depth-1 else False
                down_conv = DownConv(ins, outs, pooling=pooling, dropout=self.dropout, kernel_size=self.kernel_size)
                self.basin_down_convs.append(down_conv)
            self.basin_down_convs = nn.ModuleList(self.basin_down_convs)

        if "context" in self.scales:
            self.context_weight = config["context_weight"]
            self.context_down_convs = []
            for i in range(depth):
                ins = self.in_channels if i == 0 else outs
                outs = self.start_filts*(2**i)
                pooling = True if i < depth-1 else False
                down_conv = DownConv(ins, outs, pooling=pooling, dropout=self.dropout, kernel_size=self.kernel_size)
                self.context_down_convs.append(down_conv)
            self.context_down_convs = nn.ModuleList(self.context_down_convs)
            if "basin" in self.scales:
                self.context_attends_basin = CrossScaleAttention(config, bottleneck_channels) if self.use_attention else ConvFusion(bottleneck_channels)
                
        self.local_down_convs = []
        self.up_convs = []
        self.local_weight = config["local_weight"]
        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts*(2**i)
            pooling = True if i < depth-1 else False
            down_conv = DownConv(ins, outs, pooling=pooling, dropout=self.dropout, kernel_size=self.kernel_size)
            self.local_down_convs.append(down_conv)
        for i in range(depth-1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs, up_mode=up_mode, num_scales=len(self.scales), dropout=self.dropout, kernel_size=self.kernel_size)
            self.up_convs.append(up_conv)
        self.local_down_convs = nn.ModuleList(self.local_down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)
        self.local_final = conv1x1(outs, self.num_classes) #BclassesHW
        self.local_attends_higher = CrossScaleAttention(config, bottleneck_channels) if self.use_attention else ConvFusion(bottleneck_channels)

    def forward(self, data):

        final_features = {}

        if "basin" in self.scales:
            basin_encoder_outs = []
            if self.feature_classes_exist:
                basin = torch.concat([data[f"basin_features"]] + [self.embeddings[index](torch.clip(data[f"basin_classes"][:, index, :, :], 0, 30).int()).squeeze(1).permute(0, 3, 1, 2) for index in range(self.num_class_features)], dim=1)
            else:
                basin = data[f"basin_features"]
            basin = basin * self.basin_weight
            for i, module in enumerate(self.basin_down_convs):
                basin, basin_before_pool = module(basin)
                basin_encoder_outs.append(basin_before_pool)
            final_features["basin"] = basin

        if "context" in self.scales:
            context_encoder_outs = []
            if self.feature_classes_exist:
                context = torch.concat([data[f"context_features"]] + [self.embeddings[index](torch.clip(data[f"context_classes"][:, index, :, :], 0, 30).int()).squeeze(1).permute(0, 3, 1, 2) for index in range(self.num_class_features)], dim=1)
            else:
                context = data[f"context_features"]
            context = context * self.context_weight
            for i, module in enumerate(self.context_down_convs):
                context, context_before_pool = module(context)
                context_encoder_outs.append(context_before_pool)
            final_features["context"] = context

        local_encoder_outs = []
        if self.feature_classes_exist:
            local = torch.concat([data[f"local_features"]] + [self.embeddings[index](torch.clip(data[f"local_classes"][:, index, :, :], 0, 30).int()).squeeze(1).permute(0, 3, 1, 2) for index in range(self.num_class_features)], dim=1)
        else:
            local = data[f"local_features"]
        local = local * self.local_weight
        for i, module in enumerate(self.local_down_convs):
            local, local_before_pool = module(local)
            local_encoder_outs.append(local_before_pool)
        final_features["local"] = local

        if "basin" in self.scales:
            if "context" in self.scales:
                context_attended_basin = self.context_attends_basin(final_features["context"], final_features["basin"])
                local_attended_higher = self.local_attends_higher(final_features["local"], context_attended_basin)
            else:
                local_attended_higher = self.local_attends_higher(final_features["local"], final_features["basin"])
        else:
            local_attended_higher = self.local_attends_higher(final_features["local"], final_features["context"])

        x = local_attended_higher
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

        return {"local_pred": self.local_final(x)}
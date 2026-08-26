import torch
import torch.nn as nn
import torch.nn.functional as F
import modelling.utils as utils
import math

######## UNet code based on: https://github.com/jaxony/unet-pytorch
######## ResNet code based on: https://github.com/samcw/ResNet18-Pytorch

def conv3x3(in_channels, out_channels, stride=1, bias=True, groups=1, kernel_size=3):    
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, bias=bias, groups=groups)

def upconv2x2(in_channels, out_channels, mode='transpose'):
    if mode == 'transpose': return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    else: return nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2), conv1x1(in_channels, out_channels))

def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=groups, stride=1)

def conv3x3_1x1(feature_channels, num_scales, num_classes):
    return nn.Sequential(
            nn.Conv2d(in_channels=feature_channels * num_scales, out_channels=feature_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=feature_channels, out_channels=num_classes, kernel_size=1))

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

class CropAndResize(nn.Module):
    def __init__(self):
        super(CropAndResize, self).__init__()

        self.depth_to_bottleneck = {1: 256, 2: 128, 3: 64, 4: 32, 5: 16, 6: 8}
        self.scale_to_crop = {"local": {"nearby": {256:(77, 179), 128:(38,89), 64:(19,45), 32:(9,22), 16:(5, 11), 8:(2,5)},
                                        "context": {256: (115, 141), 128:(57,70), 64:(29,35), 32:(14,17), 16:(7, 9), 8:(3,4),}},
                              "nearby":{"context": {256:(96, 160), 128:(48, 80), 64:(24, 40), 32:(12, 20), 16:(6, 10), 8:(3, 5)}}
                              }

    def forward(self, feature_map, to_crop, cropped_to, depth):
        feature_map_size = self.depth_to_bottleneck[depth]
        start, end = self.scale_to_crop[cropped_to][to_crop][feature_map_size]
        return F.interpolate(feature_map[:, :, start:end, start:end], size=(feature_map_size, feature_map_size), mode="bilinear", align_corners=False)
    
class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0, num_scales=1, up_mode="transpose", kernel_size=3, add_residuals=True):
        super(UpConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up_mode = up_mode
        self.dropout_rate = dropout
        self.add_residuals = add_residuals
        num_scales = 0 if not self.add_residuals else num_scales
        self.upconv = upconv2x2(self.in_channels, self.out_channels, mode=self.up_mode)
        self.conv1 = conv3x3((num_scales+1)*self.out_channels, self.out_channels, kernel_size=kernel_size)
        self.conv2 = conv3x3(self.out_channels, self.out_channels, kernel_size=kernel_size)
        if self.dropout_rate > 0:
            self.dropout = nn.Dropout2d(self.dropout_rate)
    def forward(self, from_down, from_up):
        from_up = self.upconv(from_up)
        x = torch.cat((from_up, from_down), 1) if self.add_residuals else from_up
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        if self.dropout_rate > 0:
            x = self.dropout(x)
        return x
    
def create_embeddings(config, scale):
    if scale == "all":
        class_features = [feature for scale_name in config["scales"] for feature in config[f"{scale_name}_features"] if feature in utils.get_class_feature_classes().keys()]
    else:
        class_features = [feature for feature in config[f"{scale}_features"] if feature in utils.get_class_feature_classes().keys()]
    if class_features:
        return len(class_features), nn.ModuleList([nn.Embedding(num_embeddings=utils.get_class_feature_classes()[feature], embedding_dim=3) for feature in class_features])
    else:
        return 0, None

class ConvFusion(nn.Module):
    def __init__(self, fusion_channels):
        super().__init__()
        self.fusion = nn.Conv2d(fusion_channels * 2, fusion_channels, kernel_size=1)
    def forward(self, feature_map_1, feature_map_2):
        return self.fusion(torch.cat([feature_map_1, feature_map_2], dim=1))

class AttentionFusion(nn.Module):  
    def __init__(self, bottleneck_channels=512, num_fusion_inputs=2):
       super().__init__()
       self.conv = nn.Conv2d(bottleneck_channels * num_fusion_inputs, bottleneck_channels, kernel_size=1)
       self.relu = nn.ReLU()
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

class CrossScaleAttention(nn.Module):
    def __init__(self, config, query_scale, key_scale, embed_dim=512, num_heads=8, depth=5):
        super().__init__()

        self.cross_attention = nn.MultiheadAttention(embed_dim=embed_dim,
                                                     num_heads=num_heads,
                                                     batch_first=True,
                                                     dropout=config["dropout"])
        self.norm = nn.LayerNorm(embed_dim)
        self.conv = nn.Conv2d(embed_dim, embed_dim, 1)
        self.relu = nn.ReLU()
        size = int(256/(2**(depth-1)))
        self.local_residual = config.get("local_residual_within_attention", True)
        self.resolutions = {"local": 10, "nearby": 25, "context": 100, "basin": 1000}
        largest_resolution = max([self.resolutions[scale] for scale in config["scales"]])

        if config.get("use_positional_encoding", True):
            q_x_encoding = (((torch.arange(size) + 0.5) / size) * 2.0 - 1.0) * self.resolutions[query_scale] / largest_resolution
            q_y_encoding = (((torch.arange(size) + 0.5) / size) * 2.0 - 1.0) * self.resolutions[query_scale] / largest_resolution
            k_x_encoding = (((torch.arange(size) + 0.5) / size) * 2.0 - 1.0) * self.resolutions[key_scale] / largest_resolution
            k_y_encoding = (((torch.arange(size) + 0.5) / size) * 2.0 - 1.0) * self.resolutions[key_scale] / largest_resolution
        else:
            q_x_encoding = torch.linspace(-1, 1, size)
            q_y_encoding = torch.linspace(-1, 1, size)
            k_x_encoding = torch.linspace(-1, 1, size)
            k_y_encoding = torch.linspace(-1, 1, size)

        q_x_encoding = q_x_encoding.view(1, 1, 1, size).expand(1, 1, size, size)
        q_y_encoding = q_y_encoding.view(1, 1, size, 1).expand(1, 1, size, size)
        self.register_buffer("query_positional_encoding", torch.cat([q_x_encoding, q_y_encoding], dim=1), persistent=True)
        k_x_encoding = k_x_encoding.view(1, 1, 1, size).expand(1, 1, size, size)
        k_y_encoding = k_y_encoding.view(1, 1, size, 1).expand(1, 1, size, size)
        self.register_buffer("key_positional_encoding", torch.cat([k_x_encoding, k_y_encoding], dim=1), persistent=True)

        self.query_conv = nn.Conv2d(embed_dim, embed_dim, 1)
        self.key_conv = nn.Conv2d(embed_dim, embed_dim, 1)
        self.encoding_conv = nn.Conv2d(2, embed_dim, 1, bias=False)

    def forward(self, query, key):
        
        B, C, H, W = query.shape
        query_content = self.query_conv(query)
        embed_query = (query_content + self.encoding_conv(self.query_positional_encoding.expand(B, 2, H, W))).flatten(2).transpose(1, 2).contiguous()
        embed_key = (self.key_conv(key) + self.encoding_conv(self.key_positional_encoding.expand(B, 2, H, W))).flatten(2).transpose(1, 2).contiguous()

        attended, _ = self.cross_attention(query=embed_query, key=embed_key, value=embed_key, need_weights=False)

        if self.local_residual:
            attended = query_content.flatten(2).transpose(1, 2).contiguous() + attended
        attended = self.relu(self.conv((self.norm(attended)).transpose(1, 2).contiguous().reshape(B, C, H, W)))

        return attended

class BasicUNet(nn.Module):
    def __init__(self, config, depth=5, start_filts=64, up_mode="transpose"):
        super(BasicUNet, self).__init__()

        self.up_mode = up_mode
        self.scales = config["scales"]
        if config.get("exclude_scales", False):
            self.scales = [scale for scale in self.scales if scale not in config["exclude_scales"]]
        self.add_residuals = not config.get("no_residuals", False)
        self.num_classes = config["num_classes"] if not config.get("predict_feature", False) else len(utils.get_indices_per_feature()[config["predict_feature"]])
        self.crop_feature_at_start = config.get("crop_feature_at_start", False)
        self.crop_feature_at_bottleneck = config.get("crop_feature_at_bottleneck", False)
        self.crop_feature_residuals = config.get("crop_feature_residuals", False)
        if self.crop_feature_at_start or self.crop_feature_at_bottleneck or self.crop_feature_residuals:
            self.crop_and_resize = CropAndResize()
        self.kernel_size = config.get("kernel_size", 3)
        self.only_pred_local = config.get("only_pred_local", True)
        self.in_channels = sum([utils.find_num_channels(config, scale, embeddings=True) for scale in self.scales])
        self.dropout = config["dropout"]
        self.start_filts = start_filts
        self.depth = config.get("depth", depth)
        
        self.scales_with_class = {scale: sum(True for feature in config[f"{scale}_features"] if feature in utils.get_class_features()) for scale in self.scales}
        embedding_config = config.copy()
        embedding_config["scales"] = self.scales
        self.num_class_feats, self.embedding = create_embeddings(embedding_config, "all")
        self.down_convs = []
        self.up_convs = []
        for i in range(self.depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts*(2**i)
            pooling = True if i < self.depth-1 else False
            down_conv = DownConv(ins, outs, pooling=pooling, dropout=self.dropout, kernel_size=self.kernel_size)
            self.down_convs.append(down_conv)
        for i in range(self.depth-1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs, up_mode=up_mode, dropout=self.dropout, kernel_size=self.kernel_size, add_residuals=self.add_residuals)
            self.up_convs.append(up_conv)
        self.local_final = conv1x1(outs, self.num_classes)
        if ("nearby" in self.scales) and (not self.only_pred_local):
            self.nearby_final = conv1x1(outs, self.num_classes)
        if ("context" in self.scales) and (not self.only_pred_local):
            self.context_final = conv1x1(outs, self.num_classes)
        if ("basin" in self.scales) and (not self.only_pred_local):
            self.basin_final = conv1x1(outs, self.num_classes)
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)

    def forward(self, data):
        
        if self.num_class_feats:
            embedding_index = 0
            embedded_data = []
            for scale in self.scales_with_class:
                if self.scales_with_class[scale]:
                    for index in range(self.scales_with_class[scale]):
                        embedded_data.append(self.embedding[embedding_index](torch.clip(data[f"{scale}_classes"][:, index, :, :], 0, 30).int()).squeeze(1).permute(0, 3, 1, 2))
                        embedding_index += 1
            x = torch.concat([torch.concat([data[f"{scale}_features"] for scale in self.scales], dim=1), torch.concat(embedded_data, dim=1)], dim=1)
        else:
            x = torch.concat([data[f"{scale}_features"] for scale in self.scales], dim=1)

        if self.crop_feature_at_start:
            x = self.crop_and_resize(x, self.scales[0], "local", 1)

        encoder_outs = []

        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)
        if self.crop_feature_at_bottleneck:
            x = self.crop_and_resize(x, self.scales[0], "local", self.depth)
        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i+2)]
            if self.crop_feature_residuals:
                before_pool = self.crop_and_resize(before_pool, self.scales[0], "local", depth=self.depth-i-1)
            x = module(before_pool, x)

        predictions = {"local_pred": self.local_final(x)}
        if ("nearby" in self.scales) and (not self.only_pred_local):
            predictions["nearby_pred"] = self.nearby_final(x)
        if ("context" in self.scales) and (not self.only_pred_local):
            predictions["context_pred"] = self.context_final(x)
        if ("basin" in self.scales) and (not self.only_pred_local):
            predictions["basin_pred"] = self.basin_final(x)
        return predictions
    
class BranchedUNet(nn.Module):
    def __init__(self, config, depth=5, start_filts=64):
        super(BranchedUNet, self).__init__()

        # configure the architecture
        self.only_pred_local = config.get("only_pred_local", True)
        self.residuals_all_scales = config.get("residuals_all_scales", True)
        self.use_attention = config.get("use_attention", True)
        self.hierarchical_attention = config.get("hierarchical_attention", True)
        self.scale_only_bottleneck = config.get("scale_only_bottleneck", False)
        self.crop_scales_at_bottleneck = config.get("crop_scales_at_bottleneck", False)
        self.crop_scales_residuals = config.get("crop_scales_residuals", False)
        self.fuse_aligned_scales_predictions = config.get("fuse_aligned_scales_predictions", False)
        if self.crop_scales_at_bottleneck or self.crop_scales_residuals or self.fuse_aligned_scales_predictions:
            self.crop_and_resize = CropAndResize()

        self.kernel_size = config.get("kernel_size", 3)
        self.dropout = config["dropout"]
        self.num_classes = config["num_classes"]
        self.depth = config.get("depth", depth)
        self.scales = config["scales"]
        self.in_channels = {scale: utils.find_num_channels(config, scale, embeddings=True) for scale in config["scales"]}
        self.start_filts = start_filts
        bottleneck_channels = start_filts * (2 ** (self.depth - 1))

        # set up basin encoder and decoder
        if "basin" in self.scales:
            self.basin_num_class_feats, self.basin_embedding = create_embeddings(config, "basin")
            self.basin_down_convs = []
            for i in range(self.depth):
                ins = self.in_channels["basin"] if i == 0 else outs
                outs = self.start_filts*(2**i)
                pooling = True if i < self.depth-1 else False
                down_conv = DownConv(ins, outs, pooling=pooling, dropout=self.dropout, kernel_size=self.kernel_size)
                self.basin_down_convs.append(down_conv)
            self.basin_weight = config.get("basin_feat_weight", 1)
            if not self.only_pred_local:
                self.basin_up_convs = []
                for i in range(self.depth-1):
                    ins = outs
                    outs = ins // 2
                    up_conv = UpConv(ins, outs, dropout=self.dropout, kernel_size=self.kernel_size)
                    self.basin_up_convs.append(up_conv)   
                self.basin_up_convs = nn.ModuleList(self.basin_up_convs)
                self.basin_final = conv1x1(outs, self.num_classes) #BclassesHW 
            self.basin_down_convs = nn.ModuleList(self.basin_down_convs)
            if not self.hierarchical_attention:
                self.local_attends_basin = CrossScaleAttention(config, query_scale="local", key_scale="basin", embed_dim=bottleneck_channels, depth=self.depth)

        # set up context encoder and decoder
        if "context" in self.scales:
            self.context_num_class_feats, self.context_embedding = create_embeddings(config, "context")
            self.context_down_convs = []
            for i in range(self.depth):
                ins = self.in_channels["context"] if i == 0 else outs
                outs = self.start_filts*(2**i)
                pooling = True if i < self.depth-1 else False
                down_conv = DownConv(ins, outs, pooling=pooling, dropout=self.dropout, kernel_size=self.kernel_size)
                self.context_down_convs.append(down_conv)
            self.context_down_convs = nn.ModuleList(self.context_down_convs)
            self.context_weight = config.get("context_feat_weight", 1)
            if not self.only_pred_local:
                self.context_up_convs = []
                for i in range(self.depth-1):
                    ins = outs
                    outs = ins // 2
                    up_conv = UpConv(ins, outs, num_scales=2 if ("basin" in self.scales) and (self.residuals_all_scales) else 1, dropout=self.dropout, kernel_size=self.kernel_size)
                    self.context_up_convs.append(up_conv)
                self.context_up_convs = nn.ModuleList(self.context_up_convs)
                self.context_final = conv1x1(outs, self.num_classes) #BclassesHW
            if not self.hierarchical_attention: # parallel
                self.local_attends_context = CrossScaleAttention(config, query_scale="local", key_scale="context" if not self.crop_scales_at_bottleneck else "local", embed_dim=bottleneck_channels, depth=self.depth)
                if (not self.only_pred_local) and ("basin" in self.scales):
                    self.context_attends_basin = CrossScaleAttention(config, query_scale="context", key_scale="basin", embed_dim=bottleneck_channels, depth=self.depth)
                    if not self.scale_only_bottleneck:
                        self.fuse_context_attention = AttentionFusion(bottleneck_channels, num_fusion_inputs=2)
            elif "basin" in self.scales: # hierarchical
                self.context_attends_higher = CrossScaleAttention(config, query_scale="context", key_scale="basin", embed_dim=bottleneck_channels, depth=self.depth) if self.use_attention else ConvFusion(bottleneck_channels)

        # set up nearby encoder and decoder
        if "nearby" in self.scales:
            self.nearby_num_class_feats, self.nearby_embedding = create_embeddings(config, "nearby")
            self.nearby_down_convs = []
            for i in range(self.depth):
                ins = self.in_channels["nearby"] if i == 0 else outs
                outs = self.start_filts*(2**i)
                pooling = True if i < self.depth-1 else False
                down_conv = DownConv(ins, outs, pooling=pooling, dropout=self.dropout, kernel_size=self.kernel_size)
                self.nearby_down_convs.append(down_conv)
            self.nearby_down_convs = nn.ModuleList(self.nearby_down_convs)
            self.nearby_weight = config.get("nearby_feat_weight", 1)
            num_other_scales = len(self.scales)-2
            if not self.only_pred_local:
                self.nearby_up_convs = []
                for i in range(self.depth-1):
                    ins = outs
                    outs = ins // 2
                    up_conv = UpConv(ins, outs, num_scales = 1 + num_other_scales if self.residuals_all_scales else 1, dropout=self.dropout, kernel_size=self.kernel_size)
                    self.nearby_up_convs.append(up_conv)
                self.nearby_up_convs = nn.ModuleList(self.nearby_up_convs)
                self.nearby_final = conv1x1(outs, self.num_classes) #BclassesHW
            if not self.hierarchical_attention: # parallel
                self.local_attends_nearby = CrossScaleAttention(config, query_scale="local", key_scale="nearby" if not self.crop_scales_at_bottleneck else "local", embed_dim=bottleneck_channels, depth=self.depth)
                if not self.only_pred_local:
                    if "basin" in self.scales:
                        self.nearby_attends_basin = CrossScaleAttention(config, query_scale="nearby", key_scale="basin", embed_dim=bottleneck_channels, depth=self.depth)
                    if "context" in self.scales:
                        self.nearby_attends_context = CrossScaleAttention(config, query_scale="nearby", key_scale="context" if not self.crop_scales_at_bottleneck else "nearby", embed_dim=bottleneck_channels, depth=self.depth)
                    self.fuse_nearby_attention = AttentionFusion(bottleneck_channels, num_fusion_inputs=len(self.scales)-2 if self.scale_only_bottleneck else len(self.scales)-1)
            else: # hierarchical
                if ("context" in self.scales) or ("basin" in self.scales):
                    self.nearby_attends_higher = CrossScaleAttention(config, query_scale="nearby", key_scale=self.scales[2] if not self.crop_scales_at_bottleneck else "nearby", embed_dim=bottleneck_channels, depth=self.depth) if self.use_attention else ConvFusion(bottleneck_channels)

        # set up local encoder and decoder
        self.local_down_convs = []
        self.local_up_convs = []
        self.local_num_class_feats, self.local_embedding = create_embeddings(config, "local")
        self.local_weight = config.get("local_feat_weight", 1)
        for i in range(self.depth):
            ins = self.in_channels["local"] if i == 0 else outs
            outs = self.start_filts*(2**i)
            pooling = True if i < self.depth-1 else False
            down_conv = DownConv(ins, outs, pooling=pooling, dropout=self.dropout, kernel_size=self.kernel_size)
            self.local_down_convs.append(down_conv)
        for i in range(self.depth-1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs, num_scales=len(self.scales) if self.residuals_all_scales else 1, dropout=self.dropout, kernel_size=self.kernel_size)
            self.local_up_convs.append(up_conv)
        self.local_down_convs = nn.ModuleList(self.local_down_convs)
        self.local_up_convs = nn.ModuleList(self.local_up_convs)
        self.local_final = conv1x1(outs, self.num_classes) #BclassesHW
        if self.hierarchical_attention:
            self.local_attends_higher = CrossScaleAttention(config, query_scale="local", key_scale=self.scales[1]  if not self.crop_scales_at_bottleneck else "local", embed_dim=bottleneck_channels, depth=self.depth) if self.use_attention else ConvFusion(bottleneck_channels)
        else:
            self.fuse_local_attention = AttentionFusion(bottleneck_channels, num_fusion_inputs=len(self.scales)-1 if self.scale_only_bottleneck else len(self.scales))
        if self.fuse_aligned_scales_predictions and not self.only_pred_local:
            self.fuse_predictions = conv3x3_1x1(feature_channels=self.start_filts, num_scales=len(self.scales), num_classes=self.num_classes)

    def forward(self, data):

        bottleneck_features = {}

        ############ ENCODER

        ### ENCODER BRANCHES FOR EACH SCALE

        if "basin" in self.scales:
            basin_encoder_outs = []
            if self.basin_num_class_feats:
                basin = torch.concat([data[f"basin_features"]] + [self.basin_embedding[index](torch.clip(data[f"basin_classes"][:, index, :, :], 0, 30).int()).squeeze(1).permute(0, 3, 1, 2) for index in range(self.basin_num_class_feats)], dim=1)
            else:
                basin = data[f"basin_features"]
            basin = basin * self.basin_weight
            for i, module in enumerate(self.basin_down_convs):
                basin, basin_before_pool = module(basin)
                basin_encoder_outs.append(basin_before_pool)
            bottleneck_features["basin"] = basin

        if "context" in self.scales:
            context_encoder_outs = []
            if self.context_num_class_feats:
                context = torch.concat([data[f"context_features"]] + [self.context_embedding[index](torch.clip(data[f"context_classes"][:, index, :, :], 0, 30).int()).squeeze(1).permute(0, 3, 1, 2) for index in range(self.context_num_class_feats)], dim=1)
            else:
                context = data[f"context_features"]
            context = context * self.context_weight
            for i, module in enumerate(self.context_down_convs):
                context, context_before_pool = module(context)
                context_encoder_outs.append(context_before_pool)
            bottleneck_features["context"] = context

        if "nearby" in self.scales:
            nearby_encoder_outs = []
            if self.nearby_num_class_feats:
                nearby = torch.concat([data[f"nearby_features"]] + [self.nearby_embedding[index](torch.clip(data[f"nearby_classes"][:, index, :, :], 0, 30).int()).squeeze(1).permute(0, 3, 1, 2) for index in range(self.nearby_num_class_feats)], dim=1)
            else:
                nearby = data[f"nearby_features"]
            nearby = nearby * self.nearby_weight
            for i, module in enumerate(self.nearby_down_convs):
                nearby, nearby_before_pool = module(nearby)
                nearby_encoder_outs.append(nearby_before_pool)
            bottleneck_features["nearby"] = nearby

        local_encoder_outs = []
        if self.local_num_class_feats:
             local = torch.concat([data[f"local_features"]] + [self.local_embedding[index](torch.clip(data[f"local_classes"][:, index, :, :], 0, 30).int()).squeeze(1).permute(0, 3, 1, 2) for index in range(self.local_num_class_feats)], dim=1)
        else:
            local = data[f"local_features"]
        local = local * self.local_weight
        for i, module in enumerate(self.local_down_convs):
            local, local_before_pool = module(local)
            local_encoder_outs.append(local_before_pool)
        bottleneck_features["local"] = local

        ############ BOTTLENECK

        ### HIERARCHICAL ATTENTION
        if self.hierarchical_attention:
            if "basin" in self.scales:
                if "context" in self.scales: # basin and context
                    context_attended_higher = self.context_attends_higher(bottleneck_features["context"], bottleneck_features["basin"])
                    if "nearby" in self.scales: # basin and context and nearby
                        nearby_attended_higher = self.nearby_attends_higher(bottleneck_features["nearby"], context_attended_higher)
                        local_attended_higher = self.local_attends_higher(bottleneck_features["local"], nearby_attended_higher)
                    else: # basin and context, no nearby
                        local_attended_higher = self.local_attends_higher(bottleneck_features["local"], context_attended_higher)
                elif "nearby" in self.scales: # basin and nearby, no context
                    nearby_attended_higher = self.nearby_attends_higher(bottleneck_features["nearby"], bottleneck_features["basin"])
                    local_attended_higher = self.local_attends_higher(bottleneck_features["local"], nearby_attended_higher)
                else: # basin only, no context or nearby
                    local_attended_higher = self.local_attends_higher(bottleneck_features["local"], bottleneck_features["basin"])

            elif "context" in self.scales: # no basin
                if "nearby" in self.scales: # context and nearby, no basin
                    if self.crop_scales_at_bottleneck:
                        nearby_attended_higher = self.nearby_attends_higher(bottleneck_features["nearby"], self.crop_and_resize(bottleneck_features["context"], "context", "nearby", self.depth))
                        local_attended_higher = self.local_attends_higher(bottleneck_features["local"], self.crop_and_resize(nearby_attended_higher, "nearby", "local", self.depth))
                    else:
                        nearby_attended_higher = self.nearby_attends_higher(bottleneck_features["nearby"], bottleneck_features["context"])
                        local_attended_higher = self.local_attends_higher(bottleneck_features["local"], nearby_attended_higher)
                else: # context only, no nearby or basin
                    if self.crop_scales_at_bottleneck:
                        local_attended_higher = self.local_attends_higher(bottleneck_features["local"], self.crop_and_resize(bottleneck_features["context"], "context", "local", self.depth))
                    else:
                        local_attended_higher = self.local_attends_higher(bottleneck_features["local"], bottleneck_features["context"])

            elif "nearby" in self.scales: # no basin or context
                if self.crop_scales_at_bottleneck:
                    local_attended_higher = self.local_attends_higher(bottleneck_features["local"], self.crop_and_resize(bottleneck_features["nearby"], "nearby", "local", self.depth))
                else:
                    local_attended_higher = self.local_attends_higher(bottleneck_features["local"], bottleneck_features["nearby"])

        ### PARALLEL ATTENTION
        else: 
            local_attended_higher = []
            if "basin" in self.scales:
                local_attended_higher.append(self.local_attends_basin(bottleneck_features["local"], bottleneck_features["basin"]))
            if "context" in self.scales:
                if self.crop_scales_at_bottleneck:
                    local_attended_higher.append(self.local_attends_context(bottleneck_features["local"], self.crop_and_resize(bottleneck_features["context"], "context", "local", self.depth)))
                else:
                    local_attended_higher.append(self.local_attends_context(bottleneck_features["local"], bottleneck_features["context"]))
                if not self.only_pred_local and "basin" in self.scales:
                    context_attended_higher = self.context_attends_basin(bottleneck_features["context"], bottleneck_features["basin"])
            if "nearby" in self.scales:
                if self.crop_scales_at_bottleneck:
                    local_attended_higher.append(self.local_attends_nearby(bottleneck_features["local"], self.crop_and_resize(bottleneck_features["nearby"], "nearby", "local", self.depth)))
                else:
                    local_attended_higher.append(self.local_attends_nearby(bottleneck_features["local"], bottleneck_features["nearby"]))
                if not self.only_pred_local:
                    nearby_attended_higher = []
                    if "basin" in self.scales:
                        nearby_attended_higher.append(self.nearby_attends_basin(bottleneck_features["nearby"], bottleneck_features["basin"]))
                    if "context" in self.scales:
                        if self.crop_scales_at_bottleneck:
                            nearby_attended_higher.append(self.nearby_attends_context(bottleneck_features["nearby"], self.crop_and_resize(bottleneck_features["context"], "context", "nearby", self.depth)))
                        else:
                            nearby_attended_higher.append(self.nearby_attends_context(bottleneck_features["nearby"], bottleneck_features["context"]))

            if not self.scale_only_bottleneck:
                local_attended_higher = [bottleneck_features["local"]] + local_attended_higher
            local_attended_higher = self.fuse_local_attention(torch.cat(local_attended_higher, dim=1)) if len(local_attended_higher) > 1 else torch.cat(local_attended_higher, dim=1)

            if not self.only_pred_local:
                if ("context" in self.scales) and ("basin" in self.scales) and (not self.scale_only_bottleneck):
                    context_attended_higher = self.fuse_context_attention(torch.cat([bottleneck_features["context"], context_attended_higher], dim=1))
                if ("nearby" in self.scales) and (("basin" in self.scales) or ("context" in self.scales)):
                    if not self.scale_only_bottleneck:
                        nearby_attended_higher = [bottleneck_features["nearby"]] + nearby_attended_higher
                    nearby_attended_higher = self.fuse_nearby_attention(torch.cat(nearby_attended_higher, dim=1)) if len(nearby_attended_higher) > 1 else torch.cat(nearby_attended_higher, dim=1)

        ############ DECODER

        ### SEPARATE DECODER BRANCHES PER SCALE
        if not self.only_pred_local:

            if self.fuse_aligned_scales_predictions:
                aligned_scales_predictions = []

            if "basin" in self.scales:
                x = bottleneck_features["basin"]
                for i, module in enumerate(self.basin_up_convs):
                    basin_before_pool = basin_encoder_outs[-(i+2)]
                    x = module(basin_before_pool, x)
                basin_pred = self.basin_final(x)

            if "context" in self.scales:
                if (self.residuals_all_scales) and ("basin" in self.scales):
                    x = context_attended_higher
                    for i, module in enumerate(self.context_up_convs):
                        up_features = []
                        basin_before_pool = basin_encoder_outs[-(i+2)]
                        up_features.append(basin_before_pool)
                        context_before_pool = context_encoder_outs[-(i+2)]
                        up_features.append(context_before_pool)
                        before_pool = torch.cat(up_features, dim=1)
                        x = module(before_pool, x)
                    if self.fuse_aligned_scales_predictions:
                        aligned_scales_predictions.append(self.crop_and_resize(x, "context", "local", 1))
                    context_pred = self.context_final(x)
                else:
                    x = context_attended_higher if "basin" in self.scales else bottleneck_features["context"]
                    for i, module in enumerate(self.context_up_convs):
                        context_before_pool = context_encoder_outs[-(i+2)]
                        x = module(context_before_pool, x)
                    if self.fuse_aligned_scales_predictions:
                        aligned_scales_predictions.append(self.crop_and_resize(x, "context", "local", 1))
                    context_pred = self.context_final(x)

            if "nearby" in self.scales:
                if (self.residuals_all_scales) and (("basin" in self.scales) or ("context" in self.scales)):
                    x = nearby_attended_higher
                    for i, module in enumerate(self.nearby_up_convs):
                        up_features = []
                        if "basin" in self.scales:
                            basin_before_pool = basin_encoder_outs[-(i+2)]
                            up_features.append(basin_before_pool)
                        if "context" in self.scales:
                            context_before_pool = context_encoder_outs[-(i+2)]
                            if self.crop_scales_residuals:
                                context_before_pool = self.crop_and_resize(context_before_pool, to_crop="context", cropped_to="nearby", depth=self.depth-i-1)
                            up_features.append(context_before_pool)
                        nearby_before_pool = nearby_encoder_outs[-(i+2)]
                        up_features.append(nearby_before_pool)
                        before_pool = torch.cat(up_features, dim=1)
                        x = module(before_pool, x)
                    if self.fuse_aligned_scales_predictions:
                        aligned_scales_predictions.append(self.crop_and_resize(x, "nearby", "local", 1))
                    nearby_pred = self.nearby_final(x)

                else:
                    x = nearby_attended_higher if ("basin" in self.scales) or ("context" in self.scales) else bottleneck_features["nearby"]
                    for i, module in enumerate(self.nearby_up_convs):
                        nearby_before_pool = nearby_encoder_outs[-(i+2)]
                        x = module(nearby_before_pool, x)
                    if self.fuse_aligned_scales_predictions:
                        aligned_scales_predictions.append(self.crop_and_resize(x, "nearby", "local", 1))
                    nearby_pred = self.nearby_final(x)

        ### LOCAL DECODER BRANCH

        ## RESIDUALS FROM ALL SCALES
        if self.residuals_all_scales:
            x = local_attended_higher
            for i, module in enumerate(self.local_up_convs):
                up_features = []
                if "basin" in self.scales:
                    basin_before_pool = basin_encoder_outs[-(i+2)]
                    up_features.append(basin_before_pool)
                if "context" in self.scales:
                    context_before_pool = context_encoder_outs[-(i+2)]
                    if self.crop_scales_residuals:
                        context_before_pool = self.crop_and_resize(context_before_pool, to_crop="context", cropped_to="local", depth=self.depth-i-1)
                    up_features.append(context_before_pool)
                if "nearby" in self.scales:
                    nearby_before_pool = nearby_encoder_outs[-(i+2)]
                    if self.crop_scales_residuals:
                        nearby_before_pool = self.crop_and_resize(nearby_before_pool, to_crop="nearby", cropped_to="local", depth=self.depth-i-1)
                    up_features.append(nearby_before_pool)
                local_before_pool = local_encoder_outs[-(i+2)]
                up_features.append(local_before_pool)
                before_pool = torch.cat(up_features, dim=1)
                x = module(before_pool, x)
            if self.fuse_aligned_scales_predictions:
                aligned_scales_predictions.append(x)
            local_pred = self.local_final(x)

        ## ONLY LOCAL RESIDUALS
        else:
            x = local_attended_higher
            for i, module in enumerate(self.local_up_convs):
                local_before_pool = local_encoder_outs[-(i+2)]
                x = module(local_before_pool, x)
            if self.fuse_aligned_scales_predictions:
                aligned_scales_predictions.append(x)
            local_pred = self.local_final(x)
                
        ### FINAL SEGMENTATION HEAD

        if self.fuse_aligned_scales_predictions:
            local_pred = self.fuse_predictions(torch.cat(aligned_scales_predictions, dim=1))

        predictions = {"local_pred": local_pred}
        if ("nearby" in self.scales) and (not self.only_pred_local):
            predictions["nearby_pred"] = nearby_pred
        if ("context" in self.scales) and (not self.only_pred_local):
            predictions["context_pred"] = context_pred
        if ("basin" in self.scales) and (not self.only_pred_local):
            predictions["basin_pred"] = basin_pred
        return predictions
    
class ClassificationBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ClassificationBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )
            
    def forward(self, x):
        out = self.block(x)
        out = out + self.shortcut(x)
        out = F.relu(out)
        
        return out

class BasicResNet(nn.Module):
    def __init__(self, config):
        super(BasicResNet, self).__init__()

        self.scales = config["scales"]
        if config.get("exclude_scales", False):
            self.scales = [scale for scale in self.scales if scale not in config["exclude_scales"]]
        self.num_classes = 3
        self.only_pred_local = config.get("only_pred_local", True)
        self.in_channels = sum([utils.find_num_channels(config, scale, embeddings=True) for scale in self.scales])
        self.scales_with_class = {scale: sum(True for feature in config[f"{scale}_features"] if feature in utils.get_class_features()) for scale in self.scales}
        embedding_config = config.copy()
        embedding_config["scales"] = self.scales
        self.num_class_feats, self.embedding = create_embeddings(embedding_config, "all")

        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False), 
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.layer1 = self.make_layer(ClassificationBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ClassificationBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ClassificationBlock, 256, 2, stride=2)        
        self.layer4 = self.make_layer(ClassificationBlock, 512, 2, stride=2)        
        self.local_final = nn.Linear(512, self.num_classes)
        if ("nearby" in self.scales) and (not self.only_pred_local):
            self.nearby_final = nn.Linear(512, self.num_classes)
        if ("context" in self.scales) and (not self.only_pred_local):
            self.context_final = nn.Linear(512, self.num_classes)
        if ("basin" in self.scales) and (not self.only_pred_local):
            self.basin_final = nn.Linear(512, self.num_classes)
        
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)
    
    def forward(self, data):

        if self.num_class_feats:
            embedding_index = 0
            embedded_data = []
            for scale in self.scales_with_class:
                if self.scales_with_class[scale]:
                    for index in range(self.scales_with_class[scale]):
                        embedded_data.append(self.embedding[embedding_index](torch.clip(data[f"{scale}_classes"][:, index, :, :], 0, 30).int()).squeeze(1).permute(0, 3, 1, 2))
                        embedding_index += 1
            x = torch.concat([torch.concat([data[f"{scale}_features"] for scale in self.scales], dim=1), torch.concat(embedded_data, dim=1)], dim=1)
        else:
            x = torch.concat([data[f"{scale}_features"] for scale in self.scales], dim=1)
    
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)

        predictions = {"local_pred": self.local_final(out)}
        if ("nearby" in self.scales) and (not self.only_pred_local):
            predictions["nearby_pred"] = self.nearby_final(out)
        if ("context" in self.scales) and (not self.only_pred_local):
            predictions["context_pred"] = self.context_final(out)
        if ("basin" in self.scales) and (not self.only_pred_local):
            predictions["basin_pred"] = self.basin_final(out)

        return predictions
    
class BranchedResNet(nn.Module):
    def __init__(self, config):
        super(BranchedResNet, self).__init__()

        self.scales = config["scales"]
        self.only_pred_local = config.get("only_pred_local", True)
        self.hierarchical_attention = config.get("hierarchical_attention", True)
        self.scale_only_bottleneck = config.get("scale_only_bottleneck", False)
        self.crop_scales_at_bottleneck = config.get("crop_scales_at_bottleneck", False)
        self.num_classes = config["num_classes"]
        if self.crop_scales_at_bottleneck:
            self.crop_and_resize = CropAndResize()
        self.in_channels = {scale: utils.find_num_channels(config, scale, embeddings=True) for scale in config["scales"]}
        self.inchannel = {"local": 64}
        self.depth = config.get("depth", 6)
        layer4_stride = 1 if self.depth < 6 else 2
        layer3_stride = 1 if self.depth < 5 else 2

        if "basin" in self.scales:
            self.basin_num_class_feats, self.basin_embedding = create_embeddings(config, "basin")
            self.basin_weight = config.get("basin_feat_weight", 1)
            self.inchannel["basin"] = 64
            self.basin_conv1 = nn.Sequential(
                nn.Conv2d(self.in_channels["basin"], 64, kernel_size=7, stride=2, padding=3, bias=False), 
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            self.basin_layer1 = self.make_layer(ClassificationBlock, 64, 2, stride=1, scale="basin")
            self.basin_layer2 = self.make_layer(ClassificationBlock, 128, 2, stride=2, scale="basin")
            self.basin_layer3 = self.make_layer(ClassificationBlock, 256, 2, stride=layer3_stride, scale="basin")        
            self.basin_layer4 = self.make_layer(ClassificationBlock, 512, 2, stride=layer4_stride, scale="basin")     
            self.basin_final = nn.Linear(512, self.num_classes)
            if not self.hierarchical_attention:
                self.local_attends_basin = CrossScaleAttention(config, query_scale="local", key_scale="basin", depth=self.depth)

        if "context" in self.scales:
            self.context_num_class_feats, self.context_embedding = create_embeddings(config, "context")
            self.context_weight = config.get("context_feat_weight", 1)
            self.inchannel["context"] = 64
            self.context_conv1 = nn.Sequential(
                nn.Conv2d(self.in_channels["context"], 64, kernel_size=7, stride=2, padding=3, bias=False), 
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            self.context_layer1 = self.make_layer(ClassificationBlock, 64, 2, stride=1, scale="context")
            self.context_layer2 = self.make_layer(ClassificationBlock, 128, 2, stride=2, scale="context")
            self.context_layer3 = self.make_layer(ClassificationBlock, 256, 2, stride=layer3_stride, scale="context")        
            self.context_layer4 = self.make_layer(ClassificationBlock, 512, 2, stride=layer4_stride, scale="context")     
            self.context_final = nn.Linear(512, self.num_classes)
            if not self.hierarchical_attention: # parallel
                self.local_attends_context = CrossScaleAttention(config, query_scale="local", key_scale="context" if not self.crop_scales_at_bottleneck else "local", depth=self.depth)
                if (not self.only_pred_local) and ("basin" in self.scales):
                    self.context_attends_basin = CrossScaleAttention(config, query_scale="context", key_scale="basin", depth=self.depth)
                    if not self.scale_only_bottleneck:
                        self.fuse_context_attention = AttentionFusion(num_fusion_inputs=2)
            elif "basin" in self.scales: # hierarchical
                self.context_attends_higher = CrossScaleAttention(config, query_scale="context", key_scale="basin", depth=self.depth)

        if "nearby" in self.scales:
            self.nearby_num_class_feats, self.nearby_embedding = create_embeddings(config, "nearby")
            self.nearby_weight = config.get("nearby_feat_weight", 1)
            self.inchannel["nearby"] = 64
            self.nearby_conv1 = nn.Sequential(
                nn.Conv2d(self.in_channels["nearby"], 64, kernel_size=7, stride=2, padding=3, bias=False), 
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            self.nearby_layer1 = self.make_layer(ClassificationBlock, 64, 2, stride=1, scale="nearby")
            self.nearby_layer2 = self.make_layer(ClassificationBlock, 128, 2, stride=2, scale="nearby")
            self.nearby_layer3 = self.make_layer(ClassificationBlock, 256, 2, stride=layer3_stride, scale="nearby")        
            self.nearby_layer4 = self.make_layer(ClassificationBlock, 512, 2, stride=layer4_stride, scale="nearby")     
            self.nearby_final = nn.Linear(512, self.num_classes)

            if not self.hierarchical_attention: # parallel
                self.local_attends_nearby = CrossScaleAttention(config, query_scale="local", key_scale="nearby" if not self.crop_scales_at_bottleneck else "local", depth=self.depth)
                if not self.only_pred_local:
                    if "basin" in self.scales:
                        self.nearby_attends_basin = CrossScaleAttention(config, query_scale="nearby", key_scale="basin", depth=self.depth)
                    if "context" in self.scales:
                        self.nearby_attends_context = CrossScaleAttention(config, query_scale="nearby", key_scale="context" if not self.crop_scales_at_bottleneck else "nearby", depth=self.depth)
                    self.fuse_nearby_attention = AttentionFusion(num_fusion_inputs=len(self.scales)-2 if self.scale_only_bottleneck else len(self.scales)-1)
            else: # hierarchical
                if ("context" in self.scales) or ("basin" in self.scales):
                    self.nearby_attends_higher = CrossScaleAttention(config, query_scale="nearby", key_scale=self.scales[2] if not self.crop_scales_at_bottleneck else "nearby", depth=self.depth)

        self.local_num_class_feats, self.local_embedding = create_embeddings(config, "local")
        self.local_weight = config.get("local_feat_weight", 1)
        self.local_inchannel = 64
        self.local_conv1 = nn.Sequential(
            nn.Conv2d(self.in_channels["local"], 64, kernel_size=7, stride=2, padding=3, bias=False), 
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.local_layer1 = self.make_layer(ClassificationBlock, 64, 2, stride=1, scale="local")    # 64 x 32 x 32
        self.local_layer2 = self.make_layer(ClassificationBlock, 128, 2, stride=2, scale="local")   # 128 x 32 x 32
        self.local_layer3 = self.make_layer(ClassificationBlock, 256, 2, stride=layer3_stride, scale="local")   # 256 x 16 x 16
        self.local_layer4 = self.make_layer(ClassificationBlock, 512, 2, stride=layer4_stride, scale="local")   # 512 x 8 x 8  
        self.local_final = nn.Linear(512, self.num_classes)
        if self.hierarchical_attention:
            self.local_attends_higher = CrossScaleAttention(config, query_scale="local", key_scale=self.scales[1] if not self.crop_scales_at_bottleneck else "local", depth=self.depth)
        else:
            self.fuse_local_attention = AttentionFusion(num_fusion_inputs=len(self.scales)-1 if self.scale_only_bottleneck else len(self.scales))

        if ("nearby" in self.scales) and (not self.only_pred_local):
            self.nearby_final = nn.Linear(512, self.num_classes)
        if ("context" in self.scales) and (not self.only_pred_local):
            self.context_final = nn.Linear(512, self.num_classes)
        if ("basin" in self.scales) and (not self.only_pred_local):
            self.basin_final = nn.Linear(512, self.num_classes)
        
    def make_layer(self, block, channels, num_blocks, stride, scale):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel[scale], channels, stride))
            self.inchannel[scale] = channels
        return nn.Sequential(*layers)
    
    def forward(self, data):

        bottleneck_features = {}

        if "basin" in self.scales:
            if self.basin_num_class_feats:
                basin = torch.concat([data[f"basin_features"]] + [self.basin_embedding[index](torch.clip(data[f"basin_classes"][:, index, :, :], 0, 30).int()).squeeze(1).permute(0, 3, 1, 2) for index in range(self.basin_num_class_feats)], dim=1)
            else:
                basin = data[f"basin_features"]
            basin = basin * self.basin_weight

            basin = self.basin_conv1(basin)
            basin = self.basin_layer1(basin)
            basin = self.basin_layer2(basin)
            basin = self.basin_layer3(basin)
            basin = self.basin_layer4(basin)
            bottleneck_features["basin"] = basin

        if "context" in self.scales:

            if self.context_num_class_feats:
                context = torch.concat([data[f"context_features"]] + [self.context_embedding[index](torch.clip(data[f"context_classes"][:, index, :, :], 0, 30).int()).squeeze(1).permute(0, 3, 1, 2) for index in range(self.context_num_class_feats)], dim=1)
            else:
                context = data[f"context_features"]
            context = context * self.context_weight
            context = self.context_conv1(context)
            context = self.context_layer1(context)
            context = self.context_layer2(context)
            context = self.context_layer3(context)
            context = self.context_layer4(context)
            bottleneck_features["context"] = context

        if "nearby" in self.scales:

            if self.nearby_num_class_feats:
                nearby = torch.concat([data[f"nearby_features"]] + [self.nearby_embedding[index](torch.clip(data[f"nearby_classes"][:, index, :, :], 0, 30).int()).squeeze(1).permute(0, 3, 1, 2) for index in range(self.nearby_num_class_feats)], dim=1)
            else:
                nearby = data[f"nearby_features"]
            nearby = nearby * self.nearby_weight
            nearby = self.nearby_conv1(nearby)
            nearby = self.nearby_layer1(nearby)
            nearby = self.nearby_layer2(nearby)
            nearby = self.nearby_layer3(nearby)
            nearby = self.nearby_layer4(nearby)
            bottleneck_features["nearby"] = nearby

        if self.local_num_class_feats:
             local = torch.concat([data[f"local_features"]] + [self.local_embedding[index](torch.clip(data[f"local_classes"][:, index, :, :], 0, 30).int()).squeeze(1).permute(0, 3, 1, 2) for index in range(self.local_num_class_feats)], dim=1)
        else:
            local = data[f"local_features"]
        local = local * self.local_weight

        local = self.local_conv1(local)
        local = self.local_layer1(local)
        local = self.local_layer2(local)
        local = self.local_layer3(local)
        local = self.local_layer4(local)
        bottleneck_features["local"] = local
        
        if self.hierarchical_attention:
            if "basin" in self.scales:
                if "context" in self.scales: # basin and context
                    context_attended_higher = self.context_attends_higher(bottleneck_features["context"], bottleneck_features["basin"])
                    if "nearby" in self.scales: # basin and context and nearby
                        nearby_attended_higher = self.nearby_attends_higher(bottleneck_features["nearby"], context_attended_higher)
                        local_attended_higher = self.local_attends_higher(bottleneck_features["local"], nearby_attended_higher)
                    else: # basin and context, no nearby
                        local_attended_higher = self.local_attends_higher(bottleneck_features["local"], context_attended_higher)
                elif "nearby" in self.scales: # basin and nearby, no context
                    nearby_attended_higher = self.nearby_attends_higher(bottleneck_features["nearby"], bottleneck_features["basin"])
                    local_attended_higher = self.local_attends_higher(bottleneck_features["local"], nearby_attended_higher)
                else: # basin only, no context or nearby
                    local_attended_higher = self.local_attends_higher(bottleneck_features["local"], bottleneck_features["basin"])

            elif "context" in self.scales: # no basin
                if "nearby" in self.scales: # context and nearby, no basin
                    if self.crop_scales_at_bottleneck:
                        nearby_attended_higher = self.nearby_attends_higher(bottleneck_features["nearby"], self.crop_and_resize(bottleneck_features["context"], "context", "nearby", self.depth))
                        local_attended_higher = self.local_attends_higher(bottleneck_features["local"], self.crop_and_resize(nearby_attended_higher, "nearby", "local", self.depth))
                    else:
                        nearby_attended_higher = self.nearby_attends_higher(bottleneck_features["nearby"], bottleneck_features["context"])
                        local_attended_higher = self.local_attends_higher(bottleneck_features["local"], nearby_attended_higher)
                else: # context only, no nearby or basin
                    if self.crop_scales_at_bottleneck:
                        local_attended_higher = self.local_attends_higher(bottleneck_features["local"], self.crop_and_resize(bottleneck_features["context"], "context", "local", self.depth))
                    else:
                        local_attended_higher = self.local_attends_higher(bottleneck_features["local"], bottleneck_features["context"])

            elif "nearby" in self.scales: # no basin or context
                if self.crop_scales_at_bottleneck:
                    local_attended_higher = self.local_attends_higher(bottleneck_features["local"], self.crop_and_resize(bottleneck_features["nearby"], "nearby", "local", self.depth))
                else:
                    local_attended_higher = self.local_attends_higher(bottleneck_features["local"], bottleneck_features["nearby"])

        else: 
            local_attended_higher = []
            if "basin" in self.scales:
                local_attended_higher.append(self.local_attends_basin(bottleneck_features["local"], bottleneck_features["basin"]))
            if "context" in self.scales:
                if self.crop_scales_at_bottleneck:
                    local_attended_higher.append(self.local_attends_context(bottleneck_features["local"], self.crop_and_resize(bottleneck_features["context"], "context", "local", self.depth)))
                else:
                    local_attended_higher.append(self.local_attends_context(bottleneck_features["local"], bottleneck_features["context"]))
                if not self.only_pred_local and "basin" in self.scales:
                    context_attended_higher = self.context_attends_basin(bottleneck_features["context"], bottleneck_features["basin"])
            if "nearby" in self.scales:
                if self.crop_scales_at_bottleneck:
                    local_attended_higher.append(self.local_attends_nearby(bottleneck_features["local"], self.crop_and_resize(bottleneck_features["nearby"], "nearby", "local", self.depth)))
                else:
                    local_attended_higher.append(self.local_attends_nearby(bottleneck_features["local"], bottleneck_features["nearby"]))
                if not self.only_pred_local:
                    nearby_attended_higher = []
                    if "basin" in self.scales:
                        nearby_attended_higher.append(self.nearby_attends_basin(bottleneck_features["nearby"], bottleneck_features["basin"]))
                    if "context" in self.scales:
                        if self.crop_scales_at_bottleneck:
                            nearby_attended_higher.append(self.nearby_attends_context(bottleneck_features["nearby"], self.crop_and_resize(bottleneck_features["context"], "context", "nearby", self.depth)))
                        else:
                            nearby_attended_higher.append(self.nearby_attends_context(bottleneck_features["nearby"], bottleneck_features["context"]))

            if not self.scale_only_bottleneck:
                local_attended_higher = [bottleneck_features["local"]] + local_attended_higher
            local_attended_higher = self.fuse_local_attention(torch.cat(local_attended_higher, dim=1)) if len(local_attended_higher) > 1 else torch.cat(local_attended_higher, dim=1)

            if not self.only_pred_local:
                if ("context" in self.scales) and ("basin" in self.scales) and (not self.scale_only_bottleneck):
                    context_attended_higher = self.fuse_context_attention(torch.cat([bottleneck_features["context"], context_attended_higher], dim=1))
                if ("nearby" in self.scales) and (("basin" in self.scales) or ("context" in self.scales)):
                    if not self.scale_only_bottleneck:
                        nearby_attended_higher = [bottleneck_features["nearby"]] + nearby_attended_higher
                    nearby_attended_higher = self.fuse_nearby_attention(torch.cat(nearby_attended_higher, dim=1)) if len(nearby_attended_higher) > 1 else torch.cat(nearby_attended_higher, dim=1)

        if not self.only_pred_local:
            if "basin" in self.scales:
                x = bottleneck_features["basin"]
                x = F.adaptive_avg_pool2d(x, output_size=1)
                x = torch.flatten(x, 1)
                basin_pred = self.basin_final(x)

            if "context" in self.scales:
                x = context_attended_higher if "basin" in self.scales else bottleneck_features["context"]
                x = F.adaptive_avg_pool2d(x, output_size=1)
                x = torch.flatten(x, 1) 
                context_pred = self.context_final(x)

            if "nearby" in self.scales:
                x = nearby_attended_higher if ("basin" in self.scales) or ("context" in self.scales) else bottleneck_features["nearby"]
                x = F.adaptive_avg_pool2d(x, output_size=1)
                x = torch.flatten(x, 1)
                nearby_pred = self.nearby_final(x)

        x = local_attended_higher
        x = F.adaptive_avg_pool2d(x, output_size=1)
        x = torch.flatten(x, 1)
        local_pred = self.local_final(x)

        predictions = {"local_pred": local_pred}
        if ("nearby" in self.scales) and (not self.only_pred_local):
            predictions["nearby_pred"] = nearby_pred
        if ("context" in self.scales) and (not self.only_pred_local):
            predictions["context_pred"] = context_pred
        if ("basin" in self.scales) and (not self.only_pred_local):
            predictions["basin_pred"] = basin_pred

        return predictions
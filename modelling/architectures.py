import torch
import torch.nn as nn
import torch.nn.functional as F
import modelling.utils as utils
import math

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
    
class MultiHeadAttention(nn.Module):
    def __init__(self, in_dim, embed_dim, num_heads=4, out_dim=None, attn_dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.out_dim = out_dim if out_dim is not None else in_dim
        self.q_proj = nn.Conv2d(in_dim, embed_dim, kernel_size=1)
        self.k_proj = nn.Conv2d(in_dim, embed_dim, kernel_size=1)
        self.v_proj = nn.Conv2d(in_dim, embed_dim, kernel_size=1)
        self.out_proj = nn.Conv2d(embed_dim, self.out_dim, kernel_size=1)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.scale = math.sqrt(self.head_dim)
    def flatten_hw(self, x):
        B, C, H, W = x.shape
        return x.view(B, C, H*W).transpose(1, 2)
    def unflatten_hw(self, x, H, W):
        return x.transpose(1, 2).view(x.size(0), x.size(2), H, W)
    def forward_proj(self, proj, x):
        B, _, H, W = x.shape
        p = proj(x)            
        p = p.view(B, self.num_heads, self.head_dim, H*W)
        p = p.permute(0, 1, 3, 2)
        return p, H, W
    def forward(self, Q_input, K_input, V_input, mask=None):
        B = Q_input.size(0)
        Qh, H, W = self.forward_proj(self.q_proj, Q_input)
        Kh, _, _ = self.forward_proj(self.k_proj, K_input)
        Vh, _, _ = self.forward_proj(self.v_proj, V_input)
        scores = torch.matmul(Qh, Kh.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask==0, float("-1e9"))
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, Vh)
        out = out.permute(0,2,1,3).contiguous().view(B, H*W, self.embed_dim)
        out = out.transpose(1,2).contiguous().view(B, self.embed_dim, H, W)
        out = self.out_proj(out)
        return out

class AttentionFusion(nn.Module):
    def __init__(self, ch_per_branch, embed_dim=256, num_heads=4, out_ch=None, share_kv_proj=True, downsample_factor=1, scales=("basin", "context", "local")):
        super().__init__()
        assert isinstance(scales, (list, tuple))
        self.ch = ch_per_branch
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.out_ch = out_ch if out_ch is not None else self.ch
        self.downsample_factor = downsample_factor
        self.scales = list(scales)
        self.num_branches = len(self.scales)
        if share_kv_proj:
            self.kv_proj = nn.Conv2d(self.ch * self.num_branches, self.embed_dim, kernel_size=1)
            self.vv_proj = nn.Conv2d(self.ch * self.num_branches, self.embed_dim, kernel_size=1)
        else:
            self.kv_proj = None
        self.q_projs = nn.ModuleDict({
            s: nn.Conv2d(self.ch, self.embed_dim, kernel_size=1) for s in self.scales})
        self.attn = MultiHeadAttention(in_dim=self.embed_dim,
                                       embed_dim=self.embed_dim,
                                       num_heads=self.num_heads,
                                       out_dim=self.out_ch)
        self.out_projs = nn.ModuleDict({s: conv1x1(self.out_ch, self.out_ch) for s in self.scales})
        self.res_proj = conv1x1(self.ch * self.num_branches, self.out_ch)
    def forward(self, branches):
        if self.downsample_factor > 1:
            branches_ds = {s: F.avg_pool2d(branches[s], kernel_size=self.downsample_factor)
                           for s in self.scales}
        else:
            branches_ds = {s: branches[s] for s in self.scales}
        concat_ds = torch.cat([branches_ds[s] for s in self.scales], dim=1)   # B, num_branches*C, H', W'
        if self.kv_proj is not None:
            K_in = self.kv_proj(concat_ds)
            V_in = self.vv_proj(concat_ds)
        else:
            K_in = concat_ds
            V_in = concat_ds
        attn_outputs = {}
        for s in self.scales:
            Qs = self.q_projs[s](branches_ds[s])
            out_s = self.attn(Qs, K_in, V_in)
            if self.downsample_factor > 1:
                out_s = F.interpolate(out_s, size=(branches[s].shape[2], branches[s].shape[3]), mode='bilinear', align_corners=False)
            attn_outputs[s] = out_s
        res = self.res_proj(torch.cat([branches[s] for s in self.scales], dim=1))
        fused = {}
        for s in self.scales:
            fused_s = self.out_projs[s](attn_outputs[s]) + res
            fused[s] = fused_s
        return fused

class DirectedAttentionFusion(nn.Module):
    def __init__(self, ch_per_branch, sources, embed_dim=256, num_heads=4, out_ch=None, share_kv_proj=True, downsample_factor=1):
        super().__init__()
        self.ch = ch_per_branch
        self.sources = list(sources)
        self.num_sources = len(self.sources)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.out_ch = out_ch if out_ch is not None else self.ch
        self.downsample_factor = downsample_factor
        if share_kv_proj:
            self.kv_proj = nn.Conv2d(self.ch * self.num_sources, self.embed_dim, kernel_size=1)
            self.vv_proj = nn.Conv2d(self.ch * self.num_sources, self.embed_dim, kernel_size=1)
        else:
            self.kv_proj = None
        self.q_proj = nn.Conv2d(self.ch, self.embed_dim, kernel_size=1)
        self.attn = MultiHeadAttention(in_dim=self.embed_dim, embed_dim=self.embed_dim, num_heads=self.num_heads, out_dim=self.out_ch)
        self.out_proj = conv1x1(self.out_ch, self.out_ch)
        self.res_proj = conv1x1(self.ch * self.num_sources, self.out_ch)
    def forward(self, target_tensor, branches_dict):
        if self.downsample_factor > 1:
            target_ds = F.avg_pool2d(target_tensor, kernel_size=self.downsample_factor)
            sources_ds = [F.avg_pool2d(branches_dict[s], kernel_size=self.downsample_factor) for s in self.sources]
        else:
            target_ds = target_tensor
            sources_ds = [branches_dict[s] for s in self.sources]
        concat_ds = torch.cat(sources_ds, dim=1)
        if self.kv_proj is not None:
            K_in = self.kv_proj(concat_ds)
            V_in = self.vv_proj(concat_ds)
        else:
            K_in = concat_ds
            V_in = concat_ds
        Q = self.q_proj(target_ds)
        attn_out = self.attn(Q, K_in, V_in)
        if self.downsample_factor > 1:
            attn_out = F.interpolate(attn_out, size=(target_tensor.shape[2], target_tensor.shape[3]),
                                     mode='bilinear', align_corners=False)
        res = self.res_proj(torch.cat([branches_dict[s] for s in self.sources], dim=1))
        fused = self.out_proj(attn_out) + res
        return fused

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
        self.use_attention_fusion = config["use_attention_fusion"]
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
        
        fusion_channels = start_filts * (2 ** (depth - 1))
        if self.use_attention_fusion:
            self.fusion = AttentionFusion(ch_per_branch=fusion_channels, out_ch=fusion_channels, scales=self.scales)
            self.merge = conv1x1(fusion_channels * len(self.scales), fusion_channels)
        else:
            self.fusion = nn.Conv2d(fusion_channels * len(self.scales), fusion_channels, kernel_size=1)
        
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

        if self.use_attention_fusion:
            branches = {scale_branch: final_features[scale_branch] for scale_branch in self.scales}
            fused_x = self.fusion(branches)
            x = self.merge(torch.cat([fused_x[scale_branch] for scale_branch in self.scales], dim=1))
        else:
            concat = torch.cat(list(final_features.values()), dim=1)
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
        self.use_attention_fusion = config["use_attention_fusion"]
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
        
        fusion_channels = start_filts * (2 ** (depth - 1))
        if self.use_attention_fusion:
            self.fusion = AttentionFusion(ch_per_branch=fusion_channels, out_ch=fusion_channels, scales=self.scales)
        else:
            self.fusion = nn.Conv2d(fusion_channels * len(self.scales), fusion_channels, kernel_size=1)

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

        if self.use_attention_fusion:
            fused = self.fusion(final_features)
        else:
            concat = torch.cat([final_features[scale_name] for scale_name in self.scales], dim=1)
            concat = self.fusion(concat)

        if "basin" in self.scales:
            x = fused["basin"] if self.use_attention_fusion else concat
            for i, module in enumerate(self.basin_up_convs):
                basin_before_pool = basin_encoder_outs[-(i+2)]
                x = module(basin_before_pool, x)
            basin_pred = self.basin_final(x)

        if "context" in self.scales:
            x = fused["context"] if self.use_attention_fusion else concat
            for i, module in enumerate(self.context_up_convs):
                context_before_pool = context_encoder_outs[-(i+2)]
                x = module(context_before_pool, x)
            context_pred = self.context_final(x)

        x = fused["local"] if self.use_attention_fusion else concat
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

        fusion_in_ch = start_filts * (2 ** (depth - 1))
        self.basin_context_attn = None
        self.local_scale_attn = None
        if "context" in self.scales: # context fusion: context may attend to basin and context (but not local)
            if "basin" in self.scales: # build attention: sources = ['basin','context'], target will be context
                self.basin_context_attn = DirectedAttentionFusion(ch_per_branch=fusion_in_ch, sources=['basin', 'context'], out_ch=fusion_in_ch)
            else:
                self.basin_context_attn = None
            self.basin_context_fusion = nn.Conv2d(fusion_in_ch * 2, fusion_in_ch, kernel_size=1)
        else:
            self.basin_context_fusion = None
            self.basin_context_attn = None
        sources_for_local = [s for s in self.scales if s != 'local']
        if len(sources_for_local) >= 1:
            sources_for_local = sources_for_local + ['local']   # include local in K/V
            self.local_scale_attn = DirectedAttentionFusion(ch_per_branch=fusion_in_ch, sources=sources_for_local,out_ch=fusion_in_ch)
            self.local_scale_fusion = nn.Conv2d(fusion_in_ch * len(sources_for_local), fusion_in_ch, kernel_size=1)
        else:
            self.local_scale_attn = None
            self.local_scale_fusion = nn.Conv2d(fusion_in_ch * 1, fusion_in_ch, kernel_size=1)

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

            if self.basin_context_attn is not None:
                branches_for_context = {}
                if "basin" in self.scales:
                    branches_for_context['basin'] = basin
                branches_for_context['context'] = context
                context = self.basin_context_attn(context, branches_for_context)
                bottleneck_for_concat.pop()
                bottleneck_for_concat.append(context)
            else:
                if self.basin_context_fusion is not None:
                    cat_bc = torch.cat([bottleneck_for_concat[-2], bottleneck_for_concat[-1]], dim=1) \
                             if len(bottleneck_for_concat) >= 2 else torch.cat([bottleneck_for_concat[-1], bottleneck_for_concat[-1]], dim=1)
                    context = self.basin_context_fusion(cat_bc)
                    bottleneck_for_concat.pop()
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

        if self.local_scale_attn is not None:
            branches_for_local = {}
            for s in self.local_scale_attn.sources:
                if s == 'local':
                    branches_for_local['local'] = local
                else:
                    if s == 'basin':
                        branches_for_local['basin'] = basin
                    elif s == 'context':
                        branches_for_local['context'] = context
                    else:
                        raise RuntimeError(f"Unexpected source {s} for local fusion")
            local = self.local_scale_attn(local, branches_for_local)
        else:
            cat_local = torch.cat(bottleneck_for_concat, dim=1)
            local = self.local_scale_fusion(cat_local)

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
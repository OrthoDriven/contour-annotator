import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.skip  = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch)
        ) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)

class AttentionGate(nn.Module):
    def __init__(self, f_g, f_l, f_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(f_g, f_int, 1, bias=False),
            nn.BatchNorm2d(f_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(f_l, f_int, 1, bias=False),
            nn.BatchNorm2d(f_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, g, x):
        g1 = F.interpolate(
            self.W_g(g), size=x.shape[2:], mode="bilinear", align_corners=False
        )
        x1 = self.W_x(x)
        psi = self.psi(F.relu(g1 + x1))
        return x * psi

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.conv(x)

class BackboneNetworkHailo(nn.Module):
    def __init__(self, in_ch=1, base_ch=32):
        super().__init__()

        # Encoder.
        self.enc1 = ResBlock(in_ch, base_ch)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ResBlock(base_ch, base_ch * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ResBlock(base_ch * 2, base_ch * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ResBlock(base_ch * 4, base_ch * 8)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck.
        self.bottleneck = ResBlock(base_ch * 8, base_ch * 16)

        # Attention gates.
        self.att4 = AttentionGate(base_ch * 8, base_ch * 8, base_ch * 4)
        self.att3 = AttentionGate(base_ch * 4, base_ch * 4, base_ch * 2)
        self.att2 = AttentionGate(base_ch * 2, base_ch * 2, base_ch)
        self.att1 = AttentionGate(base_ch, base_ch, max(1, base_ch // 2))

        # Decoder.
        self.up4 = UpBlock(base_ch * 16, base_ch * 8)
        self.dec4 = ResBlock(base_ch * 16, base_ch * 8)
        self.up3 = UpBlock(base_ch * 8, base_ch * 4)
        self.dec3 = ResBlock(base_ch * 8, base_ch * 4)
        self.up2 = UpBlock(base_ch * 4, base_ch * 2)
        self.dec2 = ResBlock(base_ch * 4, base_ch * 2)
        self.up1 = UpBlock(base_ch * 2, base_ch)
        self.dec1 = ResBlock(base_ch * 2, base_ch)

    def forward(self, x):

        # Encode.
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        # Bottleneck.
        b = self.bottleneck(self.pool4(e4))

        # Decode with attention-based skips.
        d4 = self.up4(b)
        e4_att = self.att4(d4, e4)
        d4 = self.dec4(torch.cat([d4, e4_att], dim=1))

        d3 = self.up3(d4)
        e3_att = self.att3(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3_att], dim=1))

        d2 = self.up2(d3)
        e2_att = self.att2(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2_att], dim=1))

        d1 = self.up1(d2)
        e1_att = self.att1(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1_att], dim=1))

        return d1

class HeatmapModelHailo(nn.Module):
    def __init__(self, in_ch=1, base_ch=32, out_ch=1):
        super().__init__()
        self.backbone = BackboneNetworkHailo(in_ch=in_ch, base_ch=base_ch)
        self.hm_head = nn.Conv2d(base_ch, out_ch, 1)

    def forward(self, x):
        return self.hm_head(self.backbone(x))


class LandmarkHead(nn.Module):
    def __init__(self, in_ch, hidden_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, 1, 1),
        )

    def forward(self, x):
        return self.block(x)


class LandmarkPresenceHead(nn.Module):
    def __init__(self, in_ch, hidden_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, 1, 1),
        )
        self.presence_head = nn.Sequential(
            nn.Linear(hidden_ch * 2, hidden_ch),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_ch, 1),
        )
        self.quality_head = nn.Sequential(
            nn.Linear(hidden_ch * 2, hidden_ch),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_ch, 1),
        )

    def forward(self, x):
        hidden = self.block[2](self.block[1](self.block[0](x)))
        heatmap = self.block[3](hidden)
        avg_pool = F.adaptive_avg_pool2d(hidden, 1).flatten(1)
        max_pool = F.adaptive_max_pool2d(hidden, 1).flatten(1)
        pooled = torch.cat([avg_pool, max_pool], dim=1)
        presence_logit = self.presence_head(pooled)
        quality_logit = self.quality_head(pooled)
        return heatmap, presence_logit, quality_logit


class MultiHeadHeatmapModelHailo(nn.Module):
    def __init__(self, in_ch=1, base_ch=32, out_ch=1, task_head_channels=None):
        super().__init__()
        if task_head_channels is None:
            task_head_channels = base_ch
        self.backbone = BackboneNetworkHailo(in_ch=in_ch, base_ch=base_ch)
        self.heads = nn.ModuleList(
            [LandmarkHead(base_ch, task_head_channels) for _ in range(out_ch)]
        )

    def forward(self, x):
        features = self.backbone(x)
        return torch.cat([head(features) for head in self.heads], dim=1)


class MultiHeadPresenceHeatmapModelHailo(nn.Module):
    def __init__(self, in_ch=1, base_ch=32, out_ch=1, task_head_channels=None):
        super().__init__()
        if task_head_channels is None:
            task_head_channels = base_ch
        self.backbone = BackboneNetworkHailo(in_ch=in_ch, base_ch=base_ch)
        self.heads = nn.ModuleList(
            [LandmarkPresenceHead(base_ch, task_head_channels) for _ in range(out_ch)]
        )

    def forward(self, x):
        features = self.backbone(x)
        heatmaps = []
        presence_logits = []
        quality_logits = []
        for head in self.heads:
            heatmap, presence_logit, quality_logit = head(features)
            heatmaps.append(heatmap)
            presence_logits.append(presence_logit)
            quality_logits.append(quality_logit)
        return {
            "heatmaps": torch.cat(heatmaps, dim=1),
            "presence_logits": torch.cat(presence_logits, dim=1),
            "quality_logits": torch.cat(quality_logits, dim=1),
        }

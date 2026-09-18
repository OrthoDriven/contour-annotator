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
            self.W_g(g), size=x.shape[2:], mode='bilinear', align_corners=False)
        x1 = self.W_x(x)
        psi = self.psi(F.relu(g1 + x1))
        return x * psi

class BackboneNetwork(nn.Module):
    def __init__(self, in_ch=1, base_ch=32, out_ch=1):
        super().__init__()

        # Encoder.
        self.enc1 = ResBlock(in_ch, base_ch)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ResBlock(base_ch, base_ch*2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ResBlock(base_ch*2, base_ch*4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ResBlock(base_ch*4, base_ch*8)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck.
        self.bottleneck = ResBlock(base_ch*8, base_ch*16)

        # Attention gates.
        self.att4 = AttentionGate(f_g=base_ch*8, f_l=base_ch*8, f_int=base_ch*4)
        self.att3 = AttentionGate(f_g=base_ch*4, f_l=base_ch*4, f_int=base_ch*2)
        self.att2 = AttentionGate(f_g=base_ch*2, f_l=base_ch*2, f_int=base_ch)
        self.att1 = AttentionGate(f_g=base_ch,   f_l=base_ch,   f_int=base_ch//2)

        # Decoder.
        self.up4 = nn.ConvTranspose2d(base_ch*16, base_ch*8, 2, stride=2)
        self.dec4 = ResBlock(base_ch*16, base_ch*8)
        self.up3 = nn.ConvTranspose2d(base_ch*8,  base_ch*4, 2, stride=2)
        self.dec3 = ResBlock(base_ch*8,  base_ch*4)
        self.up2 = nn.ConvTranspose2d(base_ch*4,  base_ch*2, 2, stride=2)
        self.dec2 = ResBlock(base_ch*4,  base_ch*2)
        self.up1 = nn.ConvTranspose2d(base_ch*2,  base_ch,   2, stride=2)
        self.dec1 = ResBlock(base_ch*2,  base_ch)

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

class HeatmapModel(nn.Module):
    def __init__(self, in_ch=1, base_ch=32, out_ch=1):
        super().__init__()
        self.backbone = BackboneNetwork(in_ch=in_ch, base_ch=base_ch)
        self.hm_head = nn.Conv2d(base_ch, out_ch, 1)

    def forward(self, x):
        feat = self.backbone(x)
        heatmaps = self.hm_head(feat)
        return heatmaps
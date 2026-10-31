"""FGSA U-Net for 17-channel surface water segmentation.

Input channels: B2, B3, B4, B8, B11, B12, AWEI, MBWI, NDWI,
MNDWI, slope, elevation, wetness, ASM, contrast, dissimilarity, entropy.
The network returns logits; apply sigmoid to obtain water probabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_norm(num_channels, norm_type="group", num_groups=8):
    norm_type = norm_type.lower()
    if norm_type == "batch":
        return nn.BatchNorm2d(num_channels)
    elif norm_type == "group":
        g = min(num_groups, num_channels)
        while num_channels % g != 0 and g > 1:
            g -= 1
        return nn.GroupNorm(g, num_channels)
    elif norm_type == "instance":
        return nn.InstanceNorm2d(num_channels, affine=True)
    else:
        raise ValueError(f"Unsupported norm_type: {norm_type}")


class EncodingBlock(nn.Module):
    def __init__(
        self,
        in_size,
        out_size,
        kernel_size=3,
        batch_norm=True,
        norm_type="group",
    ):
        super().__init__()

        norm1 = make_norm(out_size, norm_type) if batch_norm else nn.Identity()
        norm2 = make_norm(out_size, norm_type) if batch_norm else nn.Identity()

        self.encoding_block = nn.Sequential(
            nn.ReflectionPad2d((kernel_size - 1) // 2),
            nn.Conv2d(in_size, out_size, kernel_size=kernel_size, bias=False),
            nn.PReLU(),
            norm1,
            nn.ReflectionPad2d((kernel_size - 1) // 2),
            nn.Conv2d(out_size, out_size, kernel_size=kernel_size, bias=False),
            nn.PReLU(),
            norm2,
        )

    def forward(self, x):
        return self.encoding_block(x)


class DecodingBlock(nn.Module):
    def __init__(self, in_size, out_size, batch_norm=True, upsampling=True, norm_type="group"):
        super().__init__()

        if upsampling:
            self.up = nn.Sequential(
                nn.Upsample(mode="bilinear", scale_factor=2, align_corners=False),
                nn.Conv2d(in_size, out_size, kernel_size=1, bias=False),
            )
        else:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)

        self.conv = EncodingBlock(
            in_size,
            out_size,
            batch_norm=batch_norm,
            norm_type=norm_type
        )

    def forward(self, input1, input2):
        output2 = self.up(input2)
        output1 = F.interpolate(input1, size=output2.shape[2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([output1, output2], dim=1))


class FeatureGroupGate(nn.Module):
    """
    Feature-grouped gated fusion module.

    Input channel order:
    0  b2
    1  b3
    2  b4
    3  b8
    4  b11
    5  b12
    6  AWEI
    7  MBWI
    8  NDWI
    9  MNDWI
    10 slope
    11 dem
    12 wetness
    13 asm
    14 contrast
    15 dissimilarity
    16 entropy

    Groups:
    - spectral: b2,b3,b4,b8,b11,b12
    - water index: AWEI,MBWI,NDWI,MNDWI
    - terrain: slope,dem
    - texture/wetness: wetness,asm,contrast,dissimilarity,entropy
    """

    def __init__(
        self,
        out_channels=32,
        norm_type="group",
        use_spatial_gate=True,
    ):
        super().__init__()

        self.spectral_idx = [0, 1, 2, 3, 4, 5]
        self.water_idx = [6, 7, 8, 9]
        self.terrain_idx = [10, 11]
        self.texture_idx = [12, 13, 14, 15, 16]

        branch_channels = out_channels

        # Group-wise feature extraction
        self.spectral_branch = EncodingBlock(
            in_size=6,
            out_size=branch_channels,
            batch_norm=True,
            norm_type=norm_type,
        )

        self.water_branch = EncodingBlock(
            in_size=4,
            out_size=branch_channels,
            batch_norm=True,
            norm_type=norm_type,
        )

        self.terrain_branch = EncodingBlock(
            in_size=2,
            out_size=branch_channels,
            batch_norm=True,
            norm_type=norm_type,
        )

        self.texture_branch = EncodingBlock(
            in_size=5,
            out_size=branch_channels,
            batch_norm=True,
            norm_type=norm_type,
        )

        # Global group gate: output 4 weights for 4 feature groups
        self.group_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(branch_channels * 4, branch_channels, kernel_size=1, bias=False),
            nn.PReLU(),
            nn.Conv2d(branch_channels, 4, kernel_size=1, bias=True),
        )

        # Optional light spatial gate after group fusion
        self.use_spatial_gate = use_spatial_gate

        if use_spatial_gate:
            self.spatial_gate = nn.Sequential(
                nn.Conv2d(
                    branch_channels,
                    branch_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=branch_channels,
                    bias=False,
                ),
                make_norm(branch_channels, norm_type),
                nn.PReLU(),
                nn.Conv2d(branch_channels, 1, kernel_size=1, bias=True),
                nn.Sigmoid(),
            )
        else:
            self.spatial_gate = None

        # Final refinement
        self.refine = nn.Sequential(
            nn.Conv2d(branch_channels, out_channels, kernel_size=1, bias=False),
            make_norm(out_channels, norm_type),
            nn.PReLU(),
        )

    def forward(self, x):
        xs = x[:, self.spectral_idx, :, :]
        xw = x[:, self.water_idx, :, :]
        xt = x[:, self.terrain_idx, :, :]
        xx = x[:, self.texture_idx, :, :]

        fs = self.spectral_branch(xs)
        fw = self.water_branch(xw)
        ft = self.terrain_branch(xt)
        fx = self.texture_branch(xx)

        concat = torch.cat([fs, fw, ft, fx], dim=1)

        # weights: [B, 4, 1, 1]
        gate_logits = self.group_gate(concat)
        weights = torch.softmax(gate_logits, dim=1)

        ws = weights[:, 0:1, :, :]
        ww = weights[:, 1:2, :, :]
        wt = weights[:, 2:3, :, :]
        wx = weights[:, 3:4, :, :]

        fused = ws * fs + ww * fw + wt * ft + wx * fx

        if self.use_spatial_gate:
            fused = fused * self.spatial_gate(fused)

        fused = self.refine(fused)

        return fused

    @torch.no_grad()
    def get_group_weights(self, x):
        """
        Return group weights for interpretability.
        Output shape: [B, 4]
        order: spectral, water_index, terrain, texture
        """
        xs = x[:, self.spectral_idx, :, :]
        xw = x[:, self.water_idx, :, :]
        xt = x[:, self.terrain_idx, :, :]
        xx = x[:, self.texture_idx, :, :]

        fs = self.spectral_branch(xs)
        fw = self.water_branch(xw)
        ft = self.terrain_branch(xt)
        fx = self.texture_branch(xx)

        concat = torch.cat([fs, fw, ft, fx], dim=1)
        gate_logits = self.group_gate(concat)
        weights = torch.softmax(gate_logits, dim=1)

        return weights.squeeze(-1).squeeze(-1)


class ELA(nn.Module):

    def __init__(self, in_channels, phi):
        super(ELA, self).__init__()
        Kernel_size = {'T': 5, 'B': 7, 'S': 5, 'L': 7}[phi]
        groups = {'T': in_channels, 'B': in_channels, 'S': in_channels // 8, 'L': in_channels // 8}[phi]
        num_groups = {'T': 32, 'B': 16, 'S': 16, 'L': 16}[phi]
        pad = Kernel_size // 2
        self.con1 = nn.Conv1d(in_channels, in_channels, kernel_size=Kernel_size, padding=pad, groups=groups, bias=False)
        self.GN = nn.GroupNorm(num_groups, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        b, c, h, w = input.size()
        x_h = torch.mean(input, dim=3, keepdim=True).view(b, c, h)
        x_w = torch.mean(input, dim=2, keepdim=True).view(b, c, w)
        x_h = self.con1(x_h)
        x_w = self.con1(x_w)
        x_h = self.sigmoid(self.GN(x_h)).view(b, c, h, 1)
        x_w = self.sigmoid(self.GN(x_w)).view(b, c, 1, w)
        return x_h * x_w * input


class ScaledResidualAttention(nn.Module):
    """Apply attention through a scaled residual connection."""

    def __init__(self, attention_module, init_alpha=0.1):
        super().__init__()
        self.attention = attention_module
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))

    def forward(self, x):
        return x + self.alpha * self.attention(x)


class ResFGELAUNet(nn.Module):
    """U-Net with feature-group fusion and efficient location attention."""

    def __init__(
        self,
        in_channels=17,
        num_classes=1,
        features=(32, 64, 128, 256, 512),
        norm_type="group",
        use_spatial_gate=True,
    ):
        super().__init__()

        if in_channels != 17:
            raise ValueError("ResFGELAUNet is designed for fixed 17-channel input.")

        f1, f2, f3, f4, f5 = features

        self.conv1_raw = EncodingBlock(
            in_channels,
            f1,
            batch_norm=True,
            norm_type=norm_type,
        )

        self.fg = FeatureGroupGate(
            out_channels=f1,
            norm_type=norm_type,
            use_spatial_gate=use_spatial_gate,
        )

        self.ela1 = ScaledResidualAttention(
            ELA(in_channels=f1, phi="B"),
            init_alpha=0.1,
        )

        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = EncodingBlock(f1, f2, batch_norm=True, norm_type=norm_type)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = EncodingBlock(f2, f3, batch_norm=True, norm_type=norm_type)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = EncodingBlock(f3, f4, batch_norm=True, norm_type=norm_type)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = EncodingBlock(f4, f5, batch_norm=True, norm_type=norm_type)

        self.decode4 = DecodingBlock(f5, f4, batch_norm=True, norm_type=norm_type)
        self.decode3 = DecodingBlock(f4, f3, batch_norm=True, norm_type=norm_type)
        self.decode2 = DecodingBlock(f3, f2, batch_norm=True, norm_type=norm_type)
        self.decode1 = DecodingBlock(f2, f1, batch_norm=True, norm_type=norm_type)

        self.final = nn.Conv2d(f1, num_classes, kernel_size=1)

    def forward(self, x):
        input_size = x.shape[2:]

        raw1 = self.conv1_raw(x)
        fg1 = self.fg(x)

        conv1 = raw1 + fg1
        conv1 = self.ela1(conv1)

        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)

        decode4 = self.decode4(conv4, center)
        decode3 = self.decode3(conv3, decode4)
        decode2 = self.decode2(conv2, decode3)
        decode1 = self.decode1(conv1, decode2)

        logits = self.final(decode1)

        return F.interpolate(
            logits,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )


FGSAUNet = ResFGELAUNet

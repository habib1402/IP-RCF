"""
PolypPVT: Multi-Perspective Feature Learning for Polyp Segmentation
==================================================================

This module implements a controlled backbone study that combines:
- A selectable pretrained encoder with four hierarchical feature stages
- A common projection interface shared by every encoder
- Multi-Perspective Feature Learning (MPFL) with dynamic and deformable convolutions
- Cross-scale Fusion Module (CFM) with attention mechanisms
- Complementary Attention Module for enhanced segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d
from torchvision.models import resnet50
import timm
from safetensors.torch import load_file
from pathlib import Path
import math


SUPPORTED_BACKBONES = (
    'dinov3_convnext_base',
    'gastronet5m_resnet50',
    'sam2_1_hiera_base_plus',
    'convnextv2_tiny_fcmae',
    'tinyvit_21m_384',
    'pvtv2_b0',
    'pvtv2_b1',
    'pvtv2_b2',
    'caformer_s18_in1k_384',
    'efficientvit_b2_r288',
    'repvit_m2_3_dist_450e',
)

SUPPORTED_DECODERS = (
    'mpfl_cfm_cam',
    'pff_baseline',
    'cdg_pff',
    'eqam_affine',
    'eqam_supervised',
    'eqam_local',
)

DECODER_DESCRIPTIONS = {
    'mpfl_cfm_cam': 'original MPFL-CFM-complementary-attention head',
    'pff_baseline': (
        'plain additive top-down progressive fusion with '
        'depthwise-separable refinement'
    ),
    'cdg_pff': (
        'cross-scale disagreement-gated top-down progressive fusion with '
        'depthwise-separable refinement'
    ),
    'eqam_affine': (
        'endoscopy-quality adaptive channel modulation before Plain-PFF'
    ),
    'eqam_supervised': (
        'artifact-supervised endoscopy-quality adaptive channel modulation '
        'before Plain-PFF'
    ),
    'eqam_local': (
        'artifact-supervised endoscopy-quality adaptive channel modulation '
        'with local reliability-gated Plain-PFF merges'
    ),
}


class DynamicConv2d(nn.Module):
    """
    Dynamic Convolution with multiple adaptive kernels.
    
    Uses attention mechanism to adaptively combine multiple convolution kernels,
    allowing the model to adapt to different feature patterns dynamically.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Convolution kernel size. Default: 3
        stride (int): Convolution stride. Default: 1
        padding (int): Convolution padding. Default: 1
        num_kernels (int): Number of dynamic kernels. Default: 4
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, num_kernels=4):
        super(DynamicConv2d, self).__init__()
        self.num_kernels = num_kernels
        
        # Multiple convolution kernels
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            for _ in range(num_kernels)
        ])
        
        # Attention mechanism for kernel selection
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, num_kernels, kernel_size=1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        """
        Forward pass with dynamic kernel selection.
        
        Args:
            x (torch.Tensor): Input tensor [B, C, H, W]
            
        Returns:
            torch.Tensor: Output tensor with adaptively combined features
        """
        weights = self.attention(x)  # [B, K, 1, 1]
        output = sum(conv(x) * weights[:, i:i+1] for i, conv in enumerate(self.convs))
        return output


class MPFL(nn.Module):
    """
    Multi-Perspective Feature Learning Module.
    
    Combines three different convolution types to capture diverse feature perspectives:
    1. Standard convolution for regular patterns
    2. Dynamic convolution for adaptive patterns
    3. Deformable convolution for irregular patterns
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
    """
    
    def __init__(self, in_channels, out_channels):
        super(MPFL, self).__init__()
        
        # Three different convolution perspectives
        self.ori_conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.dyn_conv = DynamicConv2d(in_channels, out_channels)
        
        # Deformable convolution components
        self.offset_conv = nn.Conv2d(in_channels, 2*3*3, 3, padding=1)
        self.def_conv = DeformConv2d(in_channels, out_channels, 3, padding=1)
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Forward pass combining three convolution perspectives.
        
        Args:
            x (torch.Tensor): Input feature tensor
            
        Returns:
            torch.Tensor: Fused multi-perspective features
        """
        # Three different perspectives
        p1 = self.ori_conv(x)           # Standard convolution
        p2 = self.dyn_conv(x)           # Dynamic convolution
        
        offset = self.offset_conv(x)    # Offset for deformable conv
        p3 = self.def_conv(x, offset)   # Deformable convolution
        
        # Concatenate and fuse all perspectives
        fused_features = self.fusion(torch.cat([p1, p2, p3], dim=1))
        return fused_features

class EfficientChannelAttention(nn.Module):
    """Efficient Channel Attention without dimension reduction"""
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        self.channels = channels
        self.gamma = gamma
        self.b = b
        
        # Calculate kernel size adaptively
        t = int(abs((math.log(channels, 2) + self.b) / self.gamma))
        k = t if t % 2 else t + 1
        
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Global average pooling
        y = F.adaptive_avg_pool2d(x, 1).squeeze(-1).transpose(-1, -2)  # [B, 1, C]
        
        # 1D convolution for channel interaction
        y = self.conv(y)
        
        # Sigmoid activation
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)  # [B, C, 1, 1]
        
        return y


class CoordinateAttention(nn.Module):
    """Coordinate Attention for better spatial awareness"""
    def __init__(self, channels, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        mip = max(8, channels // reduction)
        
        self.conv1 = nn.Conv2d(channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)
        
        self.conv_h = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        
        return a_h * a_w


class AdvancedSpatialAttention(nn.Module):
    """Advanced spatial attention with multi-scale context"""
    def __init__(self, channels, kernel_sizes=[3, 5, 7]):
        super().__init__()
        # Ensure proper channel division
        per_branch_channels = channels // len(kernel_sizes)
        remaining_channels = channels - (per_branch_channels * (len(kernel_sizes) - 1))
        
        self.convs = nn.ModuleList()
        for i, k in enumerate(kernel_sizes):
            out_channels = remaining_channels if i == len(kernel_sizes) - 1 else per_branch_channels
            self.convs.append(
                nn.Conv2d(channels, out_channels, kernel_size=k, padding=k//2)
            )
        
        self.conv_out = nn.Conv2d(channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Multi-scale feature extraction
        multi_scale = [conv(x) for conv in self.convs]
        concat_feat = torch.cat(multi_scale, dim=1)
        
        # Ensure exact channel match
        if concat_feat.shape[1] != x.shape[1]:
            # Adjust channels to match input
            diff = x.shape[1] - concat_feat.shape[1]
            if diff > 0:
                # Pad with zeros if needed
                padding = torch.zeros(concat_feat.shape[0], diff, *concat_feat.shape[2:], 
                                    device=concat_feat.device, dtype=concat_feat.dtype)
                concat_feat = torch.cat([concat_feat, padding], dim=1)
            elif diff < 0:
                # Trim if needed
                concat_feat = concat_feat[:, :x.shape[1]]
        
        # Spatial attention map
        att_map = self.conv_out(concat_feat)
        att_map = self.sigmoid(att_map)
        
        return att_map


class CFM(nn.Module):
    """
    Enhanced Cross-scale Fusion Module with Triple Attention Mechanisms.
    
    Combines three complementary attention mechanisms:
    1. InterChannelCorrelationAttention - For dynamic feature channel correlations
    2. PositionalEncodingAttention - For spatial-channel interaction with position encoding  
    3. NestedSpatialAttention - For hierarchical nested spatial context
    
    Args:
        channels (int): Number of channels. Default: 64
        reduction (int): Channel reduction ratio for coordinate attention. Default: 32
    """
    
    def __init__(self, channels=64, reduction=32):
        super(CFM, self).__init__()
        
        # Depthwise separable convolutions for efficient processing
        self.depthwise2 = nn.Conv2d(channels*2, channels*2, 3, padding=1, groups=channels*2)
        self.pointwise2 = nn.Conv2d(channels*2, channels, 1)
        
        self.depthwise3 = nn.Conv2d(channels*3, channels*3, 3, padding=1, groups=channels*3)
        self.pointwise3 = nn.Conv2d(channels*3, channels, 1)
        
        self.norm = nn.BatchNorm2d(channels)
        
        # Triple attention mechanisms
        self.inter_channel_correlation_attention = EfficientChannelAttention(channels)
        self.positional_encoding_attention = CoordinateAttention(channels, reduction)
        self.nested_spatial_attention = AdvancedSpatialAttention(channels)
        
        # Learnable attention fusion weights
        self.attention_weights = nn.Parameter(torch.ones(3) / 3)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, *inputs):
        """
        Forward pass for cross-scale feature fusion with triple attention.
        
        Args:
            *inputs: Variable number of input feature tensors
            
        Returns:
            torch.Tensor: Fused features with triple attention applied
        """
        # Resize all inputs to match the first input's spatial dimensions
        base_size = inputs[0].shape[2:]
        resized = [
            F.interpolate(x, base_size, mode='bilinear', align_corners=False) 
            if x.shape[2:] != base_size else x 
            for x in inputs
        ]
        
        # Process based on number of inputs
        if len(resized) == 2:
            x = torch.cat(resized, dim=1)
            x = F.relu(self.norm(self.pointwise2(self.depthwise2(x))))
        else:
            x = torch.cat(resized, dim=1)
            x = F.relu(self.norm(self.pointwise3(self.depthwise3(x))))
        
        # Apply triple attention mechanisms
        # 1. Inter-channel correlation attention - captures dynamic feature channel correlations
        inter_channel_corr_att = self.inter_channel_correlation_attention(x)
        
        # 2. Positional encoding attention - captures spatial-channel relationships with position encoding
        positional_encoding_att = self.positional_encoding_attention(x)
        
        # 3. Nested spatial attention - captures hierarchical nested spatial context
        nested_spatial_att = self.nested_spatial_attention(x)
        
        # Learnable weighted combination of attentions
        attention_weights = self.softmax(self.attention_weights)
        
        # Apply weighted attention combination
        attended_features = x * (
            attention_weights[0] * inter_channel_corr_att + 
            attention_weights[1] * positional_encoding_att + 
            attention_weights[2] * nested_spatial_att
        )
        
        return attended_features


class ComplementaryAttentionModule(nn.Module):
    """
    Complementary Attention Module.
    
    Focuses on regions that are complementary to existing attention maps,
    helping to capture missed important regions in the segmentation process.
    
    Args:
        in_channels (int): Number of input channels
        feat_channels (int): Number of feature channels
        depth (int): Depth of the convolution layers. Default: 3
    """
    
    def __init__(self, in_channels, feat_channels, depth=3):
        super(ComplementaryAttentionModule, self).__init__()
        
        # Build convolution layers
        layers = []
        for _ in range(depth-1):
            layers += [
                nn.Conv2d(feat_channels, feat_channels, 3, padding=1),
                nn.BatchNorm2d(feat_channels),
                nn.ReLU(inplace=True)
            ]
        self.convs = nn.Sequential(*layers)
        
        # Final attention map generation
        self.final_conv = nn.Conv2d(feat_channels, 1, 1)
        
        # Learnable residual scaling parameter
        self.res_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, feat, att_map):
        """
        Forward pass for complementary attention computation.
        
        Args:
            feat (torch.Tensor): Input feature tensor
            att_map (torch.Tensor): Existing attention map
            
        Returns:
            torch.Tensor: Enhanced attention map with complementary regions
        """
        # Resize attention map to match feature dimensions
        att_map = F.interpolate(att_map, feat.shape[2:], mode='bilinear', align_corners=False)
        
        # Create complementary mask (focus on unattended regions)
        comp_mask = 1 - torch.sigmoid(att_map)
        
        # Apply complementary mask and process features
        masked_feat = comp_mask * feat
        processed_feat = self.convs(masked_feat)
        
        # Generate final attention with residual connection
        complementary_att = self.final_conv(processed_feat) + self.res_scale * att_map
        
        return complementary_att


class FeatureProjection(nn.Sequential):
    """Map a native encoder stage to the common decoder width."""

    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )


class DepthwiseSeparableRefinement(nn.Module):
    """Lightweight spatial refinement used by both progressive decoders."""

    def __init__(self, channels):
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class ProgressiveFeatureFusion(nn.Module):
    """Parameter-efficient top-down fusion control for the decoder study.

    The four projected encoder maps are progressively merged from stride 32
    to stride 4 by addition and depthwise-separable refinement.  One shared
    prediction head produces the three deeply supervised logits, matching the
    response interface used by every controlled backbone experiment.
    """

    def __init__(self, channels):
        super().__init__()
        self.refinements = nn.ModuleList([
            DepthwiseSeparableRefinement(channels) for _ in range(4)
        ])
        self.prediction_head = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, features, target_size, return_fused=False):
        fused = [None] * 4
        fused[3] = self.refinements[3](features[3])
        for level in (2, 1, 0):
            coarse = F.interpolate(
                fused[level + 1],
                size=features[level].shape[2:],
                mode='bilinear',
                align_corners=False,
            )
            fused[level] = self.refinements[level](features[level] + coarse)

        outputs = tuple(
            self.prediction_head(
                F.interpolate(
                    fused[level],
                    size=target_size,
                    mode='bilinear',
                    align_corners=False,
                )
            )
            for level in (0, 1, 2)
        )
        if return_fused:
            return outputs, fused[0]
        return outputs


class CrossScaleDisagreementGate(nn.Module):
    """Selectively inject coarse semantics using cross-scale disagreement.

    The gate observes the fine feature, the resized coarse feature and their
    absolute disagreement.  A depthwise 3x3 convolution captures local
    disagreement patterns before a pointwise projection emits a channel-wise
    spatial gate.  The fine feature remains an explicit residual path.
    """

    def __init__(self, channels):
        super().__init__()
        gate_channels = channels * 3
        self.gate = nn.Sequential(
            nn.Conv2d(
                gate_channels,
                gate_channels,
                kernel_size=3,
                padding=1,
                groups=gate_channels,
                bias=False,
            ),
            nn.BatchNorm2d(gate_channels),
            nn.GELU(),
            nn.Conv2d(gate_channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.refinement = DepthwiseSeparableRefinement(channels)

    def forward(self, fine, coarse):
        coarse = F.interpolate(
            coarse,
            size=fine.shape[2:],
            mode='bilinear',
            align_corners=False,
        )
        disagreement = torch.abs(fine - coarse)
        gate = self.gate(torch.cat([fine, coarse, disagreement], dim=1))
        return self.refinement(fine + gate * coarse)


class DisagreementGatedProgressiveFusion(nn.Module):
    """Top-down progressive fusion with lightweight disagreement gates."""

    def __init__(self, channels):
        super().__init__()
        self.deep_refinement = DepthwiseSeparableRefinement(channels)
        self.fusion_blocks = nn.ModuleList([
            CrossScaleDisagreementGate(channels) for _ in range(3)
        ])
        self.prediction_head = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, features, target_size):
        fused = [None] * 4
        fused[3] = self.deep_refinement(features[3])
        for block, level in zip(self.fusion_blocks, (2, 1, 0)):
            fused[level] = block(features[level], fused[level + 1])

        return tuple(
            self.prediction_head(
                F.interpolate(
                    fused[level],
                    size=target_size,
                    mode='bilinear',
                    align_corners=False,
                )
            )
            for level in (0, 1, 2)
        )


class EndoscopyQualityDescriptor(nn.Module):
    """Encode global appearance/detail statistics from the finest PVT map."""

    def __init__(self, channels, embedding_channels=32, quality_targets=5):
        super().__init__()
        self.channels = int(channels)
        self.embedding_channels = int(embedding_channels)
        statistic_channels = 3 * self.channels
        self.embedding = nn.Sequential(
            nn.Linear(statistic_channels, self.embedding_channels),
            nn.GELU(),
            nn.Linear(self.embedding_channels, self.embedding_channels),
            nn.GELU(),
        )
        self.quality_head = nn.Linear(self.embedding_channels, quality_targets)

    def forward(self, feature):
        mean = feature.mean(dim=(2, 3))
        variance = feature.var(dim=(2, 3), unbiased=False)
        local_average = F.avg_pool2d(feature, 3, stride=1, padding=1)
        detail_energy = (feature - local_average).abs().mean(dim=(2, 3))
        statistics = torch.cat(
            (mean, torch.sqrt(variance + 1e-6), detail_energy), dim=1
        )
        embedding = self.embedding(statistics)
        quality_prediction = torch.sigmoid(self.quality_head(embedding))
        return embedding, quality_prediction


class QualityAffineModulation(nn.Module):
    """Apply a bounded, identity-initialized channel-wise affine transform."""

    def __init__(self, embedding_channels, feature_channels):
        super().__init__()
        self.affine = nn.Linear(embedding_channels, 2 * feature_channels)
        nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)

    def forward(self, feature, quality_embedding):
        gamma, beta = self.affine(quality_embedding).chunk(2, dim=1)
        gamma = 0.25 * torch.tanh(gamma).unsqueeze(-1).unsqueeze(-1)
        beta = 0.25 * torch.tanh(beta).unsqueeze(-1).unsqueeze(-1)
        return feature * (1.0 + gamma) + beta


class LocalReliabilityMerge(nn.Module):
    """Near-identity top-down merge that can suppress unreliable semantics."""

    def __init__(self, channels, hidden_channels=4):
        super().__init__()
        self.router = nn.Sequential(
            nn.Conv2d(3, hidden_channels, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=True),
        )
        nn.init.zeros_(self.router[-1].weight)
        nn.init.constant_(self.router[-1].bias, -4.0)
        self.refinement = DepthwiseSeparableRefinement(channels)
        self.last_mean_reliability = None

    @staticmethod
    def _standardize(value):
        mean = value.mean(dim=(2, 3), keepdim=True)
        scale = value.std(dim=(2, 3), keepdim=True, unbiased=False).clamp_min(1e-5)
        return (value - mean) / scale

    def forward(self, fine, coarse):
        coarse = F.interpolate(
            coarse,
            size=fine.shape[2:],
            mode='bilinear',
            align_corners=False,
        )
        fine_energy = fine.square().mean(dim=1, keepdim=True).sqrt()
        coarse_energy = coarse.square().mean(dim=1, keepdim=True).sqrt()
        disagreement = (fine - coarse).abs().mean(dim=1, keepdim=True)
        evidence = torch.cat(
            tuple(
                self._standardize(value)
                for value in (fine_energy, coarse_energy, disagreement)
            ),
            dim=1,
        )
        # Starts at approximately 0.991, preserving the Plain-PFF merge.  The
        # learned response can suppress at most half of the coarse signal.
        reliability = 1.0 - 0.5 * torch.sigmoid(self.router(evidence))
        self.last_mean_reliability = reliability.detach().mean()
        return self.refinement(fine + reliability * coarse)


class QualityAdaptiveProgressiveFusion(nn.Module):
    """Quality-conditioned feature modulation followed by Plain-PFF topology."""

    def __init__(
        self,
        channels,
        use_local_reliability=False,
        embedding_channels=32,
        quality_targets=5,
    ):
        super().__init__()
        self.use_local_reliability = bool(use_local_reliability)
        self.quality_descriptor = EndoscopyQualityDescriptor(
            channels,
            embedding_channels=embedding_channels,
            quality_targets=quality_targets,
        )
        self.modulations = nn.ModuleList([
            QualityAffineModulation(embedding_channels, channels)
            for _ in range(4)
        ])
        self.deep_refinement = DepthwiseSeparableRefinement(channels)
        self.merge_refinements = (
            None
            if self.use_local_reliability
            else nn.ModuleList([
                DepthwiseSeparableRefinement(channels) for _ in range(3)
            ])
        )
        self.local_merges = (
            nn.ModuleList([
                LocalReliabilityMerge(channels) for _ in range(3)
            ])
            if self.use_local_reliability
            else None
        )
        self.prediction_head = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, features, target_size):
        quality_embedding, quality_prediction = self.quality_descriptor(features[0])
        features = [
            modulation(feature, quality_embedding)
            for modulation, feature in zip(self.modulations, features)
        ]

        fused = [None] * 4
        fused[3] = self.deep_refinement(features[3])
        merge_index = 0
        for level in (2, 1, 0):
            if self.local_merges is None:
                coarse = F.interpolate(
                    fused[level + 1],
                    size=features[level].shape[2:],
                    mode='bilinear',
                    align_corners=False,
                )
                fused[level] = self.merge_refinements[merge_index](
                    features[level] + coarse
                )
                merge_index += 1
            else:
                fused[level] = self.local_merges[merge_index](
                    features[level], fused[level + 1]
                )
                merge_index += 1

        outputs = tuple(
            self.prediction_head(
                F.interpolate(
                    fused[level],
                    size=target_size,
                    mode='bilinear',
                    align_corners=False,
                )
            )
            for level in (0, 1, 2)
        )
        return outputs, quality_prediction


class PolypPVT(nn.Module):
    """HMPFormer with a controlled, selectable pretrained backbone.

    All encoders expose four maps at strides 4/8/16/32. Minimal 1x1
    projections convert native stage widths to ``channel``. MPFL and every
    downstream block are consequently identical across the backbone study.

    ``return_response=True`` additionally returns the raw combined logit used
    for future response-level knowledge distillation. The default return value
    remains the original three-output tuple used by Train.py and Test.py.
    """

    def __init__(
        self,
        channel=64,
        backbone='dinov3_convnext_base',
        decoder_variant='mpfl_cfm_cam',
        backbone_weights=None,
        load_backbone_weights=True,
    ):
        super().__init__()
        if backbone not in SUPPORTED_BACKBONES:
            raise ValueError(
                f'Unsupported backbone {backbone!r}; choose from {SUPPORTED_BACKBONES}'
            )
        if decoder_variant not in SUPPORTED_DECODERS:
            raise ValueError(
                f'Unsupported decoder {decoder_variant!r}; '
                f'choose from {SUPPORTED_DECODERS}'
            )

        self.backbone_name = backbone
        self.decoder_channel = channel
        self.decoder_variant = decoder_variant
        self.backbone_weights_path = str(
            Path(backbone_weights).expanduser().resolve()
            if backbone_weights
            else self._default_weight_path(backbone).resolve()
        )
        self.backbone, native_channels = self._initialize_backbone(
            backbone,
            load_backbone_weights,
        )
        self.native_channels = tuple(native_channels)

        # These are the only encoder-specific channel converters. Their output
        # shape is identical for every experiment in the controlled study.
        self.feature_projections = nn.ModuleList([
            FeatureProjection(in_channels, channel)
            for in_channels in self.native_channels
        ])

        if decoder_variant == 'mpfl_cfm_cam':
            # Original head retained verbatim for the completed backbone study.
            self.mpfl = nn.ModuleList([MPFL(channel, channel) for _ in range(4)])
            self.cfm = nn.ModuleList([CFM(channel) for _ in range(4)])
            self.comp_attention = ComplementaryAttentionModule(channel, channel)
            self.final_conv = nn.Conv2d(channel, 1, 1)
        elif decoder_variant == 'pff_baseline':
            self.progressive_decoder = ProgressiveFeatureFusion(channel)
        elif decoder_variant == 'cdg_pff':
            self.progressive_decoder = DisagreementGatedProgressiveFusion(channel)
        else:
            self.progressive_decoder = QualityAdaptiveProgressiveFusion(
                channel,
                use_local_reliability=decoder_variant == 'eqam_local',
            )

    @staticmethod
    def _default_weight_path(backbone):
        pretrained_dir = Path(__file__).resolve().parent.parent / 'pretrained_pth'
        filenames = {
            'dinov3_convnext_base': (
                'convnext_base_dinov3_lvd1689m.safetensors'
            ),
            'gastronet5m_resnet50': 'gastronet5m_resnet50_dino.pth',
            'sam2_1_hiera_base_plus': 'sam2.1_hiera_base_plus.pt',
            'convnextv2_tiny_fcmae': (
                'convnextv2_tiny.fcmae_ft_in22k_in1k_384.safetensors'
            ),
            'tinyvit_21m_384': (
                'tiny_vit_21m_384.dist_in22k_ft_in1k.safetensors'
            ),
            'pvtv2_b0': 'pvt_v2_b0.in1k.safetensors',
            'pvtv2_b1': 'pvt_v2_b1.in1k.safetensors',
            'pvtv2_b2': 'pvt_v2_b2.in1k.safetensors',
            'caformer_s18_in1k_384': (
                'caformer_s18.sail_in1k_384.safetensors'
            ),
            'efficientvit_b2_r288': (
                'efficientvit_b2.r288_in1k.safetensors'
            ),
            'repvit_m2_3_dist_450e': (
                'repvit_m2_3.dist_450e_in1k.safetensors'
            ),
        }
        return pretrained_dir / filenames[backbone]

    def _initialize_backbone(self, backbone, load_weights):
        if backbone == 'dinov3_convnext_base':
            model = timm.create_model(
                'convnext_base',
                pretrained=False,
                num_classes=0,
            )
            native_channels = (128, 256, 512, 1024)
        elif backbone == 'gastronet5m_resnet50':
            model = resnet50(weights=None)
            model.fc = nn.Identity()
            native_channels = (256, 512, 1024, 2048)
        elif backbone == 'convnextv2_tiny_fcmae':
            return self._initialize_timm_feature_backbone(
                model_name='convnextv2_tiny.fcmae_ft_in22k_in1k_384',
                native_channels=(96, 192, 384, 768),
                display_name='ConvNeXtV2-Tiny FCMAE IN22K-to-IN1K',
                load_weights=load_weights,
            )
        elif backbone == 'tinyvit_21m_384':
            return self._initialize_timm_feature_backbone(
                model_name='tiny_vit_21m_384.dist_in22k_ft_in1k',
                native_channels=(96, 192, 384, 576),
                display_name='TinyViT-21M-384 distilled IN22K-to-IN1K',
                load_weights=load_weights,
            )
        elif backbone == 'pvtv2_b0':
            return self._initialize_timm_feature_backbone(
                model_name='pvt_v2_b0.in1k',
                native_channels=(32, 64, 160, 256),
                display_name='PVTv2-B0 ImageNet-1K',
                load_weights=load_weights,
            )
        elif backbone == 'pvtv2_b1':
            return self._initialize_timm_feature_backbone(
                model_name='pvt_v2_b1.in1k',
                native_channels=(64, 128, 320, 512),
                display_name='PVTv2-B1 ImageNet-1K',
                load_weights=load_weights,
            )
        elif backbone == 'pvtv2_b2':
            return self._initialize_timm_feature_backbone(
                model_name='pvt_v2_b2.in1k',
                native_channels=(64, 128, 320, 512),
                display_name='PVTv2-B2 ImageNet-1K',
                load_weights=load_weights,
            )
        elif backbone == 'caformer_s18_in1k_384':
            return self._initialize_timm_feature_backbone(
                model_name='caformer_s18.sail_in1k_384',
                native_channels=(64, 128, 320, 512),
                display_name='CAFormer-S18 ImageNet-1K 384',
                load_weights=load_weights,
                validate_reduction_metadata=False,
            )
        elif backbone == 'efficientvit_b2_r288':
            return self._initialize_timm_feature_backbone(
                model_name='efficientvit_b2.r288_in1k',
                native_channels=(48, 96, 192, 384),
                display_name='EfficientViT-B2 ImageNet-1K 288',
                load_weights=load_weights,
            )
        elif backbone == 'repvit_m2_3_dist_450e':
            return self._initialize_timm_feature_backbone(
                model_name='repvit_m2_3.dist_450e_in1k',
                native_channels=(80, 160, 320, 640),
                display_name='RepViT-M2.3 distilled ImageNet-1K 450e',
                load_weights=load_weights,
            )
        else:
            # Use only SAM 2.1's hierarchical image trunk. Prompt encoder,
            # memory modules, FPN neck and mask decoder are intentionally not
            # part of this controlled encoder comparison.
            model = timm.create_model(
                'sam2_hiera_base_plus',
                pretrained=False,
                features_only=True,
            )
            native_channels = (112, 224, 448, 896)

        if not load_weights:
            return model, native_channels

        weight_path = Path(self.backbone_weights_path)
        if not weight_path.is_file():
            raise FileNotFoundError(
                f'Required pretrained weights for {backbone} are missing: '
                f'{weight_path}'
            )

        if backbone == 'dinov3_convnext_base':
            self._load_dinov3_convnext(model, weight_path)
        elif backbone == 'sam2_1_hiera_base_plus':
            self._load_sam2_1_hiera(model, weight_path)
        else:
            self._load_gastronet_resnet50(model, weight_path)
        return model, native_channels

    def _initialize_timm_feature_backbone(
        self,
        model_name,
        native_channels,
        display_name,
        load_weights,
        validate_reduction_metadata=True,
    ):
        """Build a four-stage timm encoder from an explicit local checkpoint."""
        create_kwargs = {
            'pretrained': load_weights,
            'features_only': True,
            'out_indices': (0, 1, 2, 3),
        }
        if load_weights:
            weight_path = Path(self.backbone_weights_path)
            if not weight_path.is_file():
                raise FileNotFoundError(
                    f'Required pretrained weights for {self.backbone_name} '
                    f'are missing: {weight_path}'
                )
            # ``file`` has priority over the model's URL/Hub entries. timm
            # loads the classification state before wrapping the network for
            # intermediate features and removes only the unused classifier.
            create_kwargs['pretrained_cfg_overlay'] = {
                'file': str(weight_path),
            }

        model = timm.create_model(model_name, **create_kwargs)
        actual_channels = tuple(
            feature['num_chs'] for feature in model.feature_info
        )
        actual_reductions = tuple(
            feature['reduction'] for feature in model.feature_info
        )
        if actual_channels != tuple(native_channels):
            raise RuntimeError(
                f'{display_name} channel mismatch: expected {native_channels}, '
                f'got {actual_channels}'
            )
        # timm 1.0.11's MetaFormer feature wrapper labels every CAFormer stage
        # with reduction=2 even though the returned tensors are at strides
        # 4/8/16/32.  Real feature geometry is covered by the controlled
        # 352x352 integration smoke test; retain strict metadata validation for
        # every backbone whose metadata is reliable.
        if (
            validate_reduction_metadata
            and actual_reductions != (4, 8, 16, 32)
        ):
            raise RuntimeError(
                f'{display_name} reduction mismatch: expected (4, 8, 16, 32), '
                f'got {actual_reductions}'
            )
        if load_weights:
            print(
                f'Successfully loaded {display_name} pretrained weights '
                f'from {self.backbone_weights_path}'
            )
        return model, native_channels

    @staticmethod
    def _load_dinov3_convnext(model, weight_path):
        state_dict = load_file(str(weight_path), device='cpu')
        incompatible = model.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                'DINOv3 ConvNeXt checkpoint mismatch: '
                f'missing={incompatible.missing_keys}, '
                f'unexpected={incompatible.unexpected_keys}'
            )
        print('Successfully loaded DINOv3 ConvNeXt-Base pretrained weights')

    @staticmethod
    def _checkpoint_state_dict(checkpoint):
        if not isinstance(checkpoint, dict):
            raise TypeError('GastroNet checkpoint must contain a state dictionary')

        # Official DINO checkpoints may store a student/teacher network, while
        # downstream releases commonly use model/state_dict/backbone wrappers.
        for key in ('teacher', 'student', 'state_dict', 'model', 'backbone'):
            value = checkpoint.get(key)
            if isinstance(value, dict) and value:
                checkpoint = value
                break

        if not checkpoint or not all(torch.is_tensor(v) for v in checkpoint.values()):
            raise TypeError('Could not locate tensor weights in GastroNet checkpoint')
        return checkpoint

    @classmethod
    def _load_gastronet_resnet50(cls, model, weight_path):
        checkpoint = torch.load(weight_path, map_location='cpu')
        source = cls._checkpoint_state_dict(checkpoint)
        target = model.state_dict()
        cleaned = {}
        removable_prefixes = (
            'module.', 'backbone.', 'encoder.', 'encoder_q.', 'model.',
            'student.', 'teacher.',
        )

        for original_key, value in source.items():
            key = original_key
            changed = True
            while changed:
                changed = False
                for prefix in removable_prefixes:
                    if key.startswith(prefix):
                        key = key[len(prefix):]
                        changed = True
            if key in target and target[key].shape == value.shape:
                cleaned[key] = value

        target_numel = sum(tensor.numel() for tensor in target.values())
        loaded_numel = sum(tensor.numel() for tensor in cleaned.values())
        coverage = loaded_numel / max(target_numel, 1)
        if coverage < 0.95:
            raise RuntimeError(
                'GastroNet checkpoint is not compatible with torchvision '
                f'ResNet-50: loaded {coverage:.1%} of backbone tensors'
            )

        incompatible = model.load_state_dict(cleaned, strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(
                f'Unexpected GastroNet keys after filtering: '
                f'{incompatible.unexpected_keys}'
            )
        print(
            'Successfully loaded GastroNet-5M DINO ResNet-50 weights '
            f'({coverage:.1%} tensor coverage)'
        )

    @staticmethod
    def _load_sam2_1_hiera(model, weight_path):
        checkpoint = torch.load(weight_path, map_location='cpu')
        if not isinstance(checkpoint, dict):
            raise TypeError('SAM 2.1 checkpoint must be a state dictionary')
        source = checkpoint.get('model', checkpoint)
        if not source or not all(torch.is_tensor(v) for v in source.values()):
            raise TypeError('Could not locate tensor weights in SAM 2.1 checkpoint')

        prefix = 'image_encoder.trunk.'
        target = model.state_dict()
        cleaned = {}
        for original_key, value in source.items():
            if not original_key.startswith(prefix):
                continue
            key = 'model.' + original_key[len(prefix):]
            # Official SAM naming differs from timm's standard MLP naming.
            key = key.replace('mlp.layers.0', 'mlp.fc1')
            key = key.replace('mlp.layers.1', 'mlp.fc2')
            if key in target and target[key].shape == value.shape:
                cleaned[key] = value

        target_numel = sum(tensor.numel() for tensor in target.values())
        loaded_numel = sum(tensor.numel() for tensor in cleaned.values())
        coverage = loaded_numel / max(target_numel, 1)
        if coverage < 0.999:
            raise RuntimeError(
                'SAM 2.1 checkpoint is incompatible with the timm Hiera-B+ '
                f'image trunk: loaded {coverage:.2%} of its tensors'
            )

        incompatible = model.load_state_dict(cleaned, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                'SAM 2.1 Hiera-B+ checkpoint mismatch: '
                f'missing={incompatible.missing_keys}, '
                f'unexpected={incompatible.unexpected_keys}'
            )
        print(
            'Successfully loaded SAM 2.1 Hiera-B+ image-trunk weights '
            f'({coverage:.1%} tensor coverage)'
        )

    def _extract_backbone_features(self, x):
        if self.backbone_name == 'dinov3_convnext_base':
            return self.backbone.forward_intermediates(
                x,
                indices=(1, 2, 3, 4),
                intermediates_only=True,
            )
        if self.backbone_name in {
            'sam2_1_hiera_base_plus',
            'convnextv2_tiny_fcmae',
            'tinyvit_21m_384',
            'pvtv2_b0',
            'pvtv2_b1',
            'pvtv2_b2',
            'caformer_s18_in1k_384',
            'efficientvit_b2_r288',
            'repvit_m2_3_dist_450e',
        }:
            return self.backbone(x)

        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        stage1 = self.backbone.layer1(x)
        stage2 = self.backbone.layer2(stage1)
        stage3 = self.backbone.layer3(stage2)
        stage4 = self.backbone.layer4(stage3)
        return stage1, stage2, stage3, stage4

    @staticmethod
    def response_logits(outputs):
        """Combine raw multilevel responses without sigmoid or normalization."""
        return torch.stack(outputs, dim=0).sum(dim=0)

    def forward(
        self,
        x,
        return_response=False,
        return_decoder_features=False,
        return_quality=False,
    ):
        if return_decoder_features and self.decoder_variant != 'pff_baseline':
            raise ValueError(
                'Decoder-feature return is currently defined only for the '
                'Plain-PFF control'
            )
        is_quality_adaptive = self.decoder_variant in {
            'eqam_affine', 'eqam_supervised', 'eqam_local'
        }
        if return_quality and not is_quality_adaptive:
            raise ValueError(
                'Quality prediction is available only for EQAM decoders'
            )
        base_size = x.shape[2:]
        native_features = self._extract_backbone_features(x)
        features = [
            projection(feature)
            for projection, feature in zip(
                self.feature_projections,
                native_features,
            )
        ]

        if self.decoder_variant == 'mpfl_cfm_cam':
            enhanced = [mpfl(feat) for mpfl, feat in zip(self.mpfl, features)]
            fused = [
                self.cfm[0](enhanced[0], enhanced[1]),
                self.cfm[1](enhanced[1], enhanced[0], enhanced[2]),
                self.cfm[2](enhanced[2], enhanced[1], enhanced[3]),
                self.cfm[3](enhanced[3], enhanced[2]),
            ]

            c3, out3 = self._process_level(fused[2], fused[3], base_size)
            c2, out2 = self._process_level(fused[1], c3, base_size)
            _, out1 = self._process_level(fused[0], c2, base_size)
            outputs = (out1, out2, out3)
        else:
            if return_decoder_features:
                outputs, decoder_features = self.progressive_decoder(
                    features, base_size, return_fused=True
                )
            elif is_quality_adaptive:
                outputs, quality_prediction = self.progressive_decoder(
                    features, base_size
                )
            else:
                outputs = self.progressive_decoder(features, base_size)

        if return_decoder_features:
            response = self.response_logits(outputs)
            if return_response:
                return outputs, response, decoder_features
            return outputs, decoder_features
        if return_quality:
            if return_response:
                return outputs, self.response_logits(outputs), quality_prediction
            return outputs, quality_prediction
        if return_response:
            return outputs, self.response_logits(outputs)
        return outputs

    def _process_level(self, feat, prev_att, target_size):
        """
        Process a single level with complementary attention.
        
        Args:
            feat (torch.Tensor): Current level features
            prev_att (torch.Tensor): Previous attention map
            target_size (tuple): Target output size
            
        Returns:
            tuple: (attention_map, upsampled_output)
        """
        # Generate complementary attention
        att = self.comp_attention(feat, prev_att)
        
        # Generate final output by upsampling to target size
        output = self.final_conv(F.interpolate(att, target_size, mode='bilinear', align_corners=False))
        
        return att, output


def main():
    """
    Example usage and model testing.
    """
    # Create model instance
    model = PolypPVT(channel=64)
    
    # Create dummy input
    batch_size, channels, height, width = 4, 3, 352, 352
    dummy_input = torch.randn(batch_size, channels, height, width)
    
    # Forward pass
    print("Testing PolypPVT model...")
    print(f"Input shape: {dummy_input.shape}")
    
    with torch.no_grad():
        out1, out2, out3 = model(dummy_input)
        print(f"Output 1 (finest) shape: {out1.shape}")
        print(f"Output 2 (medium) shape: {out2.shape}")
        print(f"Output 3 (coarsest) shape: {out3.shape}")
    
    # Calculate model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Statistics:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")


if __name__ == "__main__":
    main()

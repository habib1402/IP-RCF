"""Frequency-decoupled region-boundary progressive feature fusion.

The module implements a controlled three-arm screen on top of PVTv2-B0:

``boundary_aux``
    Plain-PFF segmentation with auxiliary boundary supervision only.
``dual_band``
    Separate low-frequency region and high-frequency boundary streams with
    learned, zero-initialized boundary injection.
``frb_full``
    The same dual streams with uncertainty-conditioned boundary injection.

The frequency split is parameter-free.  Unlike the earlier BC-MGPF study,
this decoder does not route among convolution geometries.  It maintains an
explicit boundary representation, supervises it, and lets it modify the
region stream only through bounded residual gains that start at zero.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pvt import DepthwiseSeparableRefinement, PolypPVT


FRBPFF_VARIANTS = ('boundary_aux', 'dual_band', 'frb_full')


def split_frequency_bands(feature):
    """Return same-resolution low- and high-frequency feature components."""
    low = F.avg_pool2d(feature, kernel_size=3, stride=1, padding=1)
    high = feature - low
    return low, high


class AuxiliaryBoundaryHead(nn.Module):
    """Shared lightweight head applied to high-frequency encoder features."""

    def __init__(self, channels):
        super().__init__()
        self.head = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, features, target_size):
        logits = []
        for feature in features[:3]:
            _low, high = split_frequency_bands(feature)
            logits.append(
                F.interpolate(
                    self.head(high),
                    size=target_size,
                    mode='bilinear',
                    align_corners=False,
                )
            )
        return tuple(logits)


class FrequencyDecoupledProgressiveDecoder(nn.Module):
    """Progressive region/boundary decoder with optional uncertainty gating."""

    def __init__(self, channels, uncertainty_conditioned=False):
        super().__init__()
        self.uncertainty_conditioned = bool(uncertainty_conditioned)
        self.region_refinements = nn.ModuleList([
            DepthwiseSeparableRefinement(channels) for _ in range(4)
        ])
        self.boundary_refinements = nn.ModuleList([
            DepthwiseSeparableRefinement(channels) for _ in range(4)
        ])
        self.prediction_head = nn.Conv2d(channels, 1, kernel_size=1)
        self.boundary_head = nn.Conv2d(channels, 1, kernel_size=1)

        # Exact Plain-PFF-like region behavior at initialization.  Boundary
        # gradients still reach the boundary stream through its auxiliary
        # supervision while these gains learn whether injection is useful.
        self.injection_gain_logits = nn.Parameter(torch.zeros(3))
        self.last_injection_gains = None
        self.last_gate_means = None

    @property
    def injection_gains(self):
        # Bounded to [-0.5, 0.5] to prevent the detail stream from dominating.
        return 0.5 * torch.tanh(self.injection_gain_logits)

    @staticmethod
    def _resize(feature, reference):
        return F.interpolate(
            feature,
            size=reference.shape[2:],
            mode='bilinear',
            align_corners=False,
        )

    def _injection_gate(self, coarse_region, boundary):
        if not self.uncertainty_conditioned:
            return boundary.new_ones(
                (boundary.shape[0], 1, boundary.shape[2], boundary.shape[3])
            )
        coarse_logit = self.prediction_head(coarse_region)
        probability = torch.sigmoid(coarse_logit)
        uncertainty = 4.0 * probability * (1.0 - probability)
        boundary_probability = torch.sigmoid(self.boundary_head(boundary))
        return uncertainty * boundary_probability

    def forward(
        self,
        features,
        target_size,
        return_details=False,
        return_boundary_outputs=True,
    ):
        region = [None] * 4
        boundary = [None] * 4
        fused = [None] * 4

        low, high = split_frequency_bands(features[3])
        region[3] = self.region_refinements[3](low)
        boundary[3] = self.boundary_refinements[3](high)
        fused[3] = region[3]

        gate_means = []
        for gain_index, level in enumerate((2, 1, 0)):
            low, high = split_frequency_bands(features[level])
            coarse_region = self._resize(fused[level + 1], low)
            coarse_boundary = self._resize(boundary[level + 1], high)
            region[level] = self.region_refinements[level](
                low + coarse_region
            )
            boundary[level] = self.boundary_refinements[level](
                high + coarse_boundary
            )
            gate = self._injection_gate(coarse_region, boundary[level])
            gain = self.injection_gains[gain_index]
            fused[level] = region[level] + gain * gate * boundary[level]
            gate_means.append(gate.detach().mean())

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
        boundary_outputs = (
            tuple(
                self.boundary_head(
                    F.interpolate(
                        boundary[level],
                        size=target_size,
                        mode='bilinear',
                        align_corners=False,
                    )
                )
                for level in (0, 1, 2)
            )
            if return_boundary_outputs
            else tuple()
        )
        self.last_injection_gains = self.injection_gains.detach().cpu()
        self.last_gate_means = torch.stack(gate_means).cpu()
        if return_details:
            return outputs, boundary_outputs, {
                'injection_gains': self.injection_gains,
                'gate_means': torch.stack(gate_means),
            }
        return outputs, boundary_outputs


class FRBPFFStudent(nn.Module):
    """PVTv2-B0 student for the controlled FRB-PFF screen."""

    def __init__(
        self,
        variant,
        channel=64,
        backbone='pvtv2_b0',
        backbone_weights=None,
        load_backbone_weights=True,
    ):
        super().__init__()
        if variant not in FRBPFF_VARIANTS:
            raise ValueError(
                'Unknown FRB-PFF variant {!r}; choose from {}'.format(
                    variant, FRBPFF_VARIANTS
                )
            )
        self.variant = variant
        self.base = PolypPVT(
            channel=channel,
            backbone=backbone,
            decoder_variant='pff_baseline',
            backbone_weights=backbone_weights,
            load_backbone_weights=load_backbone_weights,
        )
        if variant == 'boundary_aux':
            self.boundary_decoder = AuxiliaryBoundaryHead(channel)
        else:
            self.base.progressive_decoder = (
                FrequencyDecoupledProgressiveDecoder(
                    channel,
                    uncertainty_conditioned=(variant == 'frb_full'),
                )
            )
            self.boundary_decoder = None

    @property
    def backbone(self):
        return self.base.backbone

    @property
    def backbone_name(self):
        return self.base.backbone_name

    @property
    def backbone_weights_path(self):
        return self.base.backbone_weights_path

    @property
    def decoder_variant(self):
        return 'frbpff_{}'.format(self.variant)

    @staticmethod
    def response_logits(outputs):
        return PolypPVT.response_logits(outputs)

    def forward(self, images, return_details=False):
        target_size = images.shape[2:]
        native_features = self.base._extract_backbone_features(images)
        features = [
            projection(feature)
            for projection, feature in zip(
                self.base.feature_projections, native_features
            )
        ]
        if self.variant == 'boundary_aux':
            outputs = self.base.progressive_decoder(features, target_size)
            boundary_outputs = (
                self.boundary_decoder(features, target_size)
                if return_details
                else tuple()
            )
            details = {
                'injection_gains': images.new_zeros(3),
                'gate_means': images.new_zeros(3),
            }
        else:
            outputs, boundary_outputs, details = (
                self.base.progressive_decoder(
                    features,
                    target_size,
                    return_details=True,
                    return_boundary_outputs=return_details,
                )
            )
        response = self.response_logits(outputs)
        result = {
            'outputs': outputs,
            'response': response,
            'boundary_outputs': boundary_outputs,
            'injection_gains': details['injection_gains'],
            'gate_means': details['gate_means'],
        }
        if return_details:
            return result
        return outputs


def build_frbpff_model(
    variant,
    channel=64,
    backbone='pvtv2_b0',
    backbone_weights=None,
    load_backbone_weights=True,
):
    return FRBPFFStudent(
        variant=variant,
        channel=channel,
        backbone=backbone,
        backbone_weights=backbone_weights,
        load_backbone_weights=load_backbone_weights,
    )

"""Identity-preserving residual contour fusion for polyp segmentation.

The Plain-PFF semantic pathway is retained verbatim.  A supervised contour
path can only add a bounded, zero-initialized residual to the three decoded
feature levels, so the module starts as Plain-PFF and cannot discard the full
band semantic representation as the rejected dual-band decoder did.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pvt import DepthwiseSeparableRefinement, PolypPVT


IPRCF_VARIANTS = (
    'finest_residual',
    'multiscale_residual',
    'multiscale_consistency',
)


def local_contour_residual(feature):
    """Parameter-free local high-frequency residual at unchanged resolution."""
    local_mean = F.avg_pool2d(feature, kernel_size=3, stride=1, padding=1)
    return feature - local_mean


class IdentityPreservingResidualContourDecoder(nn.Module):
    """Plain-PFF with a parallel, bounded residual contour correction."""

    def __init__(self, channels, multiscale, semantic_decoder):
        super().__init__()
        self.multiscale = bool(multiscale)

        # Reuse the already initialized Plain-PFF modules themselves.  With
        # zero contour gains, both parameters and computation are therefore
        # exactly the corresponding Plain-PFF semantic pathway.
        self.semantic_refinements = semantic_decoder.refinements
        self.prediction_head = semantic_decoder.prediction_head

        contour_levels = 4 if self.multiscale else 1
        self.contour_refinements = nn.ModuleList([
            DepthwiseSeparableRefinement(channels)
            for _ in range(contour_levels)
        ])
        self.contour_head = nn.Conv2d(channels, 1, kernel_size=1)
        gain_count = 3 if self.multiscale else 1
        self.gain_logits = nn.Parameter(torch.zeros(gain_count))
        self.last_injection_gains = None

    @property
    def bounded_gains(self):
        # Conservative bound: contour evidence may correct but not dominate.
        return 0.25 * torch.tanh(self.gain_logits)

    @staticmethod
    def _resize(feature, reference):
        return F.interpolate(
            feature,
            size=reference.shape[2:],
            mode='bilinear',
            align_corners=False,
        )

    def _semantic_path(self, features):
        semantic = [None] * 4
        semantic[3] = self.semantic_refinements[3](features[3])
        for level in (2, 1, 0):
            coarse = self._resize(semantic[level + 1], features[level])
            semantic[level] = self.semantic_refinements[level](
                features[level] + coarse
            )
        return semantic

    def _contour_path(self, features):
        if not self.multiscale:
            return {
                0: self.contour_refinements[0](
                    local_contour_residual(features[0])
                )
            }

        contour = [None] * 4
        contour[3] = self.contour_refinements[3](
            local_contour_residual(features[3])
        )
        for level in (2, 1, 0):
            high_frequency = local_contour_residual(features[level])
            coarse_contour = self._resize(
                contour[level + 1], high_frequency
            )
            contour[level] = self.contour_refinements[level](
                high_frequency + coarse_contour
            )
        return {level: contour[level] for level in (0, 1, 2)}

    def forward(self, features, target_size, return_details=False):
        semantic = self._semantic_path(features)
        contour = self._contour_path(features)

        corrected = list(semantic)
        if self.multiscale:
            for gain_index, level in enumerate((0, 1, 2)):
                corrected[level] = (
                    semantic[level]
                    + self.bounded_gains[gain_index] * contour[level]
                )
            reported_gains = self.bounded_gains
        else:
            corrected[0] = (
                semantic[0] + self.bounded_gains[0] * contour[0]
            )
            reported_gains = torch.stack((
                self.bounded_gains[0],
                self.bounded_gains[0] * 0.0,
                self.bounded_gains[0] * 0.0,
            ))

        outputs = tuple(
            self.prediction_head(
                F.interpolate(
                    corrected[level],
                    size=target_size,
                    mode='bilinear',
                    align_corners=False,
                )
            )
            for level in (0, 1, 2)
        )
        self.last_injection_gains = reported_gains.detach().cpu()
        if not return_details:
            return outputs

        boundary_outputs = tuple(
            F.interpolate(
                self.contour_head(contour[level]),
                size=target_size,
                mode='bilinear',
                align_corners=False,
            )
            for level in sorted(contour)
        )
        return outputs, boundary_outputs, reported_gains


class IPRCFStudent(nn.Module):
    """PVTv2-B0 student for the controlled IP-RCF screen."""

    def __init__(
        self,
        variant,
        channel=64,
        backbone='pvtv2_b0',
        backbone_weights=None,
        load_backbone_weights=True,
    ):
        super().__init__()
        if variant not in IPRCF_VARIANTS:
            raise ValueError(
                'Unknown IP-RCF variant {!r}; choose from {}'.format(
                    variant, IPRCF_VARIANTS
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
        plain_decoder = self.base.progressive_decoder
        self.base.progressive_decoder = (
            IdentityPreservingResidualContourDecoder(
                channel,
                multiscale=variant != 'finest_residual',
                semantic_decoder=plain_decoder,
            )
        )

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
        return 'iprcf_{}'.format(self.variant)

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
        if return_details:
            outputs, boundary_outputs, gains = self.base.progressive_decoder(
                features, target_size, return_details=True
            )
            return {
                'outputs': outputs,
                'response': self.response_logits(outputs),
                'boundary_outputs': boundary_outputs,
                'injection_gains': gains,
                'gate_means': gains.detach() * 0.0,
            }
        return self.base.progressive_decoder(
            features, target_size, return_details=False
        )


def build_iprcf_model(
    variant,
    channel=64,
    backbone='pvtv2_b0',
    backbone_weights=None,
    load_backbone_weights=True,
):
    return IPRCFStudent(
        variant=variant,
        channel=channel,
        backbone=backbone,
        backbone_weights=backbone_weights,
        load_backbone_weights=load_backbone_weights,
    )

"""Lightweight multi-scale polyp-evidence calibration models.

The segmentation trunk remains the controlled PVTv2-B0 + Plain-PFF model.
Presence variants add a tiny image-level evidence head and calibrate the
combined segmentation logit with one bounded scalar.  No connected-component
operation or external-test information is part of the network.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.pvt import PolypPVT


PRESENCE_VARIANTS = ('deep_presence', 'mpec')


class DeepSemanticPresenceHead(nn.Module):
    """Predict polyp presence from pooled stride-32 semantic evidence."""

    def __init__(self, channels=64, hidden_channels=32):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(2 * channels),
            nn.Linear(2 * channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, deep_feature, fine_feature=None, outputs=None):
        average = F.adaptive_avg_pool2d(deep_feature, 1).flatten(1)
        maximum = F.adaptive_max_pool2d(deep_feature, 1).flatten(1)
        return self.classifier(torch.cat((average, maximum), dim=1)).squeeze(1)


class MultiScalePolypEvidenceHead(nn.Module):
    """Fuse fine, semantic and cross-output agreement evidence."""

    OUTPUT_STATISTICS = 9

    def __init__(self, channels=64, hidden_channels=64):
        super().__init__()
        input_channels = 4 * channels + self.OUTPUT_STATISTICS
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_channels),
            nn.Linear(input_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, 1),
        )

    @staticmethod
    def _pooled(feature):
        return torch.cat(
            (
                F.adaptive_avg_pool2d(feature, 1).flatten(1),
                F.adaptive_max_pool2d(feature, 1).flatten(1),
            ),
            dim=1,
        )

    @staticmethod
    def _output_statistics(outputs):
        probabilities = [torch.sigmoid(output.float()) for output in outputs]
        means = [value.mean(dim=(1, 2, 3)) for value in probabilities]
        maxima = [value.amax(dim=(1, 2, 3)) for value in probabilities]
        disagreements = []
        for first, second in ((0, 1), (0, 2), (1, 2)):
            disagreements.append(
                (probabilities[first] - probabilities[second])
                .abs()
                .mean(dim=(1, 2, 3))
            )
        return torch.stack(means + maxima + disagreements, dim=1)

    def forward(self, deep_feature, fine_feature, outputs):
        evidence = torch.cat(
            (
                self._pooled(fine_feature).float(),
                self._pooled(deep_feature).float(),
                self._output_statistics(outputs),
            ),
            dim=1,
        )
        return self.classifier(evidence).squeeze(1)


class PresenceCalibratedPolypPVT(PolypPVT):
    """Plain-PFF with a lightweight image-level presence calibration head.

    The base segmentation response is the sum of three full-resolution logits.
    A bounded, learned calibration strength shifts that response using the
    presence logit.  A neutral presence logit (zero) leaves segmentation
    unchanged, and the initial strength is deliberately small.
    """

    def __init__(
        self,
        channel=64,
        backbone='pvtv2_b0',
        presence_variant='mpec',
        backbone_weights=None,
        load_backbone_weights=True,
        initial_calibration_strength=0.10,
    ):
        if presence_variant not in PRESENCE_VARIANTS:
            raise ValueError(
                'Unsupported presence variant {!r}; choose from {}'.format(
                    presence_variant, PRESENCE_VARIANTS
                )
            )
        if not 0.0 < initial_calibration_strength < 1.0:
            raise ValueError('Initial calibration strength must be in (0, 1)')
        super().__init__(
            channel=channel,
            backbone=backbone,
            decoder_variant='pff_baseline',
            backbone_weights=backbone_weights,
            load_backbone_weights=load_backbone_weights,
        )
        self.presence_variant = presence_variant
        if presence_variant == 'deep_presence':
            self.presence_head = DeepSemanticPresenceHead(channel)
        else:
            self.presence_head = MultiScalePolypEvidenceHead(channel)
        initial_logit = math.log(
            initial_calibration_strength / (1.0 - initial_calibration_strength)
        )
        self.calibration_strength_logit = nn.Parameter(
            torch.tensor(initial_logit, dtype=torch.float32)
        )

    @property
    def calibration_strength(self):
        return torch.sigmoid(self.calibration_strength_logit)

    def forward(self, x, return_details=False):
        base_size = x.shape[2:]
        native_features = self._extract_backbone_features(x)
        features = [
            projection(feature)
            for projection, feature in zip(
                self.feature_projections, native_features
            )
        ]
        outputs, fine_feature = self.progressive_decoder(
            features, base_size, return_fused=True
        )
        presence_logit = self.presence_head(
            features[3], fine_feature, outputs
        )
        base_response = self.response_logits(outputs)
        calibrated_response = base_response + (
            self.calibration_strength
            * presence_logit[:, None, None, None]
        )
        if return_details:
            return {
                'outputs': outputs,
                'base_response': base_response,
                'response': calibrated_response,
                'presence_logit': presence_logit,
                'fine_feature': fine_feature,
                'deep_feature': features[3],
            }
        return outputs, calibrated_response, presence_logit


def build_mpec_model(
    variant,
    channel=64,
    backbone='pvtv2_b0',
    backbone_weights=None,
    load_backbone_weights=True,
    initial_calibration_strength=0.10,
):
    """Build one of the three controlled inference graphs."""
    if variant == 'data_control':
        return PolypPVT(
            channel=channel,
            backbone=backbone,
            decoder_variant='pff_baseline',
            backbone_weights=backbone_weights,
            load_backbone_weights=load_backbone_weights,
        )
    if variant not in PRESENCE_VARIANTS:
        raise ValueError(
            'Unknown MPEC study variant {!r}'.format(variant)
        )
    return PresenceCalibratedPolypPVT(
        channel=channel,
        backbone=backbone,
        presence_variant=variant,
        backbone_weights=backbone_weights,
        load_backbone_weights=load_backbone_weights,
        initial_calibration_strength=initial_calibration_strength,
    )


def forward_mpec_model(model, variant, images):
    """Return outputs, inference response and optional presence logit."""
    if variant == 'data_control':
        outputs = model(images)
        return outputs, model.response_logits(outputs), None
    return model(images)

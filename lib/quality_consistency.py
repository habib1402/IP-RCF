"""Training-only endoscopy-quality perturbations and consistency losses.

Nothing in this module is attached to the segmentation model.  Consequently,
all four experiments have an identical PVTv2-B0 + Plain-PFF inference graph.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EndoscopyArtifactSimulator(nn.Module):
    """Create a label-preserving degraded view of an ImageNet-normalized batch.

    The simulator models illumination/colour variation, blur, reduced spatial
    detail, specular highlights, and sensor/compression-like noise.  It changes
    appearance only; no additional geometry is applied, so masks and cached
    teacher responses remain spatially aligned.
    """

    def __init__(self, strength=0.5):
        super().__init__()
        if not 0.0 <= strength <= 1.0:
            raise ValueError('Artifact strength must be in [0, 1]')
        self.strength = float(strength)
        self.register_buffer(
            'mean', torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std', torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        )

    @staticmethod
    def _uniform(shape, lower, upper, reference):
        return torch.empty(shape, device=reference.device, dtype=reference.dtype).uniform_(
            lower, upper
        )

    def forward(self, normalized_images):
        if normalized_images.ndim != 4 or normalized_images.size(1) != 3:
            raise ValueError('Expected normalized RGB images with shape [B,3,H,W]')
        if self.strength == 0.0:
            batch = normalized_images.size(0)
            return normalized_images.clone(), {
                'mean_absolute_perturbation': 0.0,
                'specular_fraction': 0.0,
                'blur_fraction': 0.0,
                'resolution_fraction': 0.0,
                'quality_targets': normalized_images.new_zeros(batch, 5),
            }

        clean = (normalized_images * self.std + self.mean).clamp(0.0, 1.0)
        degraded = clean.clone()
        batch, _channels, height, width = degraded.shape
        strength = self.strength

        # Per-sample photometric variation, present in every degraded view.
        brightness = self._uniform(
            (batch, 1, 1, 1), 1.0 - 0.30 * strength,
            1.0 + 0.30 * strength, degraded,
        )
        contrast = self._uniform(
            (batch, 1, 1, 1), 1.0 - 0.25 * strength,
            1.0 + 0.25 * strength, degraded,
        )
        gamma = self._uniform(
            (batch, 1, 1, 1), 1.0 - 0.35 * strength,
            1.0 + 0.35 * strength, degraded,
        )
        colour_gain = self._uniform(
            (batch, 3, 1, 1), 1.0 - 0.15 * strength,
            1.0 + 0.15 * strength, degraded,
        )
        degraded = ((degraded - 0.5) * contrast + 0.5).clamp(0.0, 1.0)
        degraded = (degraded * brightness * colour_gain).clamp(0.0, 1.0)
        degraded = degraded.clamp_min(1e-6).pow(gamma).clamp(0.0, 1.0)

        # Defocus/motion proxy.  The mask is sampled per image, while the
        # pooling kernel is shared within a batch to keep the operator cheap.
        blur_mask = (torch.rand(batch, 1, 1, 1, device=degraded.device) < 0.5).to(
            degraded.dtype
        )
        kernel = 5 if strength >= 0.65 else 3
        blurred = F.avg_pool2d(degraded, kernel, stride=1, padding=kernel // 2)
        degraded = degraded * (1.0 - blur_mask) + blurred * blur_mask

        # Reduced-detail/compression proxy through downsample-upsample.
        resolution_mask = (
            torch.rand(batch, 1, 1, 1, device=degraded.device) < 0.5
        ).to(degraded.dtype)
        minimum_scale = 1.0 - 0.65 * strength
        scale = float(torch.empty((), device=degraded.device).uniform_(minimum_scale, 1.0))
        low_height = max(8, int(round(height * scale)))
        low_width = max(8, int(round(width * scale)))
        low_detail = F.interpolate(
            degraded, size=(low_height, low_width), mode='bilinear', align_corners=False
        )
        low_detail = F.interpolate(
            low_detail, size=(height, width), mode='bilinear', align_corners=False
        )
        degraded = degraded * (1.0 - resolution_mask) + low_detail * resolution_mask

        # Smooth white blobs approximate saturated endoscopic highlights.
        y = torch.linspace(0.0, 1.0, height, device=degraded.device, dtype=degraded.dtype)
        x = torch.linspace(0.0, 1.0, width, device=degraded.device, dtype=degraded.dtype)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        grid_y = grid_y.view(1, 1, height, width)
        grid_x = grid_x.view(1, 1, height, width)
        centre_y = self._uniform((batch, 1, 1, 1), 0.1, 0.9, degraded)
        centre_x = self._uniform((batch, 1, 1, 1), 0.1, 0.9, degraded)
        radius = self._uniform(
            (batch, 1, 1, 1), 0.025, 0.025 + 0.075 * strength, degraded
        )
        specular = torch.exp(
            -((grid_y - centre_y).square() + (grid_x - centre_x).square())
            / (2.0 * radius.square().clamp_min(1e-6))
        )
        specular_enabled = (
            torch.rand(batch, 1, 1, 1, device=degraded.device) < 0.5
        ).to(degraded.dtype)
        specular = specular * specular_enabled * (0.35 + 0.50 * strength)
        degraded = degraded + specular * (1.0 - degraded)

        # A mild noise term also approximates codec residue without using a
        # non-differentiable JPEG round trip.
        noise_std = self._uniform(
            (batch, 1, 1, 1), 0.0, 0.04 * strength, degraded
        )
        degraded = (degraded + torch.randn_like(degraded) * noise_std).clamp(0.0, 1.0)

        # Normalized per-image targets for the training-only EQAM quality
        # supervision: photometric shift, blur, resolution loss, highlight,
        # and noise.  Values are bounded in [0, 1] by construction.
        photometric_severity = torch.stack(
            (
                (brightness - 1.0).abs().view(batch)
                / max(0.30 * strength, 1e-6),
                (contrast - 1.0).abs().view(batch)
                / max(0.25 * strength, 1e-6),
                (gamma - 1.0).abs().view(batch)
                / max(0.35 * strength, 1e-6),
                (colour_gain - 1.0).abs().mean(dim=(1, 2, 3))
                / max(0.15 * strength, 1e-6),
            ),
            dim=1,
        ).mean(dim=1, keepdim=True)
        blur_severity = blur_mask.view(batch, 1)
        resolution_severity = (
            resolution_mask.view(batch, 1)
            * ((1.0 - scale) / max(0.65 * strength, 1e-6))
        )
        specular_severity = specular_enabled.view(batch, 1)
        noise_severity = noise_std.view(batch, 1) / max(0.04 * strength, 1e-6)
        quality_targets = torch.cat(
            (
                photometric_severity,
                blur_severity,
                resolution_severity,
                specular_severity,
                noise_severity,
            ),
            dim=1,
        ).clamp(0.0, 1.0)

        statistics = {
            'mean_absolute_perturbation': float(
                (degraded - clean).abs().mean().detach().item()
            ),
            'specular_fraction': float(
                (specular > 0.20).to(degraded.dtype).mean().detach().item()
            ),
            'blur_fraction': float(blur_mask.mean().detach().item()),
            'resolution_fraction': float(resolution_mask.mean().detach().item()),
            'quality_targets': quality_targets.detach(),
        }
        normalized = (degraded - self.mean) / self.std
        return normalized, statistics


def weighted_soft_structure_loss(student_logit, target_probability, reliability):
    """Reliability-weighted soft BCE plus soft Dice."""
    student_logit = student_logit.float()
    target_probability = target_probability.float().detach().clamp(0.0, 1.0)
    reliability = reliability.float().detach().clamp_min(0.0)
    denominator = reliability.sum(dim=(1, 2, 3)).clamp_min(1e-6)
    pixel_bce = F.binary_cross_entropy_with_logits(
        student_logit, target_probability, reduction='none'
    )
    weighted_bce = (
        (pixel_bce * reliability).sum(dim=(1, 2, 3)) / denominator
    ).mean()
    student_probability = torch.sigmoid(student_logit)
    intersection = (
        reliability * student_probability * target_probability
    ).sum(dim=(1, 2, 3))
    prediction_mass = (reliability * student_probability).sum(dim=(1, 2, 3))
    target_mass = (reliability * target_probability).sum(dim=(1, 2, 3))
    weighted_dice = (
        1.0 - (2.0 * intersection + 1.0)
        / (prediction_mass + target_mass + 1.0)
    ).mean()
    return weighted_bce + weighted_dice


def student_clean_to_degraded_consistency(clean_logit, degraded_logit):
    """Use a detached, confidence-weighted clean prediction as the target."""
    with torch.no_grad():
        target = torch.sigmoid(clean_logit.float())
        confidence = torch.abs(2.0 * target - 1.0)
        # Retain a small floor so uncertain regions are not silently discarded.
        reliability = 0.10 + 0.90 * confidence
    loss = weighted_soft_structure_loss(degraded_logit, target, reliability)
    return loss, {
        'reliability': float(reliability.mean().item()),
        'target_foreground_probability': float(target.mean().item()),
    }


def _morphological_region_partition(mask, radius=3):
    """Split a binary mask into soft interior, boundary, and exterior regions."""
    if radius < 1:
        raise ValueError('Morphological radius must be positive')
    mask = mask.float().clamp(0.0, 1.0)
    kernel = 2 * int(radius) + 1
    dilated = F.max_pool2d(mask, kernel, stride=1, padding=radius)
    eroded = -F.max_pool2d(-mask, kernel, stride=1, padding=radius)
    interior = eroded.clamp(0.0, 1.0)
    boundary = (dilated - eroded).clamp(0.0, 1.0)
    exterior = (1.0 - dilated).clamp(0.0, 1.0)
    return interior, boundary, exterior


def cross_quality_boundary_consistency(
    clean_logit,
    degraded_logit,
    ground_truth,
    boundary_radius=3,
):
    """Distil clean-view contour probabilities into the degraded view.

    The ground-truth-derived band only selects where consistency is useful;
    the detached clean prediction supplies the soft probability and contour
    targets.  This avoids the background domination of global response
    consistency while keeping the supervised segmentation objective fixed.
    """
    clean_logit = clean_logit.float()
    degraded_logit = degraded_logit.float()
    with torch.no_grad():
        clean_probability = torch.sigmoid(clean_logit)
        _interior, boundary, _exterior = _morphological_region_partition(
            ground_truth, radius=boundary_radius
        )
        clean_dilated = F.max_pool2d(clean_probability, 3, stride=1, padding=1)
        clean_eroded = -F.max_pool2d(
            -clean_probability, 3, stride=1, padding=1
        )
        clean_edge = (clean_dilated - clean_eroded).clamp(0.0, 1.0)
        reliability = boundary * (0.25 + 0.75 * clean_edge)

    denominator = reliability.sum(dim=(1, 2, 3)).clamp_min(1e-6)
    probability_bce = F.binary_cross_entropy_with_logits(
        degraded_logit, clean_probability, reduction='none'
    )
    probability_loss = (
        (probability_bce * reliability).sum(dim=(1, 2, 3)) / denominator
    ).mean()

    degraded_probability = torch.sigmoid(degraded_logit)
    degraded_dilated = F.max_pool2d(
        degraded_probability, 3, stride=1, padding=1
    )
    degraded_eroded = -F.max_pool2d(
        -degraded_probability, 3, stride=1, padding=1
    )
    degraded_edge = (degraded_dilated - degraded_eroded).clamp(0.0, 1.0)
    edge_error = F.smooth_l1_loss(
        degraded_edge, clean_edge, reduction='none'
    )
    edge_loss = (
        (edge_error * boundary).sum(dim=(1, 2, 3))
        / boundary.sum(dim=(1, 2, 3)).clamp_min(1e-6)
    ).mean()
    loss = probability_loss + edge_loss
    return loss, {
        'reliability': float(reliability.mean().detach().item()),
        'boundary_fraction': float(boundary.mean().detach().item()),
        'boundary_component': float(loss.detach().item()),
        'prototype_component': 0.0,
    }


def cross_quality_region_prototype_consistency(
    clean_feature,
    degraded_feature,
    ground_truth,
    boundary_radius=3,
    temperature=0.1,
):
    """Align clean/degraded interior, boundary, and exterior prototypes.

    Only the degraded-view prototype receives gradients.  Region averaging
    makes the objective insensitive to local artifact pixels while explicitly
    preserving the three segmentation-relevant feature populations.
    """
    if clean_feature.shape != degraded_feature.shape:
        raise ValueError('Clean and degraded decoder features must match')
    if clean_feature.ndim != 4:
        raise ValueError('Expected decoder features with shape [B,C,H,W]')
    if temperature <= 0.0:
        raise ValueError('Prototype temperature must be positive')
    with torch.no_grad():
        regions = _morphological_region_partition(
            ground_truth, radius=boundary_radius
        )
        region_weights = [
            F.interpolate(
                region,
                size=clean_feature.shape[2:],
                mode='bilinear',
                align_corners=False,
            ).clamp(0.0, 1.0)
            for region in regions
        ]

    clean_feature = clean_feature.float().detach()
    degraded_feature = degraded_feature.float()
    clean_prototypes = []
    degraded_prototypes = []
    valid_masks = []
    for weights in region_weights:
        mass = weights.sum(dim=(2, 3)).clamp_min(1e-6)
        clean_prototype = (clean_feature * weights).sum(dim=(2, 3)) / mass
        degraded_prototype = (degraded_feature * weights).sum(dim=(2, 3)) / mass
        clean_prototypes.append(F.normalize(clean_prototype, dim=1, eps=1e-6))
        degraded_prototypes.append(
            F.normalize(degraded_prototype, dim=1, eps=1e-6)
        )
        valid_masks.append(mass.squeeze(1) > 1.0)

    # Each degraded-view region must retrieve its corresponding detached
    # clean-view region.  The off-diagonal logits explicitly discourage
    # interior/boundary/exterior feature collapse, unlike a tiny raw cosine
    # penalty between already similar paired prototypes.
    clean_prototypes = torch.stack(clean_prototypes, dim=1)
    degraded_prototypes = torch.stack(degraded_prototypes, dim=1)
    logits = torch.bmm(
        degraded_prototypes,
        clean_prototypes.transpose(1, 2),
    ) / float(temperature)
    batch = logits.size(0)
    targets = torch.arange(3, device=logits.device).view(1, 3).expand(batch, 3)
    per_region_loss = F.cross_entropy(
        logits.reshape(-1, 3),
        targets.reshape(-1),
        reduction='none',
    ).view(batch, 3)
    valid = torch.stack(valid_masks, dim=1).to(per_region_loss.dtype)
    loss = (per_region_loss * valid).sum() / valid.sum().clamp_min(1.0)
    valid_regions = int(valid.sum().detach().item())
    return loss, {
        'reliability': float(valid_regions / max(3 * clean_feature.size(0), 1)),
        'boundary_fraction': float(region_weights[1].mean().detach().item()),
        'boundary_component': 0.0,
        'prototype_component': float(loss.detach().item()),
    }


def combined_cross_quality_shape_consistency(
    clean_logit,
    degraded_logit,
    clean_feature,
    degraded_feature,
    ground_truth,
    boundary_radius=3,
    prototype_temperature=0.1,
):
    """Combine boundary and region-prototype terms at a fixed total scale."""
    boundary_loss, boundary_statistics = cross_quality_boundary_consistency(
        clean_logit,
        degraded_logit,
        ground_truth,
        boundary_radius=boundary_radius,
    )
    prototype_loss, prototype_statistics = (
        cross_quality_region_prototype_consistency(
            clean_feature,
            degraded_feature,
            ground_truth,
            boundary_radius=boundary_radius,
            temperature=prototype_temperature,
        )
    )
    loss = 0.5 * (boundary_loss + prototype_loss)
    return loss, {
        'reliability': 0.5 * (
            boundary_statistics['reliability']
            + prototype_statistics['reliability']
        ),
        'boundary_fraction': boundary_statistics['boundary_fraction'],
        'boundary_component': float(boundary_loss.detach().item()),
        'prototype_component': float(prototype_loss.detach().item()),
    }


def compatible_dual_teacher_consistency(
    degraded_logit,
    ground_truth,
    general_logit,
    domain_logit,
    general_temperature,
    domain_temperature,
    agreement_power=1.0,
    confidence_power=1.0,
    compatibility_power=2.0,
):
    """Guide the degraded student only where calibrated teachers are reliable.

    Unlike the earlier global response-KD experiment, reliability here also
    requires teacher/annotation compatibility.  This prevents a confident but
    wrong teacher consensus from overriding supervised training.
    """
    with torch.no_grad():
        general_probability = torch.sigmoid(
            general_logit.float() / general_temperature
        )
        domain_probability = torch.sigmoid(
            domain_logit.float() / domain_temperature
        )
        target = 0.5 * (general_probability + domain_probability)
        disagreement = torch.abs(general_probability - domain_probability)
        agreement = (1.0 - disagreement).clamp(0.0, 1.0)
        general_confidence = torch.abs(2.0 * general_probability - 1.0)
        domain_confidence = torch.abs(2.0 * domain_probability - 1.0)
        joint_confidence = torch.minimum(general_confidence, domain_confidence)
        compatibility = (1.0 - torch.abs(target - ground_truth.float())).clamp(
            0.0, 1.0
        )
        reliability = (
            agreement.pow(agreement_power)
            * joint_confidence.pow(confidence_power)
            * compatibility.pow(compatibility_power)
        )
    loss = weighted_soft_structure_loss(degraded_logit, target, reliability)
    return loss, {
        'reliability': float(reliability.mean().item()),
        'disagreement': float(disagreement.mean().item()),
        'compatibility': float(compatibility.mean().item()),
        'target_foreground_probability': float(target.mean().item()),
    }

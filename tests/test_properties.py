"""CPU checks of the two formal properties stated in the paper.

Run:  python -m pytest tests -q      (or)      python tests/test_properties.py
No pretrained weights or data are needed; random initialization is used.
"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from lib.iprcf import build_iprcf_model  # noqa: E402
from lib.pvt import PolypPVT  # noqa: E402


def _models(seed=0):
    torch.manual_seed(seed)
    ip = build_iprcf_model('multiscale_residual', channel=64, backbone='pvtv2_b0',
                           load_backbone_weights=False).eval()
    # give the contour branch non-trivial weights
    with torch.no_grad():
        for name, p in ip.named_parameters():
            if 'contour' in name:
                p.normal_(0, 0.05)
    plain = PolypPVT(channel=64, backbone='pvtv2_b0', decoder_variant='pff_baseline',
                     load_backbone_weights=False).eval()
    mapped = {}
    for k, v in ip.state_dict().items():
        if 'contour' in k or 'gain_logits' in k:
            continue
        k2 = k[len('base.'):] if k.startswith('base.') else k
        mapped[k2.replace('semantic_refinements', 'refinements')] = v
    missing, unexpected = plain.load_state_dict(mapped, strict=True), None
    return ip, plain


def _logit(model, x):
    out = model(x)
    if isinstance(out, dict):
        out = out['outputs']
    if isinstance(out, (tuple, list)):
        return torch.stack(list(out), 0).sum(0)
    return out


def test_zero_gain_identity():
    """Proposition 1: with a_i = 0 the IP-RCF logit equals Plain-PFF with shared weights."""
    ip, plain = _models()
    x = torch.randn(2, 3, 352, 352)
    with torch.no_grad():
        ip.base.progressive_decoder.gain_logits.zero_()
        diff = (_logit(ip, x) - _logit(plain, x)).abs().max().item()
    assert diff == 0.0, diff


def test_logit_linear_in_gain():
    """Proposition 2: z(g) - z(0) is exactly linear in the gains g_i."""
    ip, _ = _models(1)
    x = torch.randn(1, 3, 352, 352)
    dec = ip.base.progressive_decoder
    g = torch.tensor([0.05, -0.03, 0.08])
    with torch.no_grad():
        def z_at(gains):
            dec.gain_logits.copy_(torch.atanh(4.0 * gains))   # g = 0.25 tanh(a)
            return _logit(ip, x).double()
        z0, z1, z2 = z_at(torch.zeros(3)), z_at(g), z_at(2 * g)
    err = ((z2 - z0) - 2 * (z1 - z0)).abs().max().item()
    scale = (z1 - z0).abs().max().item()
    assert scale > 0 and err < 1e-4 * max(1.0, scale), (err, scale)


if __name__ == '__main__':
    test_zero_gain_identity()
    test_logit_linear_in_gain()
    print('all property checks passed')

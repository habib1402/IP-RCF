"""Executable parameter and operation-count audit for IP-RCF variants."""

import argparse

import torch
from torch.utils.flop_counter import FlopCounterMode

from lib.iprcf import IPRCF_VARIANTS, build_iprcf_model


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--variants', nargs='+', choices=IPRCF_VARIANTS,
        default=list(IPRCF_VARIANTS),
    )
    parser.add_argument('--input_size', type=int, default=352, choices=(352,))
    parser.add_argument('--parameter_ceiling', type=int, default=3_550_000)
    parser.add_argument('--mac_ceiling', type=float, default=1.70e9)
    return parser.parse_args()


def main():
    config = parse_arguments()
    sample = torch.randn(1, 3, config.input_size, config.input_size)
    for variant in config.variants:
        model = build_iprcf_model(
            variant,
            channel=64,
            backbone='pvtv2_b0',
            load_backbone_weights=False,
        ).eval()
        parameters = sum(parameter.numel() for parameter in model.parameters())
        # ``FlopCounterMode`` must be entered inside ``no_grad``; entering it
        # inside ``inference_mode`` suppresses Python dispatch in torch 2.1.
        with torch.no_grad():
            with FlopCounterMode(model, display=False) as counter:
                outputs = model(sample)
        flops = int(counter.get_total_flops())
        macs = flops / 2.0
        if parameters > config.parameter_ceiling:
            raise RuntimeError(
                '{} exceeds parameter ceiling: {} > {}'.format(
                    variant, parameters, config.parameter_ceiling
                )
            )
        if macs > config.mac_ceiling:
            raise RuntimeError(
                '{} exceeds MAC ceiling: {:.0f} > {:.0f}'.format(
                    variant, macs, config.mac_ceiling
                )
            )
        print(
            '{}: parameters={} MACs={:.6f}G FLOPs={:.6f}G outputs={}'.format(
                variant, parameters, macs / 1e9, flops / 1e9,
                [tuple(value.shape) for value in outputs],
            ),
            flush=True,
        )


if __name__ == '__main__':
    main()

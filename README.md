# Beyond Internal Validation: Gain-Bounded Contour Injection for Polyp Segmentation

Official code for *"Beyond Internal Validation: Interventional Analysis of Gain-Bounded Contour Injection in Intelligent Polyp Segmentation Systems"*, submitted to *Expert Systems with Applications* (under review).

This repository contains the model, the locked training and evaluation pipeline, the within-model contour-gain intervention, and the scripts that reproduce every figure and table of the paper.

## Overview

**IP-RCF** adds a parallel high-frequency contour stream to a compact PVTv2-B0 + Plain-PFF decoder (3.48 M parameters, 1.686 GMACs at 352x352). The contour correction enters three semantic readouts through scalar gains

```
S~_i = S_i + g_i * C_i,    g_i = 0.25 * tanh(a_i),    a_i = 0 at initialization
```

so the network is **exactly** Plain-PFF at initialization and the semantic recursion never consumes the correction. Both properties are checked in `tests/test_properties.py`.

Key findings (seed 42, 798 development images, raw sigmoid, threshold 0.5):

| | Internal soft Dice | Five-benchmark macro Dice |
|---|---|---|
| Plain-PFF control | 0.9143 | **0.8396** |
| IP-RCF (frozen, epoch 95) | **0.9245** | 0.8315 |
| IP-RCF with gains set to 0 | n/a | 0.8190 |

- The internal gain did **not** transfer, and the locked promotion rule rejected IP-RCF.
- Within the trained model, the fitted correction **helps** (+0.0124 macro Dice), mainly by removing false-positive area.
- The co-trained semantic stream is 0.0206 weaker than an independently trained Plain-PFF.
- Twelve newly missed lesions explain most of the loss on ETIS-LaribPolypDB and CVC-ColonDB.

## Repository structure

```
Train.py                   base training utilities (structure loss, data loading, seeding)
TrainMPECStudent.py        positive/normal training; variant data_control = Plain-PFF control
TrainFRBPFFStudent.py      boundary-supervision control and dual-band decoder
TrainIPRCFStudent.py       IP-RCF variants: finest_residual, multiscale_residual, multiscale_consistency
TestMPEC.py                frozen five-benchmark evaluation of the Plain-PFF control
TestFRBPFF.py              frozen evaluation of the dual-band decoder (shared evaluation code)
TestIPRCF.py               frozen evaluation of the selected IP-RCF checkpoint
AuditIPRCFComplexity.py    parameter and MAC audit
lib/                       PVTv2 backbone wrapper (pvt.py), IP-RCF (iprcf.py), FRB-PFF, MPEC, artifact simulator
utils/                     data loaders and helpers
splits/                    positive validation split and HyperKvasir normal-frame manifest (seed 42)
tools/                     manifest builder and path localizer
analysis/                  gain intervention and figure scripts (Figs. 1-7)
tests/                     property checks (zero-gain identity, logit linearity)
```

## Installation

Tested with Python 3.8, PyTorch 2.1.1 (CUDA 12.1), and an NVIDIA RTX 4090.

```bash
pip install -r requirements.txt
```

The encoder expects ImageNet-1K PVTv2-B0 weights at `pretrained_pth/pvt_v2_b0.in1k.safetensors`. These are the timm weights `timm/pvt_v2_b0.in1k` from the Hugging Face Hub.

## Data

Set `POLYP_DATA_ROOT` (default: `./data`) to a folder with the conventional layout:

```
$POLYP_DATA_ROOT/
  TrainDataset/{images,masks}/                  Kvasir-SEG + CVC-ClinicDB training pool (1,450 images)
  TestDataset/{CVC-300,CVC-ClinicDB,Kvasir,CVC-ColonDB,ETIS-LaribPolypDB}/{images,masks}/
```

Normal (polyp-free) frames come from the labeled HyperKvasir release (cecum, retroflex-rectum, terminal ileum). Download HyperKvasir, then localize the provided manifest:

```bash
python tools/localize_normal_manifest.py --root /path/to/HyperKvasir/extracted
# writes splits/hyperkvasir_normal_manifest_seed42.local.json
```

`splits/positive_split_seed42.json` fixes the 1,305/145 positive training/validation split. The trainers verify its SHA-256.

## Training

All controlled variants share the same recipe:

```bash
COMMON="--normal_manifest splits/hyperkvasir_normal_manifest_seed42.local.json \
  --epoch 100 --lr 1e-4 --backbone_lr_mult 0.1 --batchsize 8 --trainsize 352 --channel 64 \
  --backbone pvtv2_b0 --artifact_strength 0.5 --response_loss_weight 0.5 \
  --training_seed 42 --num_workers 4 --amp"

# Plain-PFF control
python TrainMPECStudent.py --variant data_control --train_save runs/tmi_mpec_data_control_c64_seed42 $COMMON

# IP-RCF (selected variant); also finest_residual / multiscale_consistency
python TrainIPRCFStudent.py --variant multiscale_residual --train_save runs/tmi_iprcf_multiscale_c64_seed42 $COMMON \
  --boundary_loss_weight 0.25 --contour_consistency_weight 0.10 --boundary_radius 2 --boundary_tolerance 2
```

Every trainer supports `--verify_only` for a preflight check. Checkpoint selection uses only the internal positive and normal validation splits. No benchmark or external image is read during training or selection.

## Frozen evaluation

```bash
python TestMPEC.py --variant data_control --pth_path runs/tmi_mpec_data_control_c64_seed42/PolypPVT_best.pth \
  --output_dir results/tmi_mpec_data_control_c64_seed42/predictions_raw_sigmoid
python TestIPRCF.py
```

`TestIPRCF.py` is the exact frozen-evaluation script used in the paper. It verifies the SHA-256 of the released checkpoint and of the control results before evaluating, so it reproduces the reported numbers only with the released checkpoints. To evaluate your own checkpoint, adapt the constants at the top of the file.


The analysis scripts read `IPRCF_ROOT` (default: the repository root, holding `runs/`, `results/`, and `analyses/`), `POLYP_DATA_ROOT`, and, optionally, `TMI_FIGURE_FONT_DIR` for Times New Roman. Without that font they fall back to a serif font. Figure numbers in the scripts follow the preprint; the paper orders them by first appearance.

## Tests

```bash
python -m pytest tests -q
```

## Checkpoints and predictions

Frozen checkpoints and per-image prediction maps will be released upon acceptance.

## Citation

Citation details will be added upon acceptance.

## License

To be added.

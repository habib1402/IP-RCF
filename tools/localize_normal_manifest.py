"""Create a local copy of the HyperKvasir normal-frame manifest with absolute paths.

Usage:
  python tools/localize_normal_manifest.py --root /path/to/HyperKvasir/extracted \
      --out splits/hyperkvasir_normal_manifest_seed42.local.json

The root is the folder that contains labeled-images/ from the official HyperKvasir release
(https://datasets.simula.no/hyper-kvasir/). The image lists and split assignments are unchanged.
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', required=True)
    ap.add_argument('--template', default=str(HERE / 'splits' / 'hyperkvasir_normal_manifest_seed42.template.json'))
    ap.add_argument('--out', default=str(HERE / 'splits' / 'hyperkvasir_normal_manifest_seed42.local.json'))
    args = ap.parse_args()
    root = Path(args.root).expanduser().resolve()
    manifest = json.loads(Path(args.template).read_text())
    manifest['root'] = str(root)
    missing = 0
    for split in ('training', 'validation'):
        for row in manifest[split]:
            row['path'] = str(root / row['relative_path'])
            missing += not Path(row['path']).is_file()
    Path(args.out).write_text(json.dumps(manifest, indent=1))
    print('wrote {} ({} training, {} validation, {} missing files)'.format(
        args.out, len(manifest['training']), len(manifest['validation']), missing))


if __name__ == '__main__':
    main()

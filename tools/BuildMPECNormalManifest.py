"""Audit and split the HyperKvasir lower-GI normal anatomical classes.

FAR-POLYP-SEG and PolypGen are explicitly rejected as inputs.  The selected
HyperKvasir classes are cecum, retroflex rectum and terminal ileum.  Because
the release has no patient identifiers, this is an image-level stratified
split and the limitation is recorded in the output manifest.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image


CLASS_ALIASES = {
    'cecum': ('cecum', 'normal-cecum'),
    'retroflex-rectum': ('retroflex-rectum',),
    'ileum': ('ileum',),
}
EXPECTED_COUNTS = {
    'cecum': 1009,
    'retroflex-rectum': 391,
    'ileum': 9,
}
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
FORBIDDEN_PATH_TOKENS = (
    'far-polyp-seg', 'polygen', 'polypgen', 'external_polyp'
)


def sha256_file(path, block_size=1024 * 1024):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(block_size), b''):
            digest.update(block)
    return digest.hexdigest()


def sha256_records(records):
    digest = hashlib.sha256()
    for record in records:
        digest.update(record['class'].encode('utf-8'))
        digest.update(b'\0')
        digest.update(record['relative_path'].encode('utf-8'))
        digest.update(b'\0')
        digest.update(str(record['size_bytes']).encode('ascii'))
        digest.update(b'\n')
    return digest.hexdigest()


def discover_class_directories(root):
    directories = defaultdict(list)
    alias_to_class = {
        alias: class_name
        for class_name, aliases in CLASS_ALIASES.items()
        for alias in aliases
    }
    for path in root.rglob('*'):
        if path.is_dir() and path.name.lower() in alias_to_class:
            if '__macosx' in {part.lower() for part in path.parts}:
                continue
            directories[alias_to_class[path.name.lower()]].append(path)
    missing = [name for name in CLASS_ALIASES if not directories[name]]
    if missing:
        raise FileNotFoundError(
            'Could not find HyperKvasir class directories: {}'.format(missing)
        )
    return directories


def audit_images(root, directories):
    records = []
    seen_paths = set()
    for class_name in CLASS_ALIASES:
        class_records = []
        for directory in directories[class_name]:
            for path in sorted(directory.rglob('*')):
                if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                resolved = path.resolve()
                if resolved in seen_paths:
                    continue
                seen_paths.add(resolved)
                try:
                    with Image.open(str(resolved)) as image:
                        image.load()
                        width, height = image.size
                        mode = image.mode
                except Exception as error:
                    raise RuntimeError(
                        'Unreadable HyperKvasir image {}: {!r}'.format(
                            resolved, error
                        )
                    )
                if width < 64 or height < 64:
                    raise RuntimeError(
                        'Unexpectedly small image {}: {}x{}'.format(
                            resolved, width, height
                        )
                    )
                class_records.append({
                    'class': class_name,
                    'path': str(resolved),
                    'relative_path': str(resolved.relative_to(root)),
                    'width': width,
                    'height': height,
                    'mode': mode,
                    'size_bytes': resolved.stat().st_size,
                })
        if len(class_records) != EXPECTED_COUNTS[class_name]:
            raise RuntimeError(
                '{} count differs from official audited release: '
                'expected {}, found {}'.format(
                    class_name, EXPECTED_COUNTS[class_name], len(class_records)
                )
            )
        records.extend(sorted(class_records, key=lambda row: row['relative_path']))
    return records


def stratified_split(records, validation_ratio, seed):
    grouped = defaultdict(list)
    for record in records:
        grouped[record['class']].append(record)
    rng = np.random.default_rng(seed)
    training = []
    validation = []
    split_counts = {}
    for class_name in CLASS_ALIASES:
        group = grouped[class_name]
        indices = np.arange(len(group), dtype=np.int64)
        rng.shuffle(indices)
        validation_count = max(1, int(round(len(group) * validation_ratio)))
        validation_indices = set(indices[:validation_count].tolist())
        for index, record in enumerate(group):
            (validation if index in validation_indices else training).append(record)
        split_counts[class_name] = {
            'total': len(group),
            'training': len(group) - validation_count,
            'validation': validation_count,
        }
    training.sort(key=lambda row: (row['class'], row['relative_path']))
    validation.sort(key=lambda row: (row['class'], row['relative_path']))
    return training, validation, split_counts


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Build locked HyperKvasir normal-frame manifest for MPEC'
    )
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--validation_ratio', type=float, default=0.20)
    parser.add_argument('--split_seed', type=int, default=42)
    return parser.parse_args()


def main():
    config = parse_arguments()
    if config.validation_ratio != 0.20:
        raise ValueError('MPEC normal validation ratio is locked to 0.20')
    if config.split_seed != 42:
        raise ValueError('MPEC normal split seed is locked to 42')
    root = Path(config.root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError('HyperKvasir root not found: {}'.format(root))
    lower_root = str(root).lower()
    if any(token in lower_root for token in FORBIDDEN_PATH_TOKENS):
        raise RuntimeError(
            'Refusing forbidden external-test source as MPEC normal data: {}'
            .format(root)
        )
    output = Path(config.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError('Refusing to overwrite manifest: {}'.format(output))

    directories = discover_class_directories(root)
    records = audit_images(root, directories)
    training, validation, split_counts = stratified_split(
        records, config.validation_ratio, config.split_seed
    )
    manifest = {
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'dataset': 'HyperKvasir labeled images',
        'dataset_url': 'https://datasets.simula.no/hyper-kvasir/',
        'dataset_doi': '10.1038/s41597-020-00622-y',
        'license': 'CC BY 4.0; citation required',
        'root': str(root),
        'selected_lower_gi_classes': list(CLASS_ALIASES),
        'class_aliases': CLASS_ALIASES,
        'official_expected_counts': EXPECTED_COUNTS,
        'observed_total': len(records),
        'validation_ratio': config.validation_ratio,
        'split_seed': config.split_seed,
        'split_unit': 'image',
        'patient_identifiers_available': False,
        'limitation': (
            'HyperKvasir labeled images do not provide patient identifiers; '
            'patient-disjoint splitting is impossible.'
        ),
        'external_test_sources_used': [],
        'forbidden_sources': ['FAR-POLYP-SEG', 'PolypGen'],
        'split_counts': split_counts,
        'record_list_sha256': sha256_records(records),
        'training': training,
        'validation': validation,
        'builder_sha256': sha256_file(Path(__file__).resolve()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w') as handle:
        json.dump(manifest, handle, indent=2)
    print('HyperKvasir MPEC normal-frame audit passed')
    print('Classes: {}'.format(split_counts))
    print('Training: {}; validation: {}'.format(
        len(training), len(validation)
    ))
    print('Manifest: {}'.format(output))


if __name__ == '__main__':
    main()

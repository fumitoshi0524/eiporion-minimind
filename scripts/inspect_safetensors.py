"""Inspect tensor storage in a safetensors checkpoint without loading its tensors.

Usage: python scripts/inspect_safetensors.py out/model.safetensors
Requires only the Python standard library.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct


def inspect(path):
    path = Path(path)
    with path.open('rb') as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError('Missing safetensors header length')
        header_length = struct.unpack('<Q', prefix)[0]
        if header_length > path.stat().st_size - 8:
            raise ValueError('Header extends beyond the checkpoint')
        header = json.loads(stream.read(header_length))
    counts, sizes, groups = Counter(), Counter(), Counter()
    for name, tensor in header.items():
        if name == '__metadata__':
            continue
        count = 1
        for dimension in tensor['shape']:
            count *= dimension
        size = tensor['data_offsets'][1] - tensor['data_offsets'][0]
        counts[tensor['dtype']] += count
        sizes[tensor['dtype']] += size
        if name.endswith('.int_weight'):
            group = 'int8_projection_weights'
        elif name.endswith('.weight_scale'):
            group = 'projection_scales'
        elif name.endswith('._bit_handle'):
            group = 'runtime_handles'
        else:
            group = 'other_weights'
        groups[group] += size
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {
        'checkpoint': path.name,
        'sha256': digest,
        'file_bytes': path.stat().st_size,
        'header_and_prefix_bytes': header_length + 8,
        'tensor_payload_bytes': sum(sizes.values()),
        'scalar_counts_by_dtype': dict(counts),
        'tensor_bytes_by_dtype': dict(sizes),
        'tensor_bytes_by_group': dict(groups),
        'weight_and_scale_bytes': sum(value for key, value in groups.items() if key != 'runtime_handles'),
        'unit': 'bytes; MiB = bytes / 1048576',
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    print(json.dumps(inspect(parser.parse_args().checkpoint), indent=2))

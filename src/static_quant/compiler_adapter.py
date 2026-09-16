"""Opt-in adapter for the existing Python compiler's quant_param_addr keyword.

This does not establish an RTL ABI or infer a physical address. Callers must supply
an allocated, 64B-aligned base and the EXACT weight name from torch.export's
 graph_signature.inputs_to_parameters (do not reverse-engineer underscore names).
"""
import json
from pathlib import Path


def quant_param_kwargs(directory, weight_name, runtime_base_address, *,
                       output_channel_start=0, allow_unverified=False):
    manifest = json.loads((Path(directory)/'manifest.json').read_text())
    if not manifest['binary_export']['present']:
        raise ValueError('No qparams.bin was exported')
    if manifest['hardware_profile']['verification'] != 'verified' and not allow_unverified:
        raise ValueError('Profile is unverified; experimental adapter use must be explicit')
    entries = [e for e in manifest['modules'] if e['weight_name'] == weight_name]
    if len(entries) != 1:
        raise KeyError(f'No unique calibrated weight: {weight_name}')
    entry = entries[0]
    for value, label in [(runtime_base_address, 'runtime base'), (output_channel_start, 'channel start')]:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f'{label} must be a nonnegative integer')
    if runtime_base_address % 64:
        raise ValueError('runtime base must be 64B aligned')
    if output_channel_start % 16 or output_channel_start >= entry['valid_channels']:
        raise ValueError('channel start must select a valid 16-channel block')
    offset = entry['parameter_file_offset']
    size = entry['parameter_size_bytes']
    if offset % 64 or offset < 0 or size % 64 or size < entry['valid_channels']*4:
        raise ValueError('invalid parameter mapping alignment/size')
    binary_path = Path(directory)/'qparams.bin'
    if binary_path.stat().st_size < offset+size:
        raise ValueError('qparams.bin is truncated')
    address = runtime_base_address+offset+4*output_channel_start
    if runtime_base_address+offset+size > 2**64:
        raise ValueError('parameter allocation exceeds UInt64 address range')
    return {'quant_param_addr': address}

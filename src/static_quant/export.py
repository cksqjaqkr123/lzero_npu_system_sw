"""Separate artifacts and explicit file-offset mapping; never assign physical addresses."""
import json
from pathlib import Path
import numpy as np
from .hardware import get_profile
from .gemma import sha256_file


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


class Exporter:
    def __init__(self, directory, profile, metadata, allow_unverified=False, save_weights=False,
                 allow_inaccurate=False):
        self.path = Path(directory)
        # Require a new or empty directory so no previous output can be overwritten.
        if self.path.exists() and any(self.path.iterdir()):
            raise FileExistsError(f'Output directory must be empty: {self.path}')
        self.path.mkdir(parents=True, exist_ok=True)
        self.profile, self.allow_binary = profile, allow_unverified
        self.save_weights, self.allow_inaccurate = save_weights, allow_inaccurate
        self.scales, self.params, self.reports, self.vectors = {}, {}, {}, {}
        self.manifest = dict(schema_version=1, **metadata, hardware_profile=profile.metadata(),
                             axes={'X': '[M,K]', 'W': '[N,K]', 'Y': '[M,N]',
                                   'channel_axis_W': 0, 'channel_axis_Y': 1},
                             calibration_scope='prefill-only, original model inputs, local Linear error',
                             excluded=['attention QK^T / PV activation matmuls', 'decode calibration',
                                       'propagated full-model quantization', 'perplexity', 'RTL equivalence'],
                             parameter_storage=dict(bytes_per_channel=4, qb_channels=16, alignment_bytes=64,
                                                    endian='little', padding_word=0, order='output channel ascending',
                                                    address_kind='file offset, NOT runtime physical address',
                                                    repeated_over_M=False, repeated_over_K=False),
                             modules=[], binary_export=dict(present=False, status='pending'))
        self.words = []
        self.offset = 0

    def add(self, operation, report, vectors, input_stats=None, input_group=None):
        op = operation
        if any(entry['module_name'] == op.name for entry in self.manifest['modules']):
            raise ValueError(f'Duplicate module: {op.name}')
        key = f'op{len(self.manifest["modules"]):04d}'
        p = op.params
        count = ((op.n+15)//16)*16
        words = np.zeros(count, dtype='<u4')
        words[:op.n] = self.profile.pack(p['multiplier'], p['shift'])
        self.words.append(words)
        self.scales.update({key+'.s_X': np.asarray(op.sx), key+'.s_10': np.asarray(op.s10), key+'.s_W': op.sw})
        for field, value in p.items():
            self.params[key+'.'+field] = value
        self.params[key+'.packed'] = words[:op.n]
        self.params[key+'.zero_point'] = np.zeros(op.n, dtype=np.int8)
        lut_scale = op.options.fixed_lut_scale
        entry = dict(key=key, module_name=op.name, weight_name=op.name+'.weight',
                     torch_export_parameter_symbol='p_'+(op.name+'.weight').replace('.', '_'),
                     shape={'M': op.tokens, 'K': op.k, 'N': op.n},
                     M_semantics='sum of valid calibration tokens, not a parameter axis',
                     scale_keys={name: key+'.'+name for name in ('s_X', 's_10', 's_W')},
                     s_X=op.sx, s_10=op.s10, input_group=input_group,
                     parameter_file_offset=self.offset, parameter_size_bytes=count*4,
                     valid_parameter_bytes=op.n*4, valid_channels=op.n,
                     padded_channels=count-op.n, final_block_valid_channels=(op.n-1)%16+1,
                     alignment_bytes=64, padding_word=0,
                     lut=dict(input_scale=op.s10, signed_index_encoding='10-bit two\'s complement',
                              reused_lut_input_scale=lut_scale,
                              existing_gelu_index_scale_compatible=op.s10 == 0.1,
                              requires_scale_matched_lut=lut_scale is None,
                              end_to_end_signed_int8_compatible=False,
                              limitation='src/LUT.py clips GeLU output to unsigned [0,255]; no signed INT8 contract'))
        if self.save_weights:
            filename = key+'.weights.int8.npy'
            weights = np.lib.format.open_memmap(self.path/filename, mode='w+', dtype=np.int8, shape=(op.n,op.k))
            for sl in op.channels():
                w = op.weight(sl)
                weights[sl] = np.clip(np.rint(w/op.sw[sl, None]), -127, 127).astype(np.int8)
            weights.flush()
            del weights
            entry['int8_weight_file'] = filename
        for split in ('calibration', 'validation'):
            if split in report:
                report[split]['samples'] = self.manifest.get('datasets', {}).get(split, {}).get('actual_samples')
        if input_stats is not None:
            report['input_statistics'] = input_stats
        self.manifest['modules'].append(entry)
        self.reports[op.name], self.vectors[op.name] = report, vectors
        self.offset += count*4

    def finish(self):
        inaccurate = any(np.any(v != 'ok') for k,v in self.params.items() if k.endswith('.status'))
        binary = self.allow_binary and (not inaccurate or self.allow_inaccurate)
        reason = 'explicit unverified export' if binary else (
            'blocked: inaccurate/unrepresentable ratios; inspect report or explicitly allow inaccurate parameters'
            if self.allow_binary and inaccurate else 'blocked: no RTL-verified profile; use --allow-unverified-export for experiments')
        if binary:
            with (self.path/'qparams.bin').open('wb') as handle:
                for words in self.words:
                    handle.write(words.astype('<u4').tobytes())
        self.manifest['binary_export'] = dict(present=binary, status=reason,
                                              explicit_allow_unverified=self.allow_binary,
                                              explicit_allow_inaccurate=self.allow_inaccurate,
                                              size_bytes=self.offset if binary else 0)
        np.savez(self.path/'scales.npz', **self.scales)
        np.savez(self.path/'qparams.npz', **self.params)
        write_json(self.path/'report.json', dict(scope=self.manifest['calibration_scope'],
                                                rtl_bit_exact=False, modules=self.reports))
        write_json(self.path/'test_vectors.json', dict(hardware_profile=self.profile.metadata(),
                                                      status='software expected values for future RTL comparison',
                                                      modules=self.vectors))
        mapping = {entry['weight_name']: dict(module_name=entry['module_name'],
                    torch_export_parameter_symbol=entry['torch_export_parameter_symbol'],
                    parameter_file_offset=entry['parameter_file_offset'],
                    size_bytes=entry['parameter_size_bytes'], valid_channels=entry['valid_channels'])
                   for entry in self.manifest['modules']}
        write_json(self.path/'compiler_mapping.json', dict(address_kind='file offset',
                   runtime_base_address=None, compiler_field='quant_param_addr', weights=mapping))
        self.manifest['artifact_sha256'] = {
            p.name: sha256_file(p)
            for p in self.path.iterdir() if p.is_file() and p.name != 'manifest.json'
        }
        write_json(self.path/'manifest.json', self.manifest)
        return self.manifest


def reload_parameters(directory, module_name, from_binary=False):
    path = Path(directory)
    manifest = json.loads((path/'manifest.json').read_text())
    entry = next(e for e in manifest['modules'] if e['module_name'] == module_name)
    profile = get_profile(manifest['hardware_profile']['name'])
    if from_binary:
        if not manifest['binary_export']['present']:
            raise ValueError('No hardware binary was exported')
        with (path/'qparams.bin').open('rb') as handle:
            handle.seek(entry['parameter_file_offset'])
            payload = handle.read(entry['parameter_size_bytes'])
        if len(payload) != entry['parameter_size_bytes']:
            raise ValueError('Truncated parameter block')
        padded = np.frombuffer(payload, dtype='<u4')
        if np.any(padded[entry['valid_channels']:] != 0):
            raise ValueError('Nonzero padding')
        words = padded[:entry['valid_channels']].copy()
    else:
        with np.load(path/'qparams.npz', allow_pickle=False) as data:
            words = data[entry['key']+'.packed'].copy()
    return profile, profile.unpack(words), entry

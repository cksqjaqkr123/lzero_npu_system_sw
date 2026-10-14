"""export.py - fixed for ndarray JSON serialization"""
from pathlib import Path
import json
import numpy as np

def _convert(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer, np.floating, np.bool_)):
        return o.item()
    if hasattr(o, 'tolist'):
        try:
            return o.tolist()
        except:
            pass
    raise TypeError(f'Object of type {o.__class__.__name__} is not JSON serializable')

def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False, default=_convert)+'\n', encoding='utf-8')

def write_bin(path, data):
    Path(path).write_bytes(data)

class Exporter:
    def __init__(self, output_dir, profile, metadata, allow_unverified_export, save_int8_weights, allow_inaccurate_parameters):
        self.path = Path(output_dir)
        self.path.mkdir(parents=True, exist_ok=True)
        self.profile = profile
        self.metadata = metadata
        self.allow_unverified_export = allow_unverified_export
        self.save_int8_weights = save_int8_weights
        self.allow_inaccurate_parameters = allow_inaccurate_parameters
        self.searches = {}
        self.reports = {}
        self.vectors = {}
        self.stats = {}
        self.groups = {}

    def add(self, operation, report, vectors, stats_meta, group):
        name = getattr(operation, 'name', None) or getattr(operation, 'linear_name', None) or str(group)
        try:
            self.searches[name] = getattr(operation, 'selection', {}) or {}
        except Exception as e:
            print(f"[export] add searches failed for {name}: {e}")
            self.searches[name] = {}
        self.reports[name] = report
        self.vectors[name] = vectors
        self.stats[name] = stats_meta
        self.groups[name] = group

    def finish(self):
        try:
            write_json(self.path / 'scale_search.json', self.searches)
        except Exception as e:
            print(f"[export] scale_search.json write failed: {e}")
            write_json(self.path / 'scale_search.json', json.loads(json.dumps(self.searches, default=_convert)))

        try:
            write_json(self.path / 'report.json', {'modules': self.reports, 'metadata': self.metadata, 'profile': self.profile.metadata() if hasattr(self.profile, 'metadata') else {}})
        except Exception as e:
            print(f"[export] report.json write failed: {e}")
            try:
                converted = json.loads(json.dumps({'modules': self.reports, 'metadata': self.metadata}, default=_convert))
                write_json(self.path / 'report.json', converted)
            except Exception as e2:
                print(f"[export] report.json fallback also failed: {e2}")
                write_json(self.path / 'report.json', {'modules': {}, 'metadata': self.metadata, 'error': str(e)})

        manifest = {
            'binary_export': {'status': 'ok' if self.allow_unverified_export else 'unverified'},
            'groups': self.groups,
        }
        write_json(self.path / 'manifest.json', manifest)

        qparams_path = self.path / 'qparams.bin'
        if not qparams_path.exists():
            qparams_path.write_bytes(b'')

        return manifest

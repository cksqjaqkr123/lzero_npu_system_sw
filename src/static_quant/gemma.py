"""Optional Hugging Face frontend. No torch/transformers dependency for synthetic tests."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np

PROJECTIONS = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_texts(path, limit):
    if limit < 1:
        raise ValueError('sample count must be positive')
    path = Path(path)
    texts = []
    with path.open(encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            if path.suffix.lower() == '.jsonl':
                try:
                    obj = json.loads(line)
                    text = obj['text']
                    if not isinstance(text, str):
                        raise TypeError('text must be a string')
                except (ValueError, KeyError, TypeError) as exc:
                    raise ValueError(f'{path.name}:{line_number}: expected JSON object with string text') from exc
            else:
                text = line.rstrip('\r\n')
            if text.strip():
                texts.append(text)
            if len(texts) == limit:
                break
    if not texts:
        raise ValueError(f'No nonempty samples in {path}')
    fingerprint = hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest()
    return texts, dict(file_sha256=sha256_file(path), selected_texts_sha256=fingerprint,
                      requested_samples=limit, actual_samples=len(texts), selection='first nonempty records in file order',
                      format='JSONL text field' if path.suffix.lower() == '.jsonl' else 'one nonempty line per sample')


def input_group(name):
    stem, _, leaf = name.rpartition('.')
    if leaf in ('q_proj', 'k_proj', 'v_proj'):
        return stem+'.qkv_input'
    if leaf in ('gate_proj', 'up_proj'):
        return stem+'.gate_up_input'
    return name+'.input'


def select_names(layer_count, layers='0', modules='q_proj', lm_head=False):
    ids = list(range(layer_count)) if layers == 'all' else [int(s) for s in layers.split(',')]
    projections = list(PROJECTIONS) if modules == 'all' else modules.split(',')
    if not ids or any(i < 0 or i >= layer_count for i in ids) or len(set(ids)) != len(ids):
        raise ValueError(f'layers must be unique indices in [0,{layer_count-1}] or all')
    if not projections or any(p not in PROJECTIONS for p in projections) or len(set(projections)) != len(projections):
        raise ValueError(f'modules must be unique names from {PROJECTIONS} or all')
    names = [f'model.layers.{i}.{"self_attn" if p in PROJECTIONS[:4] else "mlp"}.{p}'
             for i in sorted(ids) for p in projections]
    if lm_head:
        names.append('lm_head')
    return names


def validate_config(config):
    expected = dict(model_type='gemma', num_hidden_layers=18, hidden_size=2048,
                    intermediate_size=16384, num_attention_heads=8, num_key_value_heads=1, head_dim=256)
    actual = {key: getattr(config, key, None) for key in expected}
    if actual != expected:
        raise ValueError(f'Expected original Gemma 2B architecture (not Gemma 2): {actual}; expected {expected}')
    if getattr(config, 'quantization_config', None) is not None:
        raise ValueError('Load an original floating-point checkpoint, not prequantized weights')
    return actual


class GemmaSource:
    def __init__(self, args):
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        try:
            import torch
            import transformers
        except ImportError as exc:
            raise RuntimeError('Real Gemma requires requirements-calibration.txt; synthetic mode only needs NumPy') from exc
        self.torch = torch
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        self.device = torch.device(args.device)
        self.batch_size = args.batch_size
        kwargs = dict(revision=args.revision, local_files_only=args.local_files_only, trust_remote_code=False)
        config = transformers.AutoConfig.from_pretrained(args.model, **kwargs)
        architecture = validate_config(config)
        if not Path(args.model).is_dir() and getattr(config, '_commit_hash', None):
            kwargs['revision'] = config._commit_hash  # freeze model and tokenizer to config snapshot
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, **kwargs)
        self.tokenizer.padding_side = 'right'
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.pad_token_id is None:
            raise ValueError('tokenizer has neither pad nor EOS token')
        dtype = getattr(torch, args.dtype)
        # from_pretrained only: reject missing weights rather than accepting random initialization.
        self.model, loading = transformers.AutoModelForCausalLM.from_pretrained(
            args.model, config=config, torch_dtype=dtype, attn_implementation='eager',
            output_loading_info=True, **kwargs)
        if any(loading.get(key) for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
            raise ValueError('Checkpoint loading is not exact; missing/unexpected/mismatched weights or loading errors')
        if any(p.is_meta for p in self.model.parameters()):
            raise ValueError('Meta weights are forbidden for calibration')
        self.model.to(self.device).eval()
        self.model.config.use_cache = False
        self.modules = dict(self.model.named_modules())
        self.names = select_names(config.num_hidden_layers, args.layers, args.modules, args.include_lm_head)
        expected_shapes = {
            'q_proj': (config.num_attention_heads*config.head_dim, config.hidden_size),
            'k_proj': (config.num_key_value_heads*config.head_dim, config.hidden_size),
            'v_proj': (config.num_key_value_heads*config.head_dim, config.hidden_size),
            'o_proj': (config.hidden_size, config.num_attention_heads*config.head_dim),
            'gate_proj': (config.intermediate_size, config.hidden_size),
            'up_proj': (config.intermediate_size, config.hidden_size),
            'down_proj': (config.hidden_size, config.intermediate_size),
            'lm_head': (config.vocab_size, config.hidden_size)}
        for name in self.names:
            module = self.modules[name]
            if not isinstance(module, torch.nn.Linear) or module.bias is not None:
                raise ValueError(f'{name}: only bias-free Linear is supported')
            if tuple(module.weight.shape) != expected_shapes[name.split('.')[-1]]:
                raise ValueError(f'{name}: weight shape {module.weight.shape} conflicts with config')
        self.datasets = {}
        dataset_metadata = {}
        for split, path in [('calibration', args.calibration_data), ('validation', args.validation_data)]:
            if path is None:
                continue
            limit = args.samples if split == 'calibration' else args.validation_samples
            texts, info = load_texts(path, limit)
            encoded = []
            for text in texts:
                if args.text_format == 'chat':
                    ids = self.tokenizer.apply_chat_template([{'role': 'user', 'content': text}],
                          tokenize=True, add_generation_prompt=False, truncation=True, max_length=args.sequence_length)
                else:
                    ids = self.tokenizer(text, add_special_tokens=True, truncation=True,
                                         max_length=args.sequence_length)['input_ids']
                if not ids:
                    raise ValueError('Tokenizer produced an empty sample')
                encoded.append(ids)
            self.datasets[split] = encoded
            info.update(valid_tokens=sum(map(len, encoded)),
                        token_ids_sha256=hashlib.sha256(json.dumps(encoded).encode()).hexdigest())
            dataset_metadata[split] = info
        local_files = None
        if Path(args.model).is_dir():
            files = sorted(p for p in Path(args.model).rglob('*') if p.is_file() and (
                p.suffix in ('.safetensors', '.bin') or p.name in ('config.json', 'model.safetensors.index.json', 'pytorch_model.bin.index.json')))
            local_files = {str(p.relative_to(args.model)): dict(size=p.stat().st_size, sha256=sha256_file(p)) for p in files}
        self.metadata = dict(model=args.model, model_revision_requested=args.revision,
                             model_revision_resolved=getattr(config, '_commit_hash', None),
                             local_checkpoint_files=local_files, model_config=architecture,
                             library_versions=dict(torch=torch.__version__, transformers=transformers.__version__, numpy=np.__version__),
                             device=str(self.device), dtype=args.dtype, datasets=dataset_metadata,
                             tokenizer=dict(class_name=type(self.tokenizer).__name__, format=args.text_format,
                                 revision=self.tokenizer.init_kwargs.get('_commit_hash'),
                                 chat_template=self.tokenizer.chat_template if args.text_format == 'chat' else None,
                                 add_generation_prompt=False, add_special_tokens=args.text_format == 'plain',
                                 add_bos_token=getattr(self.tokenizer, 'add_bos_token', None),
                                 add_eos_token=getattr(self.tokenizer, 'add_eos_token', None),
                                 bos_token_id=self.tokenizer.bos_token_id, eos_token_id=self.tokenizer.eos_token_id,
                                 pad_token_id=self.tokenizer.pad_token_id, padding_side='right',
                                 truncation_side=self.tokenizer.truncation_side,
                                 sequence_length=args.sequence_length),
                             deterministic_algorithms=True, attention_backend='eager',
                             execution='eval + inference_mode; original inputs; stop at selected Linear pre-hook')

    def read_weight(self, name):
        def read(sl):
            return self.modules[name].weight[sl].detach().to(device='cpu', dtype=self.torch.float32).numpy()
        return read

    def replay(self, name, split='calibration'):
        from .core import valid_rows
        torch = self.torch
        def run(callback):
            # Stopping at the selected pre-hook avoids unused layers and vocabulary logits.
            # No output substitution, quantized model mutation or propagated-error claim.
            class Observed(Exception):
                pass
            mask = None
            fired = False
            def hook(module, inputs):
                nonlocal fired
                x = inputs[0].detach().to(device='cpu', dtype=torch.float32).numpy()
                try:
                    callback(valid_rows(x, mask))
                except (ValueError, OverflowError) as exc:
                    raise type(exc)(f'{name}: {exc}') from exc
                fired = True
                raise Observed()
            handle = self.modules[name].register_forward_pre_hook(hook)
            try:
                with torch.inference_mode():
                    rows = self.datasets[split]
                    for start in range(0, len(rows), self.batch_size):
                        batch = self.tokenizer.pad({'input_ids': rows[start:start+self.batch_size]},
                                                    padding=True, return_tensors='pt', return_attention_mask=True)
                        mask = batch['attention_mask'].numpy()
                        fired = False
                        try:
                            self.model(**{k: v.to(self.device) for k,v in batch.items()}, use_cache=False)
                        except Observed:
                            pass
                        if not fired:
                            raise RuntimeError(f'{name}: Linear input hook did not run')
            finally:
                handle.remove()
        return run

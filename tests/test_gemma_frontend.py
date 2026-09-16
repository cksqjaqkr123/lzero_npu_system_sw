"""Optional dependency integration checks, tiny synthetic Gemma only, NO downloads.

This is not a real checkpoint calibration or perplexity/RTL validation.
"""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from static_quant.gemma import GemmaSource
from static_quant.core import InputStats, LinearCalibration, Options
from static_quant.hardware import Profile
try:
    import torch
    from transformers import GemmaConfig, GemmaForCausalLM
except ImportError:
    torch = None


@unittest.skipIf(torch is None,'optional torch/transformers are not installed')
class GemmaHookIntegrationTests(unittest.TestCase):
    def setUp(self):
        config = GemmaConfig(vocab_size=32,hidden_size=16,intermediate_size=32,
                             num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,
                             head_dim=8,max_position_embeddings=32,pad_token_id=0)
        config._attn_implementation = 'eager'
        model = GemmaForCausalLM(config).eval()
        # Explicit deterministic fixture weights, not a pretrained checkpoint.
        with torch.no_grad():
            for p in model.parameters():
                p.copy_(torch.sin(torch.arange(p.numel()).reshape(p.shape).float())*.03)
        source = GemmaSource.__new__(GemmaSource)
        source.torch, source.model, source.device = torch, model, torch.device('cpu')
        source.modules = dict(model.named_modules())
        source.batch_size = 2
        source.datasets = {'calibration': [[2,4,6,8],[2,9]], 'validation': [[2,3,5]]}
        class PadTokenizer:
            def pad(self, rows, **kwargs):
                ids = rows['input_ids']
                length = max(map(len,ids))
                return {'input_ids': torch.tensor([row+[0]*(length-len(row)) for row in ids]),
                        'attention_mask': torch.tensor([[1]*len(row)+[0]*(length-len(row)) for row in ids])}
        source.tokenizer = PadTokenizer()
        self.source = source

    def test_real_hook_inputs_shared_and_padding_excluded(self):
        source = self.source
        observed = {}
        for projection in ('q_proj','k_proj','v_proj'):
            name = 'model.layers.0.self_attn.'+projection
            source.replay(name)(lambda x: observed.update({projection:x.copy()}))
            self.assertFalse(source.modules[name]._forward_pre_hooks)
        self.assertEqual(observed['q_proj'].shape,(6,16))
        np.testing.assert_array_equal(observed['q_proj'],observed['k_proj'])
        np.testing.assert_array_equal(observed['q_proj'],observed['v_proj'])

    def test_actual_frontend_hook_calibration_heldout_and_lm_head(self):
        source = self.source
        for name in ('model.layers.0.self_attn.q_proj','model.layers.0.mlp.down_proj','lm_head'):
            stats = InputStats()
            source.replay(name)(stats.add)
            op = LinearCalibration(name,source.modules[name].weight.shape,source.read_weight(name),stats.scale(),
                                   Profile(),Options(m_chunk=2,n_chunk=7,k_chunk=5,search_rows=4))
            report,_ = op.run(source.replay(name),source.replay(name,'validation'))
            self.assertEqual(report['calibration']['valid_tokens'],6)
            self.assertEqual(report['validation']['valid_tokens'],3)
            self.assertFalse(source.modules[name]._forward_pre_hooks)
            self.assertIsNone(source.modules[name].weight.grad)

    def test_hook_cleanup_on_callback_failure(self):
        name = 'model.layers.0.self_attn.q_proj'
        def failure(x):
            raise ValueError('test failure')
        with self.assertRaisesRegex(ValueError,'test failure'):
            self.source.replay(name)(failure)
        self.assertFalse(self.source.modules[name]._forward_pre_hooks)

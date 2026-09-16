import ast
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import types
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from static_quant.hardware import Profile, get_profile, ideal, lut_index, check_lut_scale
from static_quant.core import (InputStats, Reservoir, Options, LinearCalibration, quantize_weight,
                               quantize_input, exact_accumulator, valid_rows)
from static_quant.export import Exporter, reload_parameters
from static_quant.compiler_adapter import quant_param_kwargs
from static_quant.gemma import input_group, load_texts, select_names, validate_config
from static_quant.cli import main


def fixture(seed=13, n=19, mode='mse'):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(17,23))
    w = rng.normal(size=(n,23))*np.linspace(.001,5,n)[:,None]
    w[0] = 0
    stats = InputStats(seed)
    stats.add(x)
    replay = lambda callback: [callback(x[:6]), callback(x[6:])]
    options = Options(m_chunk=3, n_chunk=7, k_chunk=5, seed=seed, search_rows=9, mode=mode)
    op = LinearCalibration('model.layers.0.self_attn.q_proj', w.shape, lambda sl: w[sl], stats.scale(), Profile(), options)
    report, vectors = op.run(replay)
    return op, report, vectors, x, w


class IntegerReferenceTests(unittest.TestCase):
    def test_weight_axis_and_row_reuse(self):
        w = np.array([[1.,-2.,0.], [40.,0.,-100.], [0.,0.,0.]])
        wq, sw = quantize_weight(w)
        np.testing.assert_allclose(sw, [2/127,100/127,1])
        self.assertTrue(np.all(wq[2] == 0))
        xq = np.array([[1,2,3],[-4,2,1],[1,2,3]], dtype=np.int8)
        acc = exact_accumulator(xq, wq, 1)
        params = Profile().approximate(.2*sw/.1)
        q = Profile().apply(acc, params['multiplier'], params['shift'])
        self.assertEqual(params['multiplier'].shape, (3,))
        np.testing.assert_array_equal(q[0], q[2])
        np.testing.assert_array_equal(q[:,2], 0)

    def test_k_chunk_and_python_integer_oracle(self):
        rng = np.random.default_rng(8)
        x = rng.integers(-127,128,(9,51),dtype=np.int8)
        w = rng.integers(-127,128,(7,51),dtype=np.int8)
        oracle = np.array([[sum(int(a)*int(b) for a,b in zip(row,channel)) for channel in w] for row in x])
        for chunk in (1,7,51,100):
            np.testing.assert_array_equal(exact_accumulator(x,w,chunk), oracle)

    def test_overflow(self):
        x = np.full((1,140000),127,dtype=np.int8)
        with self.assertRaisesRegex(OverflowError, 'INT32'):
            exact_accumulator(x,x,2048)

    def test_int32_extremes_product_fits(self):
        a = np.array([-(2**31),2**31-1],dtype=np.int64)
        q = Profile().apply(a,65535,0)
        np.testing.assert_array_equal(q,[-512,511])

    def test_rounding_negative_and_saturation(self):
        p = Profile()
        a = np.array([-5,-3,-1,1,3,5],dtype=np.int64)
        np.testing.assert_array_equal(p.apply(a,1,1),[-2,-2,0,0,2,2])
        np.testing.assert_array_equal(p.apply(np.array([-513,-512,-511,510,511,512]),1,0),[-512,-512,-511,510,511,511])
        np.testing.assert_array_equal(p.apply(np.array([-1,1]),1,1,zp=1),[1,1])
        legacy = get_profile('legacy-guess')
        np.testing.assert_array_equal(legacy.apply(a,1,3),[-3,-2,-1,0,1,2])
        np.testing.assert_array_equal(legacy.apply(a,1,0),a)
        with self.assertRaises(ValueError):
            p.apply(a,1,-1)
        np.testing.assert_array_equal(lut_index(np.array([-512,-1,0,511])),[512,1023,0,511])

    def test_rounding_python_oracle(self):
        rng = np.random.default_rng(4)
        a = rng.integers(-(2**31),2**31,(17,9),dtype=np.int64)
        m = rng.integers(0,65536,9,dtype=np.int64)
        s = rng.integers(0,32,9,dtype=np.int64)
        # Python round handles these exactly representable binary fractions.
        expected = [[max(-512,min(511,round(int(v)*int(mm)/(2**int(ss))))) for v,mm,ss in zip(row,m,s)] for row in a]
        np.testing.assert_array_equal(Profile().apply(a,m,s),expected)

    def test_ratio_range_status_and_rejection(self):
        p = Profile()
        r = p.approximate([0,.5,1.25,1e-30,70000])
        self.assertEqual(r['status'].tolist(),['ok','ok','ok','underflow_to_zero','above_maximum'])
        self.assertEqual(r['realized_ratio'][1],.5)
        for values in ([-1],[np.nan],[np.inf]):
            with self.assertRaises(ValueError):
                p.approximate(values)
        for m,s,z in ((65536,0,0),(-1,0,0),(1,32,0),(1,-1,0),(1,0,128),(1.,0,0)):
            with self.assertRaises(ValueError):
                p.pack(m,s,z)

    def test_pack_unpack_and_endian(self):
        p = Profile()
        words = p.pack(np.array([0xABCD,65535,0]),np.array([17,31,0]),np.array([-2,127,-128]))
        self.assertEqual(words.astype('<u4').tobytes()[:4],bytes([0xFE,0x11,0xCD,0xAB]))
        m,s,z = p.unpack(words)
        np.testing.assert_array_equal(m,[0xABCD,65535,0])
        np.testing.assert_array_equal(s,[17,31,0])
        np.testing.assert_array_equal(z,[-2,127,-128])
        with self.assertRaisesRegex(ValueError,'reserved'):
            p.unpack(np.array([0x2000]))


class CalibrationTests(unittest.TestCase):
    def test_padding_and_nonfinite(self):
        x = np.array([[[1.,2.],[1e10,1e10]],[[3.,4.],[5.,6.]]])
        mask = np.array([[1,0],[1,1]])
        rows = valid_rows(x,mask)
        s = InputStats()
        s.add(rows)
        self.assertEqual(s.scale(),6/127)
        self.assertEqual(s.tokens,3)
        x[0,1,0] = np.nan
        with self.assertRaisesRegex(ValueError,'NaN/Inf'):
            valid_rows(x,mask)
        with self.assertRaises(ValueError):
            valid_rows(np.ones((2,3,4)),np.ones((3,2)))
        for bad in (np.nan,np.inf):
            with self.assertRaises(ValueError):
                quantize_weight([[bad]])

    def test_zero_tensor(self):
        stats = InputStats()
        stats.add(np.zeros((2,3)))
        self.assertEqual(stats.scale(),1)
        op = LinearCalibration('zero', (4,3), lambda sl: np.zeros((sl.stop-sl.start,3)), stats.scale(), Profile(), Options())
        report,_ = op.run(lambda cb: cb(np.zeros((2,3))))
        self.assertEqual(report['calibration']['local_total']['mse'],0)
        self.assertEqual(op.s10,1)

    def test_reservoir_bounded_reproducible(self):
        a,b = Reservoir(17,3),Reservoir(17,3)
        data = np.arange(1000)
        a.add(data)
        for block in np.array_split(data,10):
            b.add(block)
        np.testing.assert_array_equal(a.values,b.values)
        self.assertEqual(len(a.values),17)
        self.assertEqual(a.seen,1000)
        stats = InputStats(3,90,31)
        stats.add(data.reshape(-1,1))
        self.assertEqual(len(stats.sample.values),31)
        self.assertGreater(stats.scale(),0)

    def test_fixed_lut_constraints(self):
        check_lut_scale(.1,.1)
        with self.assertRaises(ValueError):
            check_lut_scale(.2,.1)
        with self.assertRaises(ValueError):
            Options(mode='mse',fixed_lut_scale=.1)
        with self.assertRaises(ValueError):
            Options(mode='fixed',s10=.2,fixed_lut_scale=.1)
        Options(mode='fixed',s10=.1,fixed_lut_scale=.1)

    def test_metrics_against_dense_oracle_and_heldout(self):
        op,report,_,x,w = fixture()
        wq,sw = quantize_weight(w)
        acc = exact_accumulator(quantize_input(x,op.sx),wq)
        p = op.params
        hw = op.profile.apply(acc,p['multiplier'],p['shift'])
        errors = {'requantization': hw*op.s10-acc*(op.sx*sw),
                  'local_total': hw*op.s10-x@w.T,
                  'parameter_approximation': hw.astype(float)-ideal(acc,p['ratio'])}
        for key,e in errors.items():
            self.assertAlmostEqual(report['calibration'][key]['mse'],float(np.mean(e**2)))
            self.assertAlmostEqual(report['calibration'][key]['mae'],float(np.mean(np.abs(e))))
            self.assertAlmostEqual(report['calibration'][key]['max_absolute_error'],float(np.max(np.abs(e))))
        before = (op.sx,op.s10,op.params['multiplier'].copy())
        validation,_ = op.evaluate(lambda cb: cb(x*100))
        self.assertEqual(op.sx,before[0])
        self.assertEqual(op.s10,before[1])
        np.testing.assert_array_equal(op.params['multiplier'],before[2])
        self.assertGreater(validation['clipping']['overall'],0)
        self.assertEqual(report['calibration']['valid_tokens'],len(x))
        self.assertEqual(min(c['mse'] for c in report['selection']['candidates']),
                         next(c['mse'] for c in report['selection']['candidates'] if c['s10'] == op.s10))

    def test_reproducibility(self):
        a = fixture()[0]
        b = fixture()[0]
        self.assertEqual(a.s10,b.s10)
        np.testing.assert_array_equal(a.params['multiplier'],b.params['multiplier'])
        np.testing.assert_array_equal(a.params['shift'],b.params['shift'])

    def test_shared_consumer_groups_and_selection(self):
        names = select_names(18,'0','q_proj,k_proj,v_proj')
        self.assertEqual(len({input_group(name) for name in names}),1)
        self.assertEqual(len(select_names(18,'all','all',True)),127)
        self.assertEqual(input_group('model.layers.0.mlp.gate_proj'),input_group('model.layers.0.mlp.up_proj'))
        for layers,modules in [('18','q_proj'),('0','QK'),('0,0','q_proj')]:
            with self.assertRaises(ValueError):
                select_names(18,layers,modules)

    def test_gemma2_rejected(self):
        with self.assertRaisesRegex(ValueError,'original Gemma 2B'):
            validate_config(types.SimpleNamespace(model_type='gemma2'))

    def test_data_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'data.jsonl'
            p.write_text('\n{"text":"hello"}\n{"text":"world"}\n')
            texts,info = load_texts(p,1)
            self.assertEqual(texts,['hello'])
            self.assertEqual(info['actual_samples'],1)
            p.write_text('{"text":12}\n')
            with self.assertRaisesRegex(ValueError,'data.jsonl:1'):
                load_texts(p,2)
            p = Path(tmp)/'data.txt'
            p.write_text('first\n\nsecond\n')
            self.assertEqual(load_texts(p,8)[0],['first','second'])


class ExportTests(unittest.TestCase):
    def test_export_reload_padding_mapping_and_compiler(self):
        op,report,vectors,x,w = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            exporter = Exporter(tmp,op.profile,dict(model='synthetic'),allow_unverified=True,save_weights=True)
            exporter.add(op,report,vectors)
            with self.assertRaisesRegex(ValueError,'Duplicate'):
                exporter.add(op,report,vectors)
            original_name = op.name
            op.name = 'other'
            exporter.add(op,report,vectors)
            op.name = original_name
            manifest = exporter.finish()
            entries = manifest['modules']
            self.assertEqual(entries[0]['valid_channels'],19)
            self.assertEqual(entries[0]['padded_channels'],13)
            self.assertEqual(entries[1]['parameter_file_offset'],128)
            data = (Path(tmp)/'qparams.bin').read_bytes()
            self.assertEqual(len(data),256)
            words = np.frombuffer(data,dtype='<u4')
            np.testing.assert_array_equal(words[19:32],0)
            profile,(m,s,z),entry = reload_parameters(tmp,op.name,from_binary=True)
            _,npz_params,_ = reload_parameters(tmp,op.name)
            for binary_value,npz_value in zip((m,s,z),npz_params):
                np.testing.assert_array_equal(binary_value,npz_value)
            acc = exact_accumulator(quantize_input(x,op.sx),quantize_weight(w)[0])
            np.testing.assert_array_equal(profile.apply(acc,m,s,z),op.profile.apply(acc,op.params['multiplier'],op.params['shift']))
            np.testing.assert_array_equal(np.load(Path(tmp)/'op0000.weights.int8.npy'),quantize_weight(w)[0])
            with self.assertRaisesRegex(ValueError,'unverified'):
                quant_param_kwargs(tmp,op.name+'.weight',0x1000)
            kwargs = quant_param_kwargs(tmp,op.name+'.weight',0x1000,output_channel_start=16,allow_unverified=True)
            self.assertEqual(kwargs,{'quant_param_addr':0x1040})
            # Extract only existing packer AST: importing compile_to_bin would load meta model & write files.
            tree = ast.parse((ROOT/'src/compile_to_bin.py').read_text())
            node = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == 'build_rs2_struct')
            namespace = {'struct': struct}
            exec(compile(ast.Module(body=[node],type_ignores=[]),'existing compiler packer','exec'),namespace)
            packed = namespace['build_rs2_struct'](**kwargs)
            self.assertEqual(struct.unpack_from('<Q',packed,24)[0],0x1040)
            for base,channel in [(3,0),(0,1),(0,32),(2**64,0)]:
                with self.assertRaises(ValueError):
                    quant_param_kwargs(tmp,op.name+'.weight',base,output_channel_start=channel,allow_unverified=True)
            with self.assertRaises(KeyError):
                quant_param_kwargs(tmp,'unknown',0,allow_unverified=True)
            with self.assertRaises(FileExistsError):
                Exporter(tmp,Profile(),{})

    def test_default_blocks_binary(self):
        op,report,vectors,_,_ = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            ex = Exporter(tmp,op.profile,dict(model='synthetic'))
            ex.add(op,report,vectors)
            manifest = ex.finish()
            self.assertFalse(manifest['binary_export']['present'])
            self.assertFalse((Path(tmp)/'qparams.bin').exists())
            self.assertTrue((Path(tmp)/'qparams.npz').exists())

    def test_inaccurate_blocks_binary_even_with_unverified_flag(self):
        op,report,vectors,_,_ = fixture()
        op.params['status'][1] = 'underflow_to_zero'
        with tempfile.TemporaryDirectory() as tmp:
            ex = Exporter(tmp,op.profile,{},allow_unverified=True)
            ex.add(op,report,vectors)
            self.assertFalse(ex.finish()['binary_export']['present'])

    def test_cli_e2e_and_vectors(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            manifest = main(['--synthetic','--output-dir',tmp,'--allow-unverified-export','--scale-mode','mse'])
            self.assertEqual(manifest['model'],'SYNTHETIC FIXTURE, NOT GEMMA')
            self.assertTrue(manifest['binary_export']['present'])
            vectors = json.loads((Path(tmp)/'test_vectors.json').read_text())['modules']['synthetic.linear']
            p = Profile()
            for v in vectors:
                m,s,z = p.unpack(v['word'])
                self.assertEqual(int(p.apply(np.array(v['accumulator']),m,s,z)),v['expected_int10'])
            for required in ('manifest.json','scales.npz','qparams.npz','qparams.bin','report.json','compiler_mapping.json'):
                self.assertTrue((Path(tmp)/required).is_file())

    def test_existing_lut_contract(self):
        spec = importlib.util.spec_from_file_location('existing_lut',ROOT/'src/LUT.py')
        lut = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(lut)
        with tempfile.TemporaryDirectory() as tmp:
            _,_,_,values = lut.generate_hardware_lut('gelu',input_scale=.1,input_bits=10,out_dir=tmp)
            self.assertEqual(len(values),1024)
            self.assertEqual(values[1023],0)  # signed index -1 -> x=-0.1: negative GeLU clipped to 0
            self.assertGreater(values[10],127)  # existing output is unsigned, not symmetric INT8
            self.assertEqual((Path(tmp)/'gelu_lut_1024B.bin').read_bytes(),(ROOT/'luts/gelu_lut_1024B.bin').read_bytes())


if __name__ == '__main__':
    unittest.main()

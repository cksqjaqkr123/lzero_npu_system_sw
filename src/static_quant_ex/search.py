"""search.py - p999/511 fix + RepresentativeRowSampler + large p999 fallback + ndarray JSON fix"""
import numpy as np
import pathlib

try:
    from .hardware import finite, positive_scale, integers, ideal, check_lut_scale, GeluLut
    from .core import Reservoir
except (ImportError, ModuleNotFoundError, ValueError):
    from hardware import finite, positive_scale, integers, ideal, check_lut_scale, GeluLut
    from core import Reservoir

_SUMMARY_A = {}
try:
    for p in [
        pathlib.Path("luts/summary_A.csv"),
        pathlib.Path("./summary_A.csv"),
        pathlib.Path("src/static_quant_ex/summary_A.csv"),
        pathlib.Path("/home/parkchanbum/lzero_npu_system_sw/summary_A.csv"),
        pathlib.Path("/home/parkchanbum/lzero_npu_system_sw/src/static_quant_ex/summary_A.csv"),
    ]:
        if p.exists():
            import csv
            with open(p) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        layer = int(row['layer'])
                        _SUMMARY_A[layer] = float(row['pre_p999'])
                    except:
                        pass
            print(f"[search.py] Loaded summary_A.csv {p} {len(_SUMMARY_A)} layers")
            break
except Exception as e:
    print(f"[search.py] summary_A load failed: {e}")

def _extract_ndarrays_from_obj(obj, max_depth=3, visited=None):
    found = []
    if visited is None:
        visited = set()
    try:
        oid = id(obj)
        if oid in visited:
            return found
        visited.add(oid)
        if isinstance(obj, np.ndarray) and obj.size>0 and obj.dtype!=object:
            try:
                if np.issubdtype(obj.dtype, np.number):
                    found.append(obj.ravel())
                    return found
            except:
                pass
        if isinstance(obj, (list, tuple)):
            for item in obj:
                if max_depth>0:
                    found.extend(_extract_ndarrays_from_obj(item, max_depth-1, visited))
            return found
        if isinstance(obj, dict):
            for v in obj.values():
                if max_depth>0:
                    found.extend(_extract_ndarrays_from_obj(v, max_depth-1, visited))
            return found
        if hasattr(obj, '__dict__'):
            for k,v in obj.__dict__.items():
                if k.startswith('_'):
                    continue
                if max_depth>0:
                    found.extend(_extract_ndarrays_from_obj(v, max_depth-1, visited))
        for attr in ['samples','data','rows','buffer','all_data','storage','reservoir','sampled_rows','representative_rows','inputs','activations','row_data','sampled_data']:
            if hasattr(obj, attr):
                try:
                    v=getattr(obj,attr)
                    if isinstance(v, np.ndarray) and v.size>0 and v.dtype!=object:
                        found.append(v.ravel())
                except:
                    pass
        for method_name in ['all','get','to_array','as_array','get_all','collect','get_rows','sampled']:
            if hasattr(obj, method_name):
                try:
                    m=getattr(obj, method_name)
                    if callable(m):
                        v=m()
                        if isinstance(v, np.ndarray) and v.size>0 and v.dtype!=object:
                            found.append(v.ravel())
                        elif isinstance(v, list):
                            for item in v:
                                if isinstance(item, np.ndarray) and item.size>0:
                                    found.append(item.ravel())
                except:
                    pass
    except:
        pass
    return found

def _get_sample_array(sample):
    try:
        if sample is None:
            return None
        if isinstance(sample, np.ndarray) and sample.size>0 and sample.dtype!=object:
            return sample.ravel()
        extracted=_extract_ndarrays_from_obj(sample, max_depth=3)
        if not extracted:
            return None
        candidates=[]
        for arr in extracted:
            if arr.size<100:
                continue
            try:
                if arr.dtype==object:
                    continue
                amax=float(np.max(np.abs(arr[np.isfinite(arr)]))) if np.any(np.isfinite(arr)) else 0
                if 0.5<amax<200:
                    candidates.append((arr,amax,arr.size))
            except:
                pass
        if candidates:
            candidates.sort(key=lambda x: x[2], reverse=True)
            for arr,amax,size in candidates:
                if amax<50:
                    return arr
            return candidates[0][0]
        else:
            extracted.sort(key=lambda x: x.size, reverse=True)
            return extracted[0] if extracted else None
    except Exception as e:
        print(f"[_get_sample_array] failed: {e}")
    return None

def _compute_p999(arr, percentile=99.9):
    if arr is None or arr.size==0 or arr.dtype==object:
        return None
    try:
        abs_arr=np.abs(arr)
        abs_arr=abs_arr[np.isfinite(abs_arr)]
        if abs_arr.size==0:
            return None
        p=float(np.percentile(abs_arr, percentile))
        return max(p,1e-6)
    except:
        return None

def joint_scale_search(read_weight, shape, sw, sample, activation_thresholds, baseline, profile, options):
    N,K=shape
    raw=_get_sample_array(sample)
    s_X=None
    pre_p999_for_bin=None

    if raw is not None and raw.size>0:
        try:
            absmax=float(np.max(np.abs(raw[np.isfinite(raw)]))) if np.any(np.isfinite(raw)) else 1.0
            p999=_compute_p999(raw,99.9)
            p99=_compute_p999(raw,99.0)
            p95=_compute_p999(raw,95.0)
            if p999 is not None and p999>50:
                print(f"[p999 search] WARNING p999={p999:.2f} too large absmax={absmax:.2f} using p99={p99:.2f} fallback")
                if p99 is not None:
                    p999=p99*1.35
            s_X_absmax=absmax/127.0 if absmax>0 else 0.01
            if p999 is not None:
                s_X_p999=p999/127.0
                pre_p999_for_bin=p999
            s_X_orig=None
            if hasattr(sample,'scale'):
                try:
                    s_X_orig=float(sample.scale())
                except:
                    s_X_orig=s_X_absmax
            else:
                s_X_orig=s_X_absmax
            s_X=s_X_p999 if s_X_p999 is not None else s_X_orig
            print(f"[p999 search] N={N} K={K} absmax={absmax:.4f} p999={p999:.4f} p99={p99:.4f} p95={p95:.4f} s_X={s_X:.6f} raw_size={raw.size}")
        except Exception as e:
            print(f"[p999 search] calc failed {e}")
            try:
                s_X=float(sample.scale()) if hasattr(sample,'scale') else 0.1
            except:
                s_X=0.1
            pre_p999_for_bin=None
    else:
        print(f"[p999 search] raw extraction failed type={type(sample).__name__} fallback")
        try:
            s_X=float(sample.scale()) if hasattr(sample,'scale') else 0.1
        except:
            s_X=0.1
        pre_p999_for_bin=None

    if options.s10 is not None:
        s10=float(options.s10)
    else:
        if pre_p999_for_bin is not None:
            s10=pre_p999_for_bin/511.0
            if s10>0.08:
                print(f"[p999 search] s10={s10:.6f} too large capping to 0.04 (pre_p999 {pre_p999_for_bin:.2f})")
                if _SUMMARY_A:
                    median_p999=float(np.median(list(_SUMMARY_A.values())))
                    s10=median_p999/511.0
                    print(f"[p999 search] median p999 {median_p999:.4f} -> s10={s10:.6f}")
                else:
                    s10=0.04
        else:
            s10=0.04

    s10=max(s10,1e-6)
    print(f"[p999 search] final s10={s10:.6f} pre_p999_est={s10*511:.4f}")

    # FIX: sync s10 to closest LUT bin to avoid mismatch (L0 23.1 -> 40.0 problem)
    # Load scale map if available and pick closest bin pre
    try:
        _scale_map_path = None
        for cand in [
            pathlib.Path("luts/gelu_scale_map_16b.json"),
            pathlib.Path("./luts/gelu_scale_map_16b.json"),
            pathlib.Path("src/static_quant_ex/../..//luts/gelu_scale_map_16b.json"),
            pathlib.Path("/mnt/data/luts/gelu_scale_map_16b.json"),
            pathlib.Path("luts/gelu_scale_map.json"),
        ]:
            if cand.exists():
                _scale_map_path = cand
                break
        if _scale_map_path is not None and pre_p999_for_bin is not None:
            import json as _json
            with open(_scale_map_path) as _f:
                _map = _json.load(_f)
            # find closest bin pre
            _best = min(_map, key=lambda m: abs(float(m.get('pre_p999',0)) - float(pre_p999_for_bin)))
            _best_pre = float(_best.get('pre_p999'))
            _best_in = float(_best.get('in_scale'))
            # Use bin's in_scale as s10 to match LUT generation
            # Only override if close enough (< 2x) to avoid huge jump, else keep computed
            if abs(_best_pre - pre_p999_for_bin) < pre_p999_for_bin:  # within 100% diff
                print(f"[bin-sync] pre_p999 {pre_p999_for_bin:.4f} -> closest bin {_best.get('bin')} pre {_best_pre:.4f} s10 {_best_in:.6f} (was {s10:.6f})")
                s10 = _best_in
    except Exception as _e:
        print(f"[bin-sync] failed {_e}")

    check_lut_scale(s10, options.fixed_lut_scale)

    try:
        sw_arr=np.asarray(sw).ravel()
        if sw_arr.size==1:
            sw_arr=np.full(N,float(sw_arr[0]))
        elif sw_arr.size!=N:
            sw_arr=np.full(N,float(np.mean(sw_arr)))
    except:
        sw_arr=np.full(N,0.001)

    sw_arr=np.maximum(sw_arr,1e-8)
    ratios=(s_X*sw_arr)/s10
    ratios=np.maximum(ratios,0.0)
    finite(ratios,"ratio")

    approx=profile.approximate(ratios, tolerance=options.ratio_tolerance)

    chosen_arrays=dict(
        s_X=float(s_X),
        s10=float(s10),
        multiplier=approx['multiplier'],
        shift=approx['shift'],
        ratio=approx['ratio'],
        realized_ratio=approx['realized_ratio'],
        relative_error=approx['relative_error'],
        absolute_error=approx['absolute_error'],
        status=approx['status'],
        pre_p999=float(pre_p999_for_bin) if pre_p999_for_bin is not None else None,
        pre_absmax=float(np.max(np.abs(raw))) if raw is not None and raw.size else None,
    )

    selection=dict(
        selected=dict(
            s_X=float(s_X),
            s10=float(s10),
            multiplier=approx['multiplier'].tolist() if isinstance(approx['multiplier'], np.ndarray) else approx['multiplier'],
            shift=approx['shift'].tolist() if isinstance(approx['shift'], np.ndarray) else approx['shift'],
            ratio=approx['ratio'].tolist() if isinstance(approx['ratio'], np.ndarray) else approx['ratio'],
            realized_ratio=approx['realized_ratio'].tolist() if isinstance(approx['realized_ratio'], np.ndarray) else approx['realized_ratio'],
            relative_error=approx['relative_error'].tolist() if isinstance(approx['relative_error'], np.ndarray) else approx['relative_error'],
            absolute_error=approx['absolute_error'].tolist() if isinstance(approx['absolute_error'], np.ndarray) else approx['absolute_error'],
            status=approx['status'].tolist() if isinstance(approx['status'], np.ndarray) else approx['status'],
            pre_p999=float(pre_p999_for_bin) if pre_p999_for_bin is not None else None,
            pre_absmax=float(np.max(np.abs(raw))) if raw is not None and raw.size else None,
        ),
        sampling=dict(retained=int(raw.size) if raw is not None else 0, bos_rows=0, early_rows=0, general_rows=0),
        candidates={},
        bin_selection_mode="p999" if pre_p999_for_bin is not None else "fallback"
    )

    return s_X,s10,chosen_arrays,selection

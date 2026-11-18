#!/usr/bin/env python3
"""최종 CSV v3 - v2의 base 변수 섀도잉 버그 수정
- base_path: Path (절대 덮지 않음)
- lut_base / r_base: int 주소용
"""
import json, csv, pathlib
base_path = pathlib.Path("diagnostics/ex_norm10_p6_bos_int10/lut_delivery")
out = base_path / "csv_final"
out.mkdir(parents=True, exist_ok=True)
def align(a,n): return (a+n-1)//n*n

# Norm
norm_gen=json.load(open(base_path/"norm/norm_selected_parameters.json"))
norm_bos=json.load(open(base_path/"norm/calib_bos_absmax_1conv.json"))
def order(d):
    ks=[k for k in d if k.startswith("blk.")]
    def kf(k):
        pp=k.split("."); return (int(pp[1]), 0 if "attn" in k else 1)
    return sorted(ks,key=kf)+[k for k in d if not k.startswith("blk.")]
addr=0; rows=[]
for name in order(norm_gen):
    g=norm_gen[name]; b=norm_bos.get(name,g); ba=align(addr,4)
    rows.append({"norm":name,"kind":"general","s_in":g["s_in"],"s_out":g["s_out"],"s_in_addr":ba,"s_out_addr":ba+4,"block_addr":ba,"block_size":16})
    rows.append({"norm":name,"kind":"BOS","s_in":b["s_in"],"s_out":b["s_out"],"s_in_addr":ba+8,"s_out_addr":ba+12,"block_addr":ba,"block_size":16})
    addr=ba+16
with open(out/"norm_scales_final.csv","w",newline="") as f:
    w=csv.DictWriter(f, fieldnames=["norm","kind","s_in","s_out","s_in_addr","s_out_addr","block_addr","block_size"]); w.writeheader(); w.writerows(rows)
print(f"norm {len(rows)} rows {addr}B OK")

# GELU + RoPE
npu=json.load(open(base_path/"npu/npu_lut_tables.json"))
gelu_list=npu.get("gelu",[]); rope_list=npu.get("rope",[])
def gelu_key(e):
    try: return int(e.get("name","").split(".")[1])
    except: return 999
gelu_list=sorted(gelu_list, key=gelu_key)
gelu_rows=[]
for e in gelu_list:
    layer=gelu_key(e)
    gelu_rows.append({"layer":layer,"name":e.get("name"),"s_in":e.get("s_in"),"s_out":e.get("s_out"),"s_in_mode":e.get("s_in_mode"),"lut_base_addr":align(848,2048)+layer*2048,"lut_entries":len(e.get("lut",[])),"lut_bytes":2048})
with open(out/"gelu_scales_final.csv","w",newline="") as f:
    w=csv.DictWriter(f, fieldnames=["layer","name","s_in","s_out","s_in_mode","lut_base_addr","lut_entries","lut_bytes"]); w.writeheader(); w.writerows(gelu_rows)
print(f"gelu_scales {len(gelu_rows)} rows, layer0 s_in={gelu_rows[0]['s_in']}")

with open(out/"gelu_lut_full.csv","w",newline="") as f:
    w=csv.DictWriter(f, fieldnames=["layer","index","input_code","input_deq","output_code","addr"]); w.writeheader()
    for gr in gelu_rows:
        layer=gr["layer"]; s_in=float(gr["s_in"]); lut_base=gr["lut_base_addr"]
        lut=None
        for e in gelu_list:
            if gelu_key(e)==layer:
                lut=e.get("lut",[]); break
        for idx in range(1023):
            code=idx-511; deq=code*s_in; oc=lut[idx] if lut is not None and idx < len(lut) else ""
            w.writerow({"layer":layer,"index":idx,"input_code":code,"input_deq":deq,"output_code":oc,"addr":lut_base+idx*2})
print(f"gelu_lut_full {26*1023} rows OK")

# RoPE - base_path 유지, r_base로 주소 계산
rope=json.load(open(base_path/"rope/rope_calib.json"))
rope_rows=[]
for layer in range(26):
    v=rope.get(f"blk.{layer}",{})
    s_q=v.get("s_q",""); s_k=v.get("s_k","")
    r_base=align(848,2048)+26*2048+layer*8
    rope_rows.append({"layer":layer,"s_q":s_q,"s_k":s_k,"s_q_addr":r_base,"s_k_addr":r_base+4,"block_size":8,"q_p9999":v.get("q_p9999",""),"k_p9999":v.get("k_p9999","")})
with open(out/"rope_scales_final.csv","w",newline="") as f:
    w=csv.DictWriter(f, fieldnames=["layer","s_q","s_k","s_q_addr","s_k_addr","block_size","q_p9999","k_p9999"]); w.writeheader(); w.writerows(rope_rows)
print(f"rope {len(rope_rows)} rows, layer0 s_q={rope_rows[0]['s_q']} s_k={rope_rows[0]['s_k']}")

with open(out/"lut_memory_map_final.csv","w",newline="") as f:
    w=csv.DictWriter(f, fieldnames=["section","base_addr","total_bytes","entries","note"]); w.writeheader()
    w.writerow({"section":"norm_scales","base_addr":0,"total_bytes":848,"entries":212,"note":"FP32, 53 norms x 4"})
    w.writerow({"section":"gelu_luts","base_addr":align(848,2048),"total_bytes":53248,"entries":26598,"note":"INT16, npu_lut_tables.json gelu lut 그대로"})
    w.writerow({"section":"rope_scales","base_addr":align(848,2048)+53248,"total_bytes":208,"entries":52,"note":"FP32, rope_calib.json s_q/s_k"})
print("DONE v3")

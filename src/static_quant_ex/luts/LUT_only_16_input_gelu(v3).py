import math
import struct
import os
import json

def generate_hardware_lut(func_type, input_scale=0.1, output_scale=255.0,
                          input_bits=8, output_bits=8, out_dir="./luts"):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    table_size = 2 ** input_bits
    bytes_per_entry = output_bits // 8
    expected_size = table_size * bytes_per_entry
    lut_bytes = bytearray(expected_size)
    vals_for_verify = [] 
    half_range = table_size // 2
    for i in range(table_size):
        val_float = float(i if i < half_range else i - table_size)
        x = val_float * input_scale
        if func_type.startswith('gelu'):
            y = 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * (x ** 3))))
            quantized_out = int(round(y * output_scale))
        elif func_type == 'exp':
            y = math.exp(max(-80.0, min(0.0, x)))
            quantized_out = int(round(y * output_scale))
        else:
            y = 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * (x ** 3))))
            quantized_out = int(round(y * output_scale))

        if output_bits == 16:
            if func_type.startswith('gelu'):
                quantized_out = max(-512, min(511, quantized_out))
                lut_bytes[i*2:i*2+2] = struct.pack('<h', quantized_out)
            else:
                quantized_out = max(0, min(65535, quantized_out))
                lut_bytes[i*2:i*2+2] = struct.pack('<H', quantized_out)
        else:
            quantized_out = max(0, min(255, quantized_out))
            lut_bytes[i] = quantized_out & 0xFF
            
        vals_for_verify.append(quantized_out)

    file_name = f"{func_type}_lut_{expected_size}B.bin"
    file_path = os.path.join(out_dir, file_name)
    with open(file_path, "wb") as f:
        f.write(lut_bytes)
    actual_size = os.path.getsize(file_path)
    return func_type, expected_size, actual_size, vals_for_verify


if __name__ == "__main__":
    results = []
    
    # 16-bin 설계: outlier 6개 + 정밀 10개
    # 1676 elements 기준 pre_p999 분포:
    # 40.0, 20.44, 12.31, 8.12, 7.75, 5.18 = outlier 6개
    # 4.90, 4.68, 4.40, 4.21, 4.06, 3.82, 3.75, 3.68, 3.53, 3.51 = 정밀 10개
    GELU_BINS_INFO_16 = [
        (0,  40.0,      38.75,  "OUTLIER0 40.0 absmax 38.75 L17 cover"),
        (1,  20.44,     15.2694,"OUTLIER1 20.44 safety L8/L17"),
        (2,  12.3125,   12.31884,"OUTLIER2 12.31 L0 p999"),
        (3,  8.1249,    8.121019,"OUTLIER3 8.12 L1 p999"),
        (4,  7.75,      7.285714,"OUTLIER4 7.75 L17 p999"),
        (5,  5.1875,    5.0,     "OUTLIER5 5.18 L16 p999"),
        (6,  4.9062,    3.3281,  "PRECISE0 4.90 L14"),
        (7,  4.6875,    3.6406,  "PRECISE1 4.68 L15"),
        (8,  4.4062,    4.22185, "PRECISE2 4.40 L2"),
        (9,  4.2188,    3.1628,  "PRECISE3 4.21 L13"),
        (10, 4.0625,    2.4844,  "PRECISE4 4.06 L6"),
        (11, 3.8281,    2.2812,  "PRECISE5 3.82 L7"),
        (12, 3.75,      2.2344,  "PRECISE6 3.75 L3/L5"),
        (13, 3.6875,    2.7096,  "PRECISE7 3.68 L12"),
        (14, 3.5312,    2.5312,  "PRECISE8 3.53 L11"),
        (15, 3.5156,    2.2344,  "PRECISE9 3.51 L8 minimal"),
    ]

    # 16-bin 10->8 메인: 16*1024 = 16384B
    GELU_16BIN_10_8 = [(bid, pre/511.0, 255.0/post if post else 0, desc) for bid, pre, post, desc in GELU_BINS_INFO_16]
    combined_16_8 = bytearray()
    for bin_id, in_scale, out_scale, desc in GELU_16BIN_10_8:
        func_type = f"gelu_BIN{bin_id}_16b"
        ftype, exp_sz, actual_sz, vals = generate_hardware_lut(
            func_type, input_scale=in_scale, output_scale=out_scale,
            input_bits=10, output_bits=8)
        results.append((ftype, exp_sz, actual_sz, vals))
        with open(f"./luts/{func_type}_lut_{exp_sz}B.bin", "rb") as f:
            combined_16_8.extend(f.read())
    combined_16_8_path = "./luts/gelu_16BIN_combined_16384B_10_8.bin"
    with open(combined_16_8_path, "wb") as f:
        f.write(combined_16_8)

    # 16-bin 10->10 참고용: 16*2048 = 32768B
    GELU_16BIN_10_10 = [(bid, pre/511.0, 511.0/post if post else 0, desc) for bid, pre, post, desc in GELU_BINS_INFO_16]
    combined_16_10 = bytearray()
    for bin_id, in_scale, out_scale, desc in GELU_16BIN_10_10:
        func_type = f"gelu_BIN{bin_id}_16b_10_10"
        ftype, exp_sz, actual_sz, vals = generate_hardware_lut(
            func_type, input_scale=in_scale, output_scale=out_scale,
            input_bits=10, output_bits=16)
        results.append((ftype, exp_sz, actual_sz, vals))
        with open(f"./luts/{func_type}_lut_{exp_sz}B.bin", "rb") as f:
            combined_16_10.extend(f.read())
    combined_16_10_path = "./luts/gelu_16BIN_combined_32768B_10_10.bin"
    with open(combined_16_10_path, "wb") as f:
        f.write(combined_16_10)

    print("\n--- [1] LUT 생성 16-bin ---")
    for func_type, expected, actual, vals in results:
        status = "PASS" if expected == actual else "FAIL"
        print(f"[{func_type:<24}] {expected}B -> {actual}B {status}")

    print(f"\n--- [2] GeLU 16-bin (10->8 메인) ---")
    print(f"메인 10->8 16384B: {combined_16_8_path} {len(combined_16_8)}B = 16*1024")
    print(f"참고 10->10 32768B: {combined_16_10_path} {len(combined_16_10)}B = 16*2048")
    print("addr 10->8 = base + (bin_id << 10) + q10_idx")
    print("addr 10->10 = base + (bin_id << 11) + q10_idx*2")
    print("bin_id 0-5: outlier 4-6개, 6-15: 정밀 10개")

    scale_map_16_8 = [{"bin": bid, "in_scale": ins, "out_scale": outs, "desc": desc,
                    "pre_p999": ins*511, "post_p999": 255/outs if outs else 0}
                   for bid, ins, outs, desc in GELU_16BIN_10_8]
    with open("./luts/gelu_scale_map_16b.json", "w") as f:
        json.dump(scale_map_16_8, f, indent=2, ensure_ascii=False)
    print(f"\nscale_map 16b 10->8: ./luts/gelu_scale_map_16b.json")

    scale_map_16_10 = [{"bin": bid, "in_scale": ins, "out_scale": outs, "desc": desc,
                  "pre_p999": ins*511, "post_p999": 511/outs if outs else 0}
                 for bid, ins, outs, desc in GELU_16BIN_10_10]
    with open("./luts/gelu_scale_map_16b_10_10.json", "w") as f:
        json.dump(scale_map_16_10, f, indent=2, ensure_ascii=False)
    print(f"scale_map 16b 10->10: ./luts/gelu_scale_map_16b_10_10.json")

    # 하위호환 8-bin도 유지
    print("\n--- [3] 기존 8-bin도 함께 생성 (하위호환) ---")
    GELU_BINS_INFO_8 = [
        (0, 20.44,    15.2694, "SAFETY"),
        (1, 12.3125, 12.31884, "L0"),
        (2, 8.1249,   8.121019, "L1"),
        (3, 7.75,     7.285714, "L17"),
        (4, 5.1875,   5.0, "L16"),
        (5, 4.875,    3.32898, "L14"),
        (6, 4.4062,   4.22185, "L2"),
        (7, 3.5156,   2.2388, "L8 minimal"),
    ]
    GELU_8BIN_10_8 = [(bid, pre/511.0, 255.0/post if post else 0, desc) for bid, pre, post, desc in GELU_BINS_INFO_8]
    combined_8 = bytearray()
    for bin_id, in_scale, out_scale, desc in GELU_8BIN_10_8:
        func_type = f"gelu_BIN{bin_id}"
        ftype, exp_sz, actual_sz, vals = generate_hardware_lut(func_type, input_scale=in_scale, output_scale=out_scale, input_bits=10, output_bits=8)
        with open(f"./luts/{func_type}_lut_{exp_sz}B.bin", "rb") as f:
            combined_8.extend(f.read())
    with open("./luts/gelu_BIN0_7_combined_8192B.bin", "wb") as f:
        f.write(combined_8)
    print(f"8-bin 8192B 유지: ./luts/gelu_BIN0_7_combined_8192B.bin")

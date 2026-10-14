import math
import struct
import os

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
        elif func_type == 'sin':
            y = math.sin(x)
            quantized_out = int(round((y + 1.0) * 32767.5))
        elif func_type == 'cos':
            y = math.cos(x)
            quantized_out = int(round((y + 1.0) * 32767.5))
        elif func_type == 'scale':
            epsilon = 1e-5
            y = 1.0 / (abs(x) + epsilon)
            quantized_out = int(round(y * output_scale))
        elif func_type == 'rsqrt':
            epsilon = 1e-5
            y = 1.0 / math.sqrt(abs(x) + epsilon)
            quantized_out = int(round(y * output_scale))
        else:
            y = 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * (x ** 3))))
            quantized_out = int(round(y * output_scale))
        if output_bits == 16:
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
    results.append(generate_hardware_lut('exp', input_scale=0.1, output_scale=65535.0, input_bits=8, output_bits=16))
    
    # === gelu만 8-bin: BIN0~7, 8192B 합본 ===
    # 근거: in_scale = pre_p999/511, out_scale = 255/post_p999 (8bit 출력이라 255 기준)
    GELU_8BIN = [
        (0, 0.04,     16.7,  "SAFETY 20.44/15.25"),
        (1, 0.024095, 20.7,  "L0 12.31"),
        (2, 0.0159,   31.4,  "L1 8.12"),
        (3, 0.015166, 35.0,  "L17 7.75"),
        (4, 0.010152, 51.0,  "L16 5.18"),
        (5, 0.00954,  76.6,  "L14 4.87"),
        (6, 0.008623, 60.4,  "L2 4.40"),
        (7, 0.00688,  113.9, "L8 3.51 최소"),
    ]
    combined_bytes = bytearray()
    for bin_id, in_scale, out_scale, desc in GELU_8BIN:
        func_type = f"gelu_BIN{bin_id}"
        ftype, exp_sz, actual_sz, vals = generate_hardware_lut(
            func_type, input_scale=in_scale, output_scale=out_scale,
            input_bits=10, output_bits=8)
        results.append((ftype, exp_sz, actual_sz, vals))
        with open(f"./luts/{func_type}_lut_{exp_sz}B.bin", "rb") as f:
            combined_bytes.extend(f.read())
    combined_path = "./luts/gelu_BIN0_7_combined_8192B.bin"
    with open(combined_path, "wb") as f:
        f.write(combined_bytes)
    
    results.append(generate_hardware_lut('sin', input_scale=0.1, input_bits=10, output_bits=16))
    results.append(generate_hardware_lut('cos', input_scale=0.1, input_bits=10, output_bits=16))
    results.append(generate_hardware_lut('scale', input_scale=0.1, output_scale=6553.5, input_bits=8, output_bits=16))
    results.append(generate_hardware_lut('rsqrt', input_scale=0.1, output_scale=6553.5, input_bits=8, output_bits=16))

    print("\n--- [1] LUT 생성 ---")
    for func_type, expected, actual, vals in results:
        status = "PASS" if expected == actual else "FAIL"
        print(f"[{func_type.upper():<12}] {expected}B -> {actual}B {status}")
    print(f"\nCombined: {combined_path} {len(combined_bytes)}B = 8*1024")
    print("addr = base + (bin_id << 10) + q10_idx")

    import json
    scale_map = [{"bin": bid, "in_scale": ins, "out_scale": outs, "desc": d,
                  "pre_p999": ins*511, "post_p999": 255/outs if outs else 0}
                 for bid, ins, outs, d in GELU_8BIN]
    with open("./luts/gelu_scale_map.json", "w") as f:
        json.dump(scale_map, f, indent=2)

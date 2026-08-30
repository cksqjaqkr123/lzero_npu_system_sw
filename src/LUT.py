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

        if func_type == 'exp':
            y = math.exp(max(-80.0, min(0.0, x)))
            quantized_out = int(round(y * output_scale))
            
        elif func_type == 'gelu':
            y = 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * (x ** 3))))
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
            
        elif func_type == 'rsqrt':  # 새로 추가된 루트 스케일 (역제곱근)
            epsilon = 1e-5
            y = 1.0 / math.sqrt(abs(x) + epsilon)
            quantized_out = int(round(y * output_scale))

        if output_bits == 16:
            quantized_out = max(0, min(65535, quantized_out))
            lut_bytes[i*2:i*2+2] = struct.pack('<H', quantized_out)
        else:
            # 명시적 Unsigned 8bit 클리핑으로 Wraparound 차단
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
    
    results.append(generate_hardware_lut('exp',   input_scale=0.1, output_scale=65535.0,
                                         input_bits=8,  output_bits=16))
    
    # [확신도 중간] Act(GELU) 10bit입력/8bit출력은 손글씨 설계노트 판독 기반 추정. 하드웨어팀 확인 권장.
    results.append(generate_hardware_lut('gelu',  input_scale=0.1, output_scale=255.0,
                                         input_bits=10, output_bits=8))
    
    results.append(generate_hardware_lut('sin',   input_scale=0.1,
                                         input_bits=10, output_bits=16))
    
    results.append(generate_hardware_lut('cos',   input_scale=0.1,
                                         input_bits=10, output_bits=16))
    
    results.append(generate_hardware_lut('scale', input_scale=0.1, output_scale=6553.5,
                                         input_bits=8,  output_bits=16))
                                         
    # 새로 추가된 루트 스케일 (RMSNorm / LayerNorm 분산 정규화용)
    results.append(generate_hardware_lut('rsqrt', input_scale=0.1, output_scale=6553.5,
                                         input_bits=8,  output_bits=16))

    print("\n--- [1] 칩셋 메모리 적재용 정적 LUT 생성 결과 ---")
    for func_type, expected, actual, vals in results:
        status = "✅ PASS" if expected == actual else "❌ FAIL"
        print(f"[{func_type.upper():<5}] 예상 크기: {expected:<4}B | 실제 크기: {actual:<4}B -> {status}")

    print("\n--- [2] LUT 데이터 무결성 및 수치 검증 ---")
    for func_type, expected, actual, vals in results:
        min_val = min(vals)
        max_val = max(vals)
        
        range_str = f"Min: {min_val:>5}, Max: {max_val:>5}"
        print(f"[{func_type.upper():<5}] {range_str}")
        
        if func_type == 'cos':
            cos_0 = vals[0]
            if cos_0 >= 65530:
                print(f"   ↳ [cos(0) Check] idx=0 값 {cos_0} (최댓값 근접) -> ✅ PASS")
            else:
                print(f"   ↳ [cos(0) Check] idx=0 값 {cos_0} -> ❌ FAIL")
                
        if func_type in ['scale', 'rsqrt']:
            diffs = [abs(vals[i] - vals[i+1]) for i in range(1, 8)]
            print(f"   ↳ [{func_type.upper()} Curve Check] 연속 Diff (idx 1~8): {diffs}")
            
            if diffs[0] > diffs[-1] and all(diffs[i] >= diffs[i+1] for i in range(len(diffs)-1)):
                print(f"   ↳ [{func_type.upper()} Curve Check] Diff가 지속적으로 감소하는 완벽한 감소 곡선 확인 -> ✅ PASS")
            else:
                print(f"   ↳ [{func_type.upper()} Curve Check] 값이 포화되었거나 직선 형태 -> ❌ FAIL")
                
        if func_type == 'gelu':
            v1000 = vals[1000]
            v1023 = vals[1023]
            print(f"   ↳ [GELU Negative Check] idx=1000 (x=-2.4): {v1000}")
            print(f"   ↳ [GELU Negative Check] idx=1023 (x=-0.1): {v1023}")
            
            if 0 <= v1000 <= 5 and 0 <= v1023 <= 5:
                print(f"   ↳ [GELU Negative Check] 음수가 0 근처로 정상 클리핑됨 -> ✅ PASS")
            else:
                print(f"   ↳ [GELU Negative Check] 클리핑 실패 (Wraparound 발생) -> ❌ FAIL")
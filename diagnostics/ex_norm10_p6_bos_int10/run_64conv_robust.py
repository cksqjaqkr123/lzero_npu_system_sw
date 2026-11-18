#!/usr/bin/env python3
"""
64 conv 최종 테스트 - 안 튕기는 드라이버
- conv마다 별도 프로세스로 실행 -> 한 conv가 터져도 전체가 안 죽음
- conv마다 결과 파일을 즉시 저장 -> 끊겨도 이어서 가능
- 이미 있는 conv는 스킵 -> resume
- 실패하면 2번 재시도하고 error 파일 남기고 다음 conv로 넘어감

사용법 (서버, ex_norm10_p6_bos_int10 폴더에서):
  python3 run_64conv_robust.py --n_conv 64 --prefill 256 --tail_k 26
  # 끊겼으면 같은 명령어 다시 치면 남은 conv부터 이어서 함
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_conv", type=int, default=64)
    ap.add_argument("--prefill", type=int, default=256)
    ap.add_argument("--tail_k", type=int, default=26)
    ap.add_argument("--rms_json", default="diagnostics/ex_norm10_p6_bos_int10/calib/norm_selected_parameters.json")
    ap.add_argument("--bos_json", default="diagnostics/ex_norm10_p6_bos_int10/calib/calib_bos_absmax_1conv.json")
    ap.add_argument("--rope_calib", default="diagnostics/ex_norm10_p4/e2e/rope_calib.json")
    ap.add_argument("--base_script", default="diagnostics/ex_norm10_p6_bos_int10/run_tail_sweep.py")
    ap.add_argument("--output_dir", default="diagnostics/ex_norm10_p6_bos_int10/results/final_64conv")
    ap.add_argument("--max_retries", type=int, default=2)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "driver.log", "a", buffering=1)
    def say(s):
        print(s, flush=True)
        log.write(s + "\n")

    # 이미 완료된 conv 확인
    done = set()
    for f in out.glob("conv_*.json"):
        try:
            d = json.loads(f.read_text())
            if "top1" in d or "n_match" in d:
                done.add(f.stem)
        except: pass
    say(f"[driver] resume: {len(done)}/{args.n_conv} done, output={out}")

    for idx in range(args.n_conv):
        stem = f"conv_{idx:03d}"
        if stem in done:
            say(f"[{idx:02d}] skip - already done")
            continue
        # 1 conv만 하는 서브프로세스. run_tail_sweep.py는 --conv_idx를 지원해야 함 (아래 패치 참조)
        # 지원 안 하면 --conv_idx 대신 환경변수 CONV_IDX로도 받게 패치해 둘 것
        ok = False
        for attempt in range(args.max_retries + 1):
            say(f"[{idx:02d}] run attempt {attempt+1}/{args.max_retries+1}")
            env = dict(os.environ)
            env["CONV_IDX"] = str(idx)
            cmd = [
                sys.executable, "-u", args.base_script,
                "--rope_calib", args.rope_calib,
                "--n_conv", "1",
                "--conv_idx", str(idx),
                "--prefill", str(args.prefill),
                "--rms_json", args.rms_json,
                "--tail_k", str(args.tail_k),
                "--output_dir", str(out / stem),
            ]
            # bos_json은 argparse 패치가 안 먹혔던 전적이 있어서, 코드에 하드코딩이 기본.
            # 만약 스크립트가 --bos_json을 지원하면 아래 줄 주석 해제:
            # cmd += ["--bos_json", args.bos_json]
            t0 = time.time()
            r = subprocess.run(cmd, env=env)
            dt = time.time() - t0
            if r.returncode == 0 and (out / stem / "result.json").exists():
                # conv 단위 결과를 평탄화: out/conv_XXX.json 로 복사
                try:
                    src = out / stem / "result.json"
                    dst = out / f"{stem}.json"
                    dst.write_text(src.read_text())
                    say(f"[{idx:02d}] OK {dt:.1f}s")
                    ok = True
                    break
                except Exception as e:
                    say(f"[{idx:02d}] copy fail {e}")
            else:
                say(f"[{idx:02d}] FAIL rc={r.returncode} {dt:.1f}s, retry...")
                time.sleep(5)
        if not ok:
            (out / f"{stem}.error").write_text(f"failed after {args.max_retries+1} attempts\n")
            say(f"[{idx:02d}] ERROR - skip to next conv, 전체 중단 안 함")
    say("[driver] all done. aggregate with: python3 aggregate_64conv.py --input_dir " + str(out))

if __name__ == "__main__":
    main()

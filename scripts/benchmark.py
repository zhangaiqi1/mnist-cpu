import time, torch, argparse, numpy as np
from pathlib import Path
from PIL import Image
from mycv.predictor import Predictor
import os


def benchmark(ckpt, data, quantized, batch_size=1, runs=100):
    pred = Predictor(ckpt, data=data, quantized=quantized)
    dummy = Image.new('L', (28, 28)) if data == 'mnist' else Image.new('RGB', (32, 32))
    # warmup
    for _ in range(10):
        pred.predict(dummy)

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        pred.predict(dummy)
        times.append((time.perf_counter() - t0) * 1000)  # ms
    lat = np.mean(times)
    thr = 1000 / lat * batch_size
    return lat, thr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt')
    parser.add_argument('--data', default='mnist')
    parser.add_argument('--quantized', action='store_true')
    args = parser.parse_args()

    lat, thr = benchmark(args.ckpt, args.data, args.quantized)
    tag = "INT8" if args.quantized else "FP32"
    print(f"{tag}: latency={lat:.2f} ms, throughput={thr:.1f} img/s")
    # with open('results/benchmark.md', 'w') as f:
    #     f.write(f"| 精度 | 大小(MB) | 延迟(ms) | 吞吐(img/s) |\n")
    #     f.write(f"|------|----------|----------|-------------|\n")
    #     fp32_sz = os.path.getsize('ckpt/best.pth') / 1024 / 1024
    #     int8_sz = os.path.getsize('ckpt/best_int8.pth') / 1024 / 1024
    #     f.write(f"| FP32 | {fp32_sz:.1f} | 28.4 | 35.2 |\n")
    #     f.write(f"| INT8 | {int8_sz:.1f} | 11.1 | 90.1 |\n")
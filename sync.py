import time
import cupy as cp

x = cp.arange(10_000_000, dtype=cp.float32)
y = (x * 1.0001).sum()   # reduction -> scalar on GPU

# warmup
float(y)

t0 = time.perf_counter()
for _ in range(10000):
    float(y)            # blocking readback of a scalar
t1 = time.perf_counter()

print("avg scalar readback (us):", (t1 - t0) * 1e6 / 10000)
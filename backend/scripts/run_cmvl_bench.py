import sys
import time


def dummy_cmvl_workload(n):
    # deterministic, cheap operation to approximate CPU work for a benchmark placeholder
    s = 0
    for i in range(n):
        s += (i & 0xFF)
    return s


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000000
    start = time.time()
    _ = dummy_cmvl_workload(n)
    elapsed = time.time() - start
    print(f"CMVL dummy workload: iterations={n}, elapsed={elapsed:.4f}s")

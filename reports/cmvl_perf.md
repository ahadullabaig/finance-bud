# CMVL Performance Benchmark Report

This document stores baseline benchmark results for the CMVL verification workload and instructions to reproduce.

## How to run (example)

PowerShell (Windows):

```powershell
.\scripts\run_cmvl_bench.ps1 -iterations 1000000
```

Or run directly with Python:

```powershell
python .\scripts\run_cmvl_bench.py 1000000
```

## Baseline result (placeholder)

- Date: YYYY-MM-DD
- Machine: replace-with-machine-info
- Iterations: 1000000
- Elapsed: 0.1234s

## Notes

- Replace the dummy workload with a real CMVL invocation (call into the verifier entrypoint) when available.
- Add memory and CPU profiling steps here.

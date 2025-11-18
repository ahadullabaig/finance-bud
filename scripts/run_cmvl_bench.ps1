param(
    [int]$iterations = 1000000
)

Write-Output "Running CMVL benchmark placeholder with $iterations iterations..."
if (Get-Command python -ErrorAction SilentlyContinue) {
    python .\scripts\run_cmvl_bench.py $iterations
} else {
    Write-Output "Python not found. Install Python or run the Python script directly: python .\scripts\run_cmvl_bench.py <iterations>"
}

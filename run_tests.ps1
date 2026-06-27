# ============================================================
# Jalankan SEMUA test internal dari root project di Windows NATIVE.
# Tidak butuh bash / Git Bash. Pakai interpreter Python dari venv yg aktif.
#
# Cara pakai (PowerShell, venv sudah aktif):
#   .\run_tests.ps1
#
# Kalau diblokir execution policy, jalankan sekali:
#   powershell -ExecutionPolicy Bypass -File .\run_tests.ps1
# ============================================================
Set-Location -Path $PSScriptRoot
$env:PYTHONPATH = "."

# Pilih interpreter: utamakan 'python', fallback ke launcher 'py'.
$py = $null
foreach ($cand in @("python", "py")) {
    if (Get-Command $cand -ErrorAction SilentlyContinue) { $py = $cand; break }
}
if (-not $py) {
    Write-Host "ERROR: Python tidak ditemukan. Aktifkan venv dulu, mis:"
    Write-Host "       .\.venv-dml\Scripts\Activate.ps1"
    exit 127
}

$pass = 0; $fail = 0; $failed = @()
Get-ChildItem -Path "tests" -Filter "test_*.py" | Sort-Object Name | ForEach-Object {
    $f = "tests/" + $_.Name
    $log = & $py $_.FullName 2>&1
    if ($LASTEXITCODE -eq 0) {
        $pass++
        Write-Host "PASS  $f"
    } else {
        $fail++
        $failed += $f
        Write-Host "FAIL  $f"
        $log | Select-Object -Last 8 | ForEach-Object { Write-Host "      $_" }
    }
}
Write-Host "============================="
Write-Host "PASS=$pass FAIL=$fail"
if ($failed.Count -gt 0) { Write-Host ("FAILED: " + ($failed -join " ")) }
exit $fail

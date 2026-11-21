# Prefer a ready native Windows environment; WSL is only a fallback.
# Arguments are passed as separate values, including paths containing spaces.
$ErrorActionPreference = 'Continue'
$demoArguments = @($args)
$projectDir = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectDir
$candidates = @()
if ($env:PLANT_DISEASE_PYTHON) {
    $candidates += @{ Command = $env:PLANT_DISEASE_PYTHON; Prefix = @() }
} else {
    foreach ($candidate in @('.venv\Scripts\python.exe', 'venv\Scripts\python.exe')) {
        $candidates += @{ Command = (Join-Path $projectDir $candidate); Prefix = @() }
    }
    if ($env:VIRTUAL_ENV) {
        $candidates += @{ Command = (Join-Path $env:VIRTUAL_ENV 'Scripts\python.exe'); Prefix = @() }
    }
    foreach ($name in @('python.exe', 'python3.exe', 'py.exe')) {
        foreach ($command in @(Get-Command $name -CommandType Application -ErrorAction SilentlyContinue)) {
            # Windows Store aliases may open the Store instead of running Python.
            if ($command.Source -notlike '*\WindowsApps\*') {
                $prefix = @()
                if ($name -eq 'py.exe') { $prefix = @('-3') }
                $candidates += @{ Command = $command.Source; Prefix = $prefix }
            }
        }
    }
}
foreach ($candidate in $candidates) {
    $python = $candidate.Command
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { continue }
    $prefix = $candidate.Prefix
    $LASTEXITCODE = 1
    $reason = & $python @prefix scripts/launch_demo.py --check @demoArguments 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host 'Using native Windows Python; WSL is not needed.'
        & $python @prefix scripts/launch_demo.py @demoArguments
        exit $LASTEXITCODE
    }
    Write-Host "Not ready: $python"
    $reason | ForEach-Object { Write-Host "$_" }
}
# An explicitly selected interpreter should fail clearly instead of being ignored.
if (-not $env:PLANT_DISEASE_PYTHON -and (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
    Write-Host 'No ready native environment. Checking the default WSL distribution...'
    & wsl.exe --exec bash scripts/start_demo.sh @demoArguments
    exit $LASTEXITCODE
}
Write-Host 'No ready Python environment found. See docs/presentation/demo-guide.md.'
exit 1

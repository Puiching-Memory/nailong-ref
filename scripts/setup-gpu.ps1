param(
    [string]$Python = "data/gpu_env/Scripts/python.exe"
)

$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path (Join-Path $PSScriptRoot ".."))

if (-not (Test-Path -LiteralPath $Python)) {
    $environment = Split-Path (Split-Path $Python -Parent) -Parent
    uv venv $environment --python 3.12
    if ($LASTEXITCODE -ne 0) { throw "uv venv failed" }
}

uv pip install --python $Python -e . --no-deps
if ($LASTEXITCODE -ne 0) { throw "Editable project installation failed" }

& $Python -c "import demucs, funasr, modelscope, numpy, soundfile, yt_dlp"
if ($LASTEXITCODE -ne 0) {
    uv pip install --python $Python -e . demucs funasr modelscope yt-dlp
    if ($LASTEXITCODE -ne 0) { throw "Pipeline dependency installation failed" }
}

# Keep torch, torchaudio and torchvision on the same official CUDA wheel line.
& $Python -c "import torch, torchaudio, torchvision; assert torch.__version__.startswith('2.11.0+cu130') and torchaudio.__version__.startswith('2.11.0+cu130') and torchvision.__version__.startswith('0.26.0+cu130') and torch.cuda.is_available()"
if ($LASTEXITCODE -ne 0) {
    uv pip install --python $Python --no-deps --reinstall --index-url https://download.pytorch.org/whl/cu130 `
        torch==2.11.0 torchaudio==2.11.0 torchvision==0.26.0
    if ($LASTEXITCODE -ne 0) { throw "CUDA wheel installation failed" }
}

& $Python -c "import torch, torchaudio; assert torch.cuda.is_available(), 'CUDA unavailable'; assert '+cu' in torch.__version__ and '+cu' in torchaudio.__version__, 'CPU wheel still installed'; print(torch.__version__, torchaudio.__version__, torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) { throw "CUDA verification failed" }

Write-Output "GPU pipeline ready: $Python"

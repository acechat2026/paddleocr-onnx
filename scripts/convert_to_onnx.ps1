# convert_to_onnx.ps1
# Convert Paddle models to ONNX format (Windows PowerShell)
# Skips gracefully if paddle2onnx is not installed.

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir

$PaddleDir = Join-Path $ProjectDir "models\paddle"
$OnnxDir = Join-Path $ProjectDir "models\onnx"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Convert Paddle Models to ONNX" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if paddle2onnx is installed
$paddle2onnxCheck = python -c "import paddle2onnx; print(paddle2onnx.__version__)" 2>$null
if (-not $?) {
    Write-Host "[WARN] paddle2onnx not found. Skipping local conversion." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "You can still obtain ONNX models via:"
    Write-Host "  1. GitHub Actions CI (see .github/workflows/convert-models.yml)"
    Write-Host "  2. Install paddle2onnx locally:"
    Write-Host "     pip install paddle2onnx==2.1.0 paddlepaddle==3.0.0 packaging"
    Write-Host ""
    exit 0
}

Write-Host "[INFO] paddle2onnx found: v$paddle2onnxCheck" -ForegroundColor Green

# Create ONNX directory
New-Item -ItemType Directory -Force -Path $OnnxDir | Out-Null

function Convert-Model {
    param(
        [string]$ModelName,
        [string]$ModelType  # det or rec
    )

    $PaddleModelDir = Join-Path $PaddleDir $ModelName
    $OnnxFile = Join-Path $OnnxDir "$ModelName.onnx"

    if (-not (Test-Path $PaddleModelDir)) {
        Write-Host "[WARN] $ModelName not found in $PaddleDir, skipping..." -ForegroundColor Yellow
        return
    }

    if (Test-Path $OnnxFile) {
        Write-Host "[WARN] $OnnxFile already exists, skipping..." -ForegroundColor Yellow
        return
    }

    Write-Host "[INFO] Converting $ModelName..." -ForegroundColor Green

    # Paddle 3.0 models use inference.json; older models use inference.pdmodel
    $modelFile = "inference.pdmodel"
    if (-not (Test-Path (Join-Path $PaddleModelDir "inference.pdmodel")) -and (Test-Path (Join-Path $PaddleModelDir "inference.json"))) {
        $modelFile = "inference.json"
    }

    $cmd = "paddle2onnx --model_dir `"$PaddleModelDir`" --model_filename $modelFile --params_filename inference.pdiparams --save_file `"$OnnxFile`" --opset_version 11 --enable_onnx_checker True"

    try {
        Invoke-Expression $cmd 2>&1 | Where-Object { $_ -notmatch "UserWarning" }

        if (Test-Path $OnnxFile) {
            $fileSize = (Get-Item $OnnxFile).Length
            $sizeMB = [math]::Round($fileSize / 1MB, 2)
            Write-Host "[INFO] $ModelName converted successfully (size: $sizeMB MB)" -ForegroundColor Green
        }
        else {
            Write-Host "[ERROR] Failed to convert $ModelName" -ForegroundColor Red
        }
    }
    catch {
        Write-Host "[ERROR] Conversion failed: $_" -ForegroundColor Red
    }
}

Write-Host "--- Converting PP-OCRv5 Server Models ---" -ForegroundColor Cyan
Write-Host ""

# Detection model
Convert-Model -ModelName "PP-OCRv5_server_det" -ModelType "det"

# Recognition model
Convert-Model -ModelName "PP-OCRv5_server_rec" -ModelType "rec"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Conversion Complete!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "ONNX models saved to: $OnnxDir"
Write-Host ""

Get-ChildItem "$OnnxDir\*.onnx" | ForEach-Object {
    $sizeMB = [math]::Round($_.Length / 1MB, 2)
    Write-Host "  $($_.Name) ($sizeMB MB)"
}

Write-Host ""
Write-Host "You can now run OCR with:"
Write-Host "  .\build\Release\ocr_cli.exe input.jpg -d $OnnxDir\PP-OCRv5_server_det.onnx -r $OnnxDir\PP-OCRv5_server_rec.onnx --dict models\dicts\ppocr_keys_v1.txt"
Write-Host ""

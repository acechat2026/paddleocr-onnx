# download_models.ps1
# Download PP-OCRv5 official models (Windows PowerShell)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir

$ModelBaseUrl = "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0"
$PaddleDir = Join-Path $ProjectDir "models\paddle"
$DictDir = Join-Path $ProjectDir "models\dicts"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Download PP-OCRv5 Official Models" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "[INFO] Paddle models will be saved to: $PaddleDir" -ForegroundColor Green
Write-Host "[INFO] Dictionaries will be saved to: $DictDir" -ForegroundColor Green
Write-Host ""

# Create directories
New-Item -ItemType Directory -Force -Path $PaddleDir | Out-Null
New-Item -ItemType Directory -Force -Path $DictDir | Out-Null

function Download-And-Extract {
    param(
        [string]$ModelName
    )

    $Url = "$ModelBaseUrl/${ModelName}_infer.tar"
    $TarFile = Join-Path $PaddleDir "$ModelName.tar"
    $ExtractDir = Join-Path $PaddleDir $ModelName

    if (Test-Path $ExtractDir) {
        Write-Host "[WARN] $ModelName already exists, skipping..." -ForegroundColor Yellow
        return
    }

    Write-Host "[INFO] Downloading $ModelName..." -ForegroundColor Green

    try {
        Invoke-WebRequest -Uri $Url -OutFile $TarFile -ErrorAction Stop
    }
    catch {
        Write-Host "[ERROR] Failed to download $ModelName : $_" -ForegroundColor Red
        exit 1
    }

    Write-Host "[INFO] Extracting $ModelName..." -ForegroundColor Green

    # Extract using tar (built into Windows 10+)
    tar -xf $TarFile -C $PaddleDir

    # Detect extracted directory (some tars use different top-level names)
    $ExtractedDir = $null
    foreach ($d in Get-ChildItem -Path $PaddleDir -Directory) {
        if ((Test-Path (Join-Path $d.FullName "inference.pdmodel")) -or
            (Test-Path (Join-Path $d.FullName "inference.pdiparams"))) {
            $ExtractedDir = $d
            break
        }
    }

    # Rename to expected name if needed
    if ($ExtractedDir -and ($ExtractedDir.Name -ne $ModelName)) {
        Rename-Item -Path $ExtractedDir.FullName -NewName $ModelName -Force
    }

    Remove-Item $TarFile

    # Verify model files (Paddle 3.0 uses inference.json instead of inference.pdmodel)
    $ModelPath = Join-Path $PaddleDir $ModelName
    $HasPdModel = Test-Path (Join-Path $ModelPath "inference.pdmodel")
    $HasJson = Test-Path (Join-Path $ModelPath "inference.json")
    $HasParams = Test-Path (Join-Path $ModelPath "inference.pdiparams")

    if ($HasParams -and ($HasPdModel -or $HasJson)) {
        Write-Host "[INFO] $ModelName downloaded and extracted." -ForegroundColor Green
    }
    else {
        Write-Host "[ERROR] Model files not found for $ModelName" -ForegroundColor Red
        Get-ChildItem -Path $ModelPath -Recurse -File | ForEach-Object { Write-Host "  $($_.FullName)" }
        exit 1
    }
}

# ==========================================
# Download Server models
# ==========================================
Write-Host "--- Downloading PP-OCRv5 Server Models ---" -ForegroundColor Cyan

# Detection model
Download-And-Extract -ModelName "PP-OCRv5_server_det"

# Recognition model
Download-And-Extract -ModelName "PP-OCRv5_server_rec"

# ==========================================
# Download dictionary files
# ==========================================
Write-Host ""
Write-Host "--- Downloading Dictionary Files ---" -ForegroundColor Cyan

Set-Location $DictDir

# Chinese dictionary
if (-not (Test-Path "ppocr_keys_v1.txt")) {
    Write-Host "[INFO] Downloading Chinese dictionary..." -ForegroundColor Green
    Invoke-WebRequest -Uri "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/release/2.7/ppocr/utils/ppocr_keys_v1.txt" -OutFile "ppocr_keys_v1.txt"
}
else {
    Write-Host "[WARN] Chinese dictionary already exists, skipping..." -ForegroundColor Yellow
}

# English dictionary
if (-not (Test-Path "ic15_dict.txt")) {
    Write-Host "[INFO] Downloading English dictionary..." -ForegroundColor Green
    Invoke-WebRequest -Uri "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/release/2.7/ppocr/utils/ic15_dict.txt" -OutFile "ic15_dict.txt"
}
else {
    Write-Host "[WARN] English dictionary already exists, skipping..." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Download Complete!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Downloaded models:"
Write-Host "  - $PaddleDir\PP-OCRv5_server_det\ (84.3 MB)"
Write-Host "  - $PaddleDir\PP-OCRv5_server_rec\ (81 MB)"
Write-Host ""
Write-Host "Downloaded dictionaries:"
Write-Host "  - $DictDir\ppocr_keys_v1.txt (Chinese, 6623 chars)"
Write-Host "  - $DictDir\ic15_dict.txt (English, 36 classes)"
Write-Host ""
Write-Host "Next step: Convert models to ONNX format"
Write-Host "  Run: .\scripts\convert_to_onnx.ps1"
Write-Host ""

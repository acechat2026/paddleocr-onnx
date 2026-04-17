# setup_deps.ps1
# PaddleOCR ONNX - Dependency setup script (PowerShell)
# Downloads ONNX Runtime, OpenCV, cpp-httplib, and cxxopts for Windows builds.
# Compatible with both interactive desktop and CI environments.

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir
$DepsDir = Join-Path $ProjectDir "3rdparty"

if (-not (Test-Path $DepsDir)) {
    New-Item -ItemType Directory -Force -Path $DepsDir | Out-Null
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "PaddleOCR ONNX - Setup Dependencies" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# ==========================================
# 1. ONNX Runtime
# ==========================================
Write-Host "[1/4] Setting up ONNX Runtime..."

$OnnxVersion = "1.20.1"
$OnnxUrl = "https://github.com/microsoft/onnxruntime/releases/download/v${OnnxVersion}/onnxruntime-win-x64-${OnnxVersion}.zip"
$OnnxZip = Join-Path $DepsDir "onnxruntime.zip"
$OnnxDir = Join-Path $DepsDir "onnxruntime"

if (Test-Path $OnnxDir) {
    Write-Host "  - ONNX Runtime already exists, skipping..." -ForegroundColor Yellow
}
else {
    Write-Host "  - Downloading ONNX Runtime v${OnnxVersion}..."
    try {
        Invoke-WebRequest -Uri $OnnxUrl -OutFile $OnnxZip -ErrorAction Stop
        Write-Host "    Download complete."
    }
    catch {
        Write-Host "    Error downloading ONNX Runtime: $_" -ForegroundColor Red
        exit 1
    }

    Write-Host "  - Extracting..."
    try {
        Expand-Archive -Path $OnnxZip -DestinationPath $DepsDir -Force
        Write-Host "    Extraction complete."
    }
    catch {
        Write-Host "    Error extracting: $_" -ForegroundColor Red
        exit 1
    }

    # Rename extracted folder
    $Extracted = Get-ChildItem -Path $DepsDir -Filter "onnxruntime-win-x64-*" | Select-Object -First 1
    if ($Extracted) {
        Rename-Item -Path $Extracted.FullName -NewName "onnxruntime" -Force
    }

    if (Test-Path $OnnxZip) {
        Remove-Item $OnnxZip -Force
    }

    Write-Host "  - ONNX Runtime setup complete." -ForegroundColor Green
}

# ==========================================
# 2. OpenCV
# ==========================================
Write-Host ""
Write-Host "[2/4] Setting up OpenCV..."

$OpencvVersion = "4.10.0"
$OpencvUrl = "https://github.com/opencv/opencv/releases/download/${OpencvVersion}/opencv-${OpencvVersion}-windows.exe"
$OpencvExe = Join-Path $DepsDir "opencv.exe"
$OpencvDir = Join-Path $DepsDir "opencv"

if (Test-Path $OpencvDir) {
    Write-Host "  - OpenCV already exists, skipping..." -ForegroundColor Yellow
}
else {
    Write-Host "  - Downloading OpenCV v${OpencvVersion}..."
    try {
        Invoke-WebRequest -Uri $OpencvUrl -OutFile $OpencvExe -ErrorAction Stop
        Write-Host "    Download complete."
    }
    catch {
        Write-Host "    Error downloading OpenCV: $_" -ForegroundColor Red
        exit 1
    }

    Write-Host "  - Extracting OpenCV (this may take a while)..."
    Start-Process -FilePath $OpencvExe -ArgumentList "-o`"$DepsDir`"", "-y" -Wait -NoNewWindow

    Start-Sleep -Seconds 2

    if (Test-Path $OpencvExe) {
        Remove-Item $OpencvExe -Force
    }

    Write-Host "  - OpenCV setup complete." -ForegroundColor Green
}

# ==========================================
# 3. cpp-httplib
# ==========================================
Write-Host ""
Write-Host "[3/4] Setting up cpp-httplib..."

$HttplibDir = Join-Path $DepsDir "cpp-httplib"
$HttplibUrl = "https://raw.githubusercontent.com/yhirose/cpp-httplib/v0.14.3/httplib.h"

if (-not (Test-Path $HttplibDir)) {
    New-Item -ItemType Directory -Force -Path $HttplibDir | Out-Null
}

if (Test-Path (Join-Path $HttplibDir "httplib.h")) {
    Write-Host "  - cpp-httplib already exists, skipping..." -ForegroundColor Yellow
}
else {
    Write-Host "  - Downloading cpp-httplib v0.14.3..."
    try {
        Invoke-WebRequest -Uri $HttplibUrl -OutFile (Join-Path $HttplibDir "httplib.h") -ErrorAction Stop
        Write-Host "    Download complete."
    }
    catch {
        Write-Host "    Error downloading cpp-httplib: $_" -ForegroundColor Red
        exit 1
    }
    Write-Host "  - cpp-httplib setup complete." -ForegroundColor Green
}

# ==========================================
# 4. cxxopts
# ==========================================
Write-Host ""
Write-Host "[4/4] Setting up cxxopts..."

$CxxoptsUrl = "https://raw.githubusercontent.com/jarro2783/cxxopts/v3.1.1/include/cxxopts.hpp"
$CxxoptsFile = Join-Path $DepsDir "cxxopts.hpp"

if (Test-Path $CxxoptsFile) {
    Write-Host "  - cxxopts already exists, skipping..." -ForegroundColor Yellow
}
else {
    Write-Host "  - Downloading cxxopts v3.1.1..."
    try {
        Invoke-WebRequest -Uri $CxxoptsUrl -OutFile $CxxoptsFile -ErrorAction Stop
        Write-Host "    Download complete."
    }
    catch {
        Write-Host "    Error downloading cxxopts: $_" -ForegroundColor Red
        exit 1
    }
    Write-Host "  - cxxopts setup complete." -ForegroundColor Green
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Dependencies installed to: $DepsDir"
Write-Host ""
Write-Host "Directory structure:"
Write-Host "  3rdparty/"
Write-Host "  -- onnxruntime/"
Write-Host "  --   -- include/"
Write-Host "  --   -- lib/"
Write-Host "  --   -- bin/"
Write-Host "  -- opencv/"
Write-Host "  --   -- build/"
Write-Host "  -- cpp-httplib/"
Write-Host "  --   -- httplib.h"
Write-Host "  -- cxxopts.hpp"
Write-Host ""
Write-Host "You can now build the project:"
Write-Host "  mkdir build"
Write-Host "  cd build"
Write-Host "  cmake .. -G \"Visual Studio 17 2022\" -A x64"
Write-Host "  cmake --build . --config Release"
Write-Host ""

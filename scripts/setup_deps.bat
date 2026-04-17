@echo off
setlocal enabledelayedexpansion

echo ========================================
echo PaddleOCR ONNX - Setup Dependencies
echo ========================================
echo.

set "SCRIPT_DIR=%~dp0"
set "DEPS_DIR=%SCRIPT_DIR%..\3rdparty"

if not exist "%DEPS_DIR%" mkdir "%DEPS_DIR%"

:: ==========================================
:: 1. ONNX Runtime
:: ==========================================
echo [1/4] Setting up ONNX Runtime...

set "ONNX_VERSION=1.20.1"
set "ONNX_URL=https://github.com/microsoft/onnxruntime/releases/download/v%ONNX_VERSION%/onnxruntime-win-x64-%ONNX_VERSION%.zip"
set "ONNX_ZIP=%DEPS_DIR%\onnxruntime.zip"
set "ONNX_DIR=%DEPS_DIR%\onnxruntime"

if exist "%ONNX_DIR%" (
    echo   - ONNX Runtime already exists, skipping...
    goto :setup_opencv
)

echo   - Downloading ONNX Runtime v%ONNX_VERSION%...
powershell -Command "& {
    try {
        Invoke-WebRequest -Uri '%ONNX_URL%' -OutFile '%ONNX_ZIP%' -ErrorAction Stop
        Write-Host '    Download complete.'
    } catch {
        Write-Host '    Error downloading ONNX Runtime: ' $_.Exception.Message
        exit 1
    }
}"

if %errorlevel% neq 0 (
    echo   - Failed to download ONNX Runtime
    exit /b 1
)

echo   - Extracting...
powershell -Command "& {
    try {
        Expand-Archive -Path '%ONNX_ZIP%' -DestinationPath '%DEPS_DIR%' -Force
        Write-Host '    Extraction complete.'
    } catch {
        Write-Host '    Error extracting: ' $_.Exception.Message
        exit 1
    }
}"

:: Rename extracted folder
for /d %%i in ("%DEPS_DIR%\onnxruntime-win-x64-*") do (
    if exist "%%i" (
        move "%%i" "%ONNX_DIR%" >nul 2>&1
    )
)

:: Clean up zip file
if exist "%ONNX_ZIP%" del "%ONNX_ZIP%"

echo   - ONNX Runtime setup complete.

:setup_opencv
:: ==========================================
:: 2. OpenCV
:: ==========================================
echo.
echo [2/4] Setting up OpenCV...

set "OPENCV_VERSION=4.10.0"
set "OPENCV_URL=https://github.com/opencv/opencv/releases/download/%OPENCV_VERSION%/opencv-%OPENCV_VERSION%-windows.exe"
set "OPENCV_EXE=%DEPS_DIR%\opencv.exe"
set "OPENCV_DIR=%DEPS_DIR%\opencv"

if exist "%OPENCV_DIR%" (
    echo   - OpenCV already exists, skipping...
    goto :setup_httplib
)

echo   - Downloading OpenCV v%OPENCV_VERSION%...
powershell -Command "& {
    try {
        Invoke-WebRequest -Uri '%OPENCV_URL%' -OutFile '%OPENCV_EXE%' -ErrorAction Stop
        Write-Host '    Download complete.'
    } catch {
        Write-Host '    Error downloading OpenCV: ' $_.Exception.Message
        exit 1
    }
}"

if %errorlevel% neq 0 (
    echo   - Failed to download OpenCV
    exit /b 1
)

echo   - Extracting OpenCV (this may take a while)...
"%OPENCV_EXE%" -o"%DEPS_DIR%" -y

:: Wait for extraction to complete
timeout /t 2 /nobreak >nul

:: Clean up exe file
if exist "%OPENCV_EXE%" del "%OPENCV_EXE%"

echo   - OpenCV setup complete.

:setup_httplib
:: ==========================================
:: 3. cpp-httplib
:: ==========================================
echo.
echo [3/4] Setting up cpp-httplib...

set "HTTPLIB_DIR=%DEPS_DIR%\cpp-httplib"
set "HTTPLIB_URL=https://raw.githubusercontent.com/yhirose/cpp-httplib/v0.14.3/httplib.h"

if not exist "%HTTPLIB_DIR%" mkdir "%HTTPLIB_DIR%"

if exist "%HTTPLIB_DIR%\httplib.h" (
    echo   - cpp-httplib already exists, skipping...
    goto :setup_cxxopts
)

echo   - Downloading cpp-httplib v0.14.3...
powershell -Command "& {
    try {
        Invoke-WebRequest -Uri '%HTTPLIB_URL%' -OutFile '%HTTPLIB_DIR%\httplib.h' -ErrorAction Stop
        Write-Host '    Download complete.'
    } catch {
        Write-Host '    Error downloading cpp-httplib: ' $_.Exception.Message
        exit 1
    }
}"

if %errorlevel% neq 0 (
    echo   - Failed to download cpp-httplib
    exit /b 1
)

echo   - cpp-httplib setup complete.

:setup_cxxopts
:: ==========================================
:: 4. cxxopts
:: ==========================================
echo.
echo [4/4] Setting up cxxopts...

set "CXXOPTS_URL=https://raw.githubusercontent.com/jarro2783/cxxopts/v3.1.1/include/cxxopts.hpp"

if exist "%DEPS_DIR%\cxxopts.hpp" (
    echo   - cxxopts already exists, skipping...
    goto :setup_complete
)

echo   - Downloading cxxopts v3.1.1...
powershell -Command "& {
    try {
        Invoke-WebRequest -Uri '%CXXOPTS_URL%' -OutFile '%DEPS_DIR%\cxxopts.hpp' -ErrorAction Stop
        Write-Host '    Download complete.'
    } catch {
        Write-Host '    Error downloading cxxopts: ' $_.Exception.Message
        exit 1
    }
}"

if %errorlevel% neq 0 (
    echo   - Failed to download cxxopts
    exit /b 1
)

echo   - cxxopts setup complete.

:setup_complete
echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Dependencies installed to: %DEPS_DIR%
echo.
echo Directory structure:
echo   3rdparty/
echo   -- onnxruntime/
echo   --   -- include/
echo   --   -- lib/
echo   --   -- bin/
echo   -- opencv/
echo   --   -- build/
echo   -- cpp-httplib/
echo   --   -- httplib.h
echo   -- cxxopts.hpp
echo.
echo You can now build the project:
echo   mkdir build
echo   cd build
echo   cmake .. -G "Visual Studio 17 2022" -A x64
echo   cmake --build . --config Release
echo.

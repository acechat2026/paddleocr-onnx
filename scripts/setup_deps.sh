#!/bin/bash
# PaddleOCR ONNX - Dependency setup script
# Automatically downloads ONNX Runtime, OpenCV, and header-only libraries.
# Supports CI environments with limited permissions (no sudo required for local builds).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPS_DIR="$SCRIPT_DIR/../3rdparty"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
print_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

echo "========================================"
echo "PaddleOCR ONNX - Setup Dependencies"
echo "========================================"
echo ""

mkdir -p "$DEPS_DIR"

OS_TYPE=$(uname -s)
ARCH_TYPE=$(uname -m)

print_info "Detected OS: $OS_TYPE"
print_info "Detected Arch: $ARCH_TYPE"

# ==========================================
# 1. ONNX Runtime
# ==========================================
echo ""
echo "[1/4] Setting up ONNX Runtime..."

ONNX_VERSION="1.20.1"

if [ -d "$DEPS_DIR/onnxruntime" ]; then
    print_warn "ONNX Runtime already exists, skipping..."
else
    cd "$DEPS_DIR"

    if [ "$OS_TYPE" = "Linux" ]; then
        ONNX_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/onnxruntime-linux-x64-${ONNX_VERSION}.tgz"
        ONNX_FILE="onnxruntime-linux-x64-${ONNX_VERSION}.tgz"
    elif [ "$OS_TYPE" = "Darwin" ]; then
        if [ "$ARCH_TYPE" = "arm64" ]; then
            ONNX_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/onnxruntime-osx-arm64-${ONNX_VERSION}.tgz"
            ONNX_FILE="onnxruntime-osx-arm64-${ONNX_VERSION}.tgz"
        else
            ONNX_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/onnxruntime-osx-x86_64-${ONNX_VERSION}.tgz"
            ONNX_FILE="onnxruntime-osx-x86_64-${ONNX_VERSION}.tgz"
        fi
    else
        print_error "Unsupported OS: $OS_TYPE"
        exit 1
    fi

    print_info "Downloading ONNX Runtime v${ONNX_VERSION}..."
    print_info "URL: $ONNX_URL"

    if command -v wget &> /dev/null; then
        wget -q --show-progress "$ONNX_URL" -O "$ONNX_FILE"
    elif command -v curl &> /dev/null; then
        curl -L -# "$ONNX_URL" -o "$ONNX_FILE"
    else
        print_error "Neither wget nor curl found. Please install one of them."
        exit 1
    fi

    print_info "Extracting..."
    tar -xzf "$ONNX_FILE"

    for dir in onnxruntime-*; do
        if [ -d "$dir" ]; then
            mv "$dir" onnxruntime
            break
        fi
    done

    rm "$ONNX_FILE"
    print_info "ONNX Runtime setup complete."
fi

# ==========================================
# 2. OpenCV
# ==========================================
echo ""
echo "[2/4] Setting up OpenCV..."

OPENCV_SETUP=false

if [ "$OS_TYPE" = "Linux" ]; then
    # Try system package manager first
    if pkg-config --exists opencv4 2>/dev/null || pkg-config --exists opencv 2>/dev/null; then
        print_warn "OpenCV already installed system-wide, skipping local setup..."
        OPENCV_SETUP=true
    elif command -v apt-get &> /dev/null; then
        print_info "Attempting to install OpenCV via apt..."
        if command -v sudo &> /dev/null; then
            sudo apt-get update && sudo apt-get install -y libopencv-dev || true
        else
            apt-get update && apt-get install -y libopencv-dev || true
        fi
        if pkg-config --exists opencv4 2>/dev/null || pkg-config --exists opencv 2>/dev/null; then
            OPENCV_SETUP=true
        else
            print_warn "apt installation failed or requires privileges. Will build from source."
        fi
    fi

    if [ "$OPENCV_SETUP" = false ]; then
        print_info "Building minimal OpenCV from source (no sudo required)..."
        OPENCV_VERSION="4.10.0"
        OPENCV_DIR="$DEPS_DIR/opencv"
        if [ ! -d "$OPENCV_DIR" ]; then
            cd "$DEPS_DIR"
            if [ ! -f "opencv-${OPENCV_VERSION}.zip" ]; then
                if command -v wget &> /dev/null; then
                    wget -q "https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip" -O "opencv-${OPENCV_VERSION}.zip"
                else
                    curl -L "https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip" -o "opencv-${OPENCV_VERSION}.zip"
                fi
            fi
            unzip -q "opencv-${OPENCV_VERSION}.zip"
            mv "opencv-${OPENCV_VERSION}" opencv_src
            mkdir -p opencv_src/build
            cd opencv_src/build
            cmake .. \
                -DCMAKE_INSTALL_PREFIX="$OPENCV_DIR" \
                -DCMAKE_BUILD_TYPE=Release \
                -DBUILD_SHARED_LIBS=OFF \
                -DBUILD_LIST=core,imgproc,imgcodecs \
                -DBUILD_EXAMPLES=OFF \
                -DBUILD_DOCS=OFF \
                -DBUILD_TESTS=OFF \
                -DBUILD_PERF_TESTS=OFF \
                -DWITH_WEBP=OFF \
                -DWITH_OPENEXR=OFF \
                -DWITH_TIFF=OFF \
                -DWITH_JASPER=OFF \
                -DWITH_FFMPEG=OFF
            cmake --build . --parallel $(nproc 2>/dev/null || echo 2)
            cmake --install .
            cd "$DEPS_DIR"
            rm -rf opencv_src "opencv-${OPENCV_VERSION}.zip"
            print_info "OpenCV built and installed to $OPENCV_DIR"
        fi
        OPENCV_SETUP=true
    fi
elif [ "$OS_TYPE" = "Darwin" ]; then
    if pkg-config --exists opencv4 2>/dev/null; then
        print_warn "OpenCV already installed system-wide, skipping..."
        OPENCV_SETUP=true
    elif command -v brew &> /dev/null && brew list opencv &> /dev/null; then
        print_warn "OpenCV already installed via Homebrew, skipping..."
        OPENCV_SETUP=true
    else
        print_warn "Homebrew OpenCV not found. Building minimal OpenCV from source..."
        OPENCV_VERSION="4.10.0"
        OPENCV_DIR="$DEPS_DIR/opencv"
        if [ ! -d "$OPENCV_DIR" ]; then
            cd "$DEPS_DIR"
            if [ ! -f "opencv-${OPENCV_VERSION}.zip" ]; then
                if command -v wget &> /dev/null; then
                    wget -q "https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip" -O "opencv-${OPENCV_VERSION}.zip"
                else
                    curl -L "https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip" -o "opencv-${OPENCV_VERSION}.zip"
                fi
            fi
            unzip -q "opencv-${OPENCV_VERSION}.zip"
            mv "opencv-${OPENCV_VERSION}" opencv_src
            mkdir -p opencv_src/build
            cd opencv_src/build
            cmake .. \
                -DCMAKE_INSTALL_PREFIX="$OPENCV_DIR" \
                -DCMAKE_BUILD_TYPE=Release \
                -DBUILD_SHARED_LIBS=OFF \
                -DBUILD_LIST=core,imgproc,imgcodecs \
                -DBUILD_EXAMPLES=OFF \
                -DBUILD_DOCS=OFF \
                -DBUILD_TESTS=OFF \
                -DBUILD_PERF_TESTS=OFF \
                -DWITH_WEBP=OFF \
                -DWITH_OPENEXR=OFF \
                -DWITH_TIFF=OFF \
                -DWITH_JASPER=OFF \
                -DWITH_FFMPEG=OFF
            cmake --build . --parallel $(sysctl -n hw.ncpu 2>/dev/null || echo 2)
            cmake --install .
            cd "$DEPS_DIR"
            rm -rf opencv_src "opencv-${OPENCV_VERSION}.zip"
            print_info "OpenCV built and installed to $OPENCV_DIR"
        fi
        OPENCV_SETUP=true
    fi
else
    print_warn "Please install OpenCV manually on this platform."
fi

# ==========================================
# 3. cpp-httplib
# ==========================================
echo ""
echo "[3/4] Setting up cpp-httplib..."

HTTPLIB_DIR="$DEPS_DIR/cpp-httplib"
HTTPLIB_VERSION="0.14.3"
HTTPLIB_URL="https://raw.githubusercontent.com/yhirose/cpp-httplib/v${HTTPLIB_VERSION}/httplib.h"

mkdir -p "$HTTPLIB_DIR"

if [ -f "$HTTPLIB_DIR/httplib.h" ]; then
    print_warn "cpp-httplib already exists, skipping..."
else
    print_info "Downloading cpp-httplib v${HTTPLIB_VERSION}..."
    if command -v wget &> /dev/null; then
        wget -q "$HTTPLIB_URL" -O "$HTTPLIB_DIR/httplib.h"
    elif command -v curl &> /dev/null; then
        curl -s "$HTTPLIB_URL" -o "$HTTPLIB_DIR/httplib.h"
    else
        print_error "Neither wget nor curl found."
        exit 1
    fi
    print_info "cpp-httplib setup complete."
fi

# ==========================================
# 4. cxxopts
# ==========================================
echo ""
echo "[4/4] Setting up cxxopts..."

CXXOPTS_FILE="$DEPS_DIR/cxxopts.hpp"
CXXOPTS_VERSION="3.1.1"
CXXOPTS_URL="https://raw.githubusercontent.com/jarro2783/cxxopts/v${CXXOPTS_VERSION}/include/cxxopts.hpp"

if [ -f "$CXXOPTS_FILE" ]; then
    print_warn "cxxopts already exists, skipping..."
else
    print_info "Downloading cxxopts v${CXXOPTS_VERSION}..."
    if command -v wget &> /dev/null; then
        wget -q "$CXXOPTS_URL" -O "$CXXOPTS_FILE"
    elif command -v curl &> /dev/null; then
        curl -s "$CXXOPTS_URL" -o "$CXXOPTS_FILE"
    else
        print_error "Neither wget nor curl found."
        exit 1
    fi
    print_info "cxxopts setup complete."
fi

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
print_info "Dependencies installed to: $DEPS_DIR"
echo ""
echo "Directory structure:"
echo "  3rdparty/"
echo "  ├── onnxruntime/"
echo "  │   ├── include/"
echo "  │   ├── lib/"
echo "  │   └── bin/ (or lib/)"
echo "  ├── opencv/ (Linux/macOS: local build if needed)"
echo "  ├── cpp-httplib/"
echo "  │   └── httplib.h"
echo "  └── cxxopts.hpp"
echo ""
echo "You can now build the project:"
echo "  mkdir build && cd build"
echo "  cmake .. -DCMAKE_BUILD_TYPE=Release"
echo "  make -j\$(nproc)"
echo ""

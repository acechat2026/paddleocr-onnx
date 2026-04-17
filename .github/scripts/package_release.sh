#!/bin/bash
# Package release artifacts

set -e

VERSION="${1:-unknown}"
RELEASE_DIR="release-${VERSION}"

echo "Packaging release ${VERSION}..."

# Create release directory
mkdir -p "${RELEASE_DIR}"

# Copy models
if [ -d "models/onnx" ]; then
    mkdir -p "${RELEASE_DIR}/models"
    cp -r models/onnx "${RELEASE_DIR}/models/"
    cp -r models/dicts "${RELEASE_DIR}/models/" 2>/dev/null || true
fi

# Copy binaries based on platform
PLATFORM=$(uname -s)
case "${PLATFORM}" in
    Linux*)
        mkdir -p "${RELEASE_DIR}/bin"
        cp build/bin/ocr_cli "${RELEASE_DIR}/bin/"
        cp build/bin/ocr_server "${RELEASE_DIR}/bin/" 2>/dev/null || true
        tar -czf "paddleocr-onnx-linux-${VERSION}.tar.gz" -C "${RELEASE_DIR}" .
        ;;
    Darwin*)
        mkdir -p "${RELEASE_DIR}/bin"
        cp build/bin/ocr_cli "${RELEASE_DIR}/bin/"
        cp build/bin/ocr_server "${RELEASE_DIR}/bin/" 2>/dev/null || true
        tar -czf "paddleocr-onnx-macos-${VERSION}.tar.gz" -C "${RELEASE_DIR}" .
        ;;
    MINGW*|MSYS*)
        mkdir -p "${RELEASE_DIR}/bin"
        cp build/bin/Release/*.exe "${RELEASE_DIR}/bin/"
        cp build/bin/Release/*.dll "${RELEASE_DIR}/bin/" 2>/dev/null || true
        zip -r "paddleocr-onnx-windows-${VERSION}.zip" "${RELEASE_DIR}"
        ;;
esac

echo "Release packaged successfully"
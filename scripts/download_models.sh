#!/bin/bash
# Download PP-OCRv5 official Paddle models and dictionaries
# Works in CI environments without requiring extra dependencies beyond wget/curl

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR/.."

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
print_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }

MODEL_BASE_URL="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0"
PADDLE_DIR="$PROJECT_DIR/models/paddle"
DICT_DIR="$PROJECT_DIR/models/dicts"

echo "========================================"
echo "Download PP-OCRv5 Official Models"
echo "========================================"
echo ""

print_info "Paddle models will be saved to: $PADDLE_DIR"
print_info "Dictionaries will be saved to: $DICT_DIR"
echo ""

mkdir -p "$PADDLE_DIR"
mkdir -p "$DICT_DIR"

# Download and extract a Paddle model
download_and_extract() {
    local model_name=$1
    local url="${MODEL_BASE_URL}/${model_name}_infer.tar"
    local tar_file="${PADDLE_DIR}/${model_name}.tar"
    local extract_dir="${PADDLE_DIR}/${model_name}"

    if [ -d "$extract_dir" ]; then
        print_warn "$model_name already exists, skipping..."
        return 0
    fi

    print_info "Downloading $model_name..."
    cd "$PADDLE_DIR"

    if command -v wget &> /dev/null; then
        wget -c "$url" -O "$tar_file"
    elif command -v curl &> /dev/null; then
        curl -L -# "$url" -o "$tar_file"
    else
        echo "Error: Neither wget nor curl found."
        exit 1
    fi

    print_info "Extracting $model_name..."
    tar -xf "$tar_file"
    mv "${model_name}_infer" "$model_name"
    rm "$tar_file"

    print_info "$model_name downloaded and extracted."
}

# ==========================================
# Download Server models
# ==========================================
echo "--- Downloading PP-OCRv5 Server Models ---"

# Detection model
download_and_extract "PP-OCRv5_server_det"

# Recognition model
download_and_extract "PP-OCRv5_server_rec"

# Optional: Mobile models (uncomment if needed)
# echo ""
# echo "--- Downloading PP-OCRv5 Mobile Models (Optional) ---"
# download_and_extract "PP-OCRv5_mobile_det"
# download_and_extract "PP-OCRv5_mobile_rec"

# ==========================================
# Download dictionary files
# ==========================================
echo ""
echo "--- Downloading Dictionary Files ---"

cd "$DICT_DIR"

# Chinese dictionary (6623 chars)
if [ ! -f "ppocr_keys_v1.txt" ]; then
    print_info "Downloading Chinese dictionary..."
    if command -v wget &> /dev/null; then
        wget -q "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/release/2.7/ppocr/utils/ppocr_keys_v1.txt"
    else
        curl -s "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/release/2.7/ppocr/utils/ppocr_keys_v1.txt" -o ppocr_keys_v1.txt
    fi
else
    print_warn "Chinese dictionary already exists, skipping..."
fi

# English dictionary (36 classes)
if [ ! -f "ic15_dict.txt" ]; then
    print_info "Downloading English dictionary..."
    if command -v wget &> /dev/null; then
        wget -q "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/release/2.7/ppocr/utils/ic15_dict.txt"
    else
        curl -s "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/release/2.7/ppocr/utils/ic15_dict.txt" -o ic15_dict.txt
    fi
else
    print_warn "English dictionary already exists, skipping..."
fi

echo ""
echo "========================================"
echo "Download Complete!"
echo "========================================"
echo ""
echo "Downloaded models:"
echo "  - $PADDLE_DIR/PP-OCRv5_server_det/ (84.3 MB)"
echo "  - $PADDLE_DIR/PP-OCRv5_server_rec/ (81 MB)"
echo ""
echo "Downloaded dictionaries:"
echo "  - $DICT_DIR/ppocr_keys_v1.txt (Chinese, 6623 chars)"
echo "  - $DICT_DIR/ic15_dict.txt (English, 36 classes)"
echo ""
echo "Next step: Convert models to ONNX format"
echo "  Run: ./scripts/convert_to_onnx.sh"
echo ""

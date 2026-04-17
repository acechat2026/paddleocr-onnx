#!/bin/bash
# Convert Paddle models to ONNX format
# Falls back gracefully if paddle2onnx is not installed (common in CI environments)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR/.."

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
print_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

PADDLE_DIR="$PROJECT_DIR/models/paddle"
ONNX_DIR="$PROJECT_DIR/models/onnx"

echo "========================================"
echo "Convert Paddle Models to ONNX"
echo "========================================"
echo ""

# Check if paddle2onnx is installed
if ! command -v paddle2onnx &> /dev/null; then
    print_warn "paddle2onnx not found. Skipping local conversion."
    echo ""
    echo "You can still obtain ONNX models via:"
    echo "  1. GitHub Actions CI (see .github/workflows/convert-models.yml)"
    echo "  2. Install paddle2onnx locally:"
    echo "     pip install paddle2onnx==1.2.0 paddlepaddle==3.0.0"
    echo ""
    exit 0
fi

print_info "paddle2onnx found: $(paddle2onnx --version 2>&1 | head -1)"

mkdir -p "$ONNX_DIR"

convert_model() {
    local model_name=$1
    local model_type=$2  # det or rec
    local paddle_model_dir="$PADDLE_DIR/$model_name"
    local onnx_file="$ONNX_DIR/${model_name}.onnx"

    if [ ! -d "$paddle_model_dir" ]; then
        print_warn "$model_name not found in $PADDLE_DIR, skipping..."
        return 0
    fi

    if [ -f "$onnx_file" ]; then
        print_warn "$onnx_file already exists, skipping..."
        return 0
    fi

    print_info "Converting $model_name..."

    # Detection and recognition models use different input shapes
    local input_shape=""
    if [ "$model_type" = "det" ]; then
        # Detection model: dynamic batch, 3 channels, dynamic height and width
        input_shape=" -1 3 -1 -1"
    else
        # Recognition model: dynamic batch, 3 channels, height 48, dynamic width
        input_shape=" -1 3 48 -1"
    fi

    # Paddle 3.0 models use inference.json; older models use inference.pdmodel
    local model_file="inference.pdmodel"
    if [ ! -f "$paddle_model_dir/inference.pdmodel" ] && [ -f "$paddle_model_dir/inference.json" ]; then
        model_file="inference.json"
    fi

    paddle2onnx \
        --model_dir "$paddle_model_dir" \
        --model_filename "$model_file" \
        --params_filename inference.pdiparams \
        --save_file "$onnx_file" \
        --opset_version 11 \
        --enable_onnx_checker True \
        --input_shape_dict "x${input_shape}" \
        2>&1 | grep -v "UserWarning" || true

    if [ -f "$onnx_file" ]; then
        local file_size=$(du -h "$onnx_file" | cut -f1)
        print_info "$model_name converted successfully (size: $file_size)"
    else
        print_error "Failed to convert $model_name"
        return 1
    fi
}

echo "--- Converting PP-OCRv5 Server Models ---"
echo ""

# Detection model
convert_model "PP-OCRv5_server_det" "det"

# Recognition model
convert_model "PP-OCRv5_server_rec" "rec"

# Optional: Mobile models
# echo ""
# echo "--- Converting PP-OCRv5 Mobile Models (Optional) ---"
# echo ""
# convert_model "PP-OCRv5_mobile_det" "det"
# convert_model "PP-OCRv5_mobile_rec" "rec"

echo ""
echo "========================================"
echo "Conversion Complete!"
echo "========================================"
echo ""
echo "ONNX models saved to: $ONNX_DIR"
echo ""
ls -lh "$ONNX_DIR"/*.onnx 2>/dev/null || echo "No ONNX models found."
echo ""
echo "You can now run OCR with:"
echo "  ./build/ocr_cli input.jpg -d $ONNX_DIR/PP-OCRv5_server_det.onnx -r $ONNX_DIR/PP-OCRv5_server_rec.onnx --dict models/dicts/ppocr_keys_v1.txt"
echo ""

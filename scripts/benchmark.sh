#!/bin/bash
# OCR performance benchmark script
# Uses awk instead of bc for arithmetic to avoid an extra dependency.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR/.."
BUILD_DIR="$PROJECT_DIR/build"
MODELS_DIR="$PROJECT_DIR/models"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
print_result() { echo -e "${BLUE}[RESULT]${NC} $1"; }

OCR_CLI="$BUILD_DIR/ocr_cli"
if [ ! -f "$OCR_CLI" ]; then
    OCR_CLI="$BUILD_DIR/Release/ocr_cli.exe"
fi

if [ ! -f "$OCR_CLI" ]; then
    echo "Error: ocr_cli not found. Please build the project first."
    echo "  mkdir build && cd build"
    echo "  cmake .. -DCMAKE_BUILD_TYPE=Release"
    echo "  make -j\$(nproc)"
    exit 1
fi

DET_MODEL="$MODELS_DIR/onnx/PP-OCRv5_server_det.onnx"
REC_MODEL="$MODELS_DIR/onnx/PP-OCRv5_server_rec.onnx"
DICT="$MODELS_DIR/dicts/ppocr_keys_v1.txt"
TEST_IMAGES_DIR="$PROJECT_DIR/examples"
OUTPUT_DIR="$PROJECT_DIR/benchmark_results"

mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "PaddleOCR ONNX Benchmark"
echo "========================================"
echo ""

print_info "OCR CLI: $OCR_CLI"
print_info "Detection Model: $DET_MODEL"
print_info "Recognition Model: $REC_MODEL"
print_info "Dictionary: $DICT"
print_info "Test Images: $TEST_IMAGES_DIR"
echo ""

run_benchmark() {
    local test_name=$1
    local image_path=$2
    local extra_args=$3

    echo "--- $test_name ---"

    # Warm-up run
    "$OCR_CLI" "$image_path" \
        -d "$DET_MODEL" \
        -r "$REC_MODEL" \
        --dict "$DICT" \
        $extra_args \
        > /dev/null 2>&1

    # Benchmark (5 runs, average)
    local total_time=0
    local det_time=0
    local rec_time=0
    local runs=5

    for i in $(seq 1 $runs); do
        echo -n "  Run $i/$runs... "

        output=$("$OCR_CLI" "$image_path" \
            -d "$DET_MODEL" \
            -r "$REC_MODEL" \
            --dict "$DICT" \
            $extra_args \
            2>&1)

        t_total=$(echo "$output" | grep -oP "Total time: \K[0-9.]+" || echo "0")
        t_det=$(echo "$output" | grep -oP "Detection: \K[0-9.]+" || echo "0")
        t_rec=$(echo "$output" | grep -oP "Recognition: \K[0-9.]+" || echo "0")

        echo "${t_total}ms"

        total_time=$(awk "BEGIN {print $total_time + $t_total}")
        det_time=$(awk "BEGIN {print $det_time + $t_det}")
        rec_time=$(awk "BEGIN {print $rec_time + $t_rec}")
    done

    local avg_total=$(awk "BEGIN {printf \"%.2f\", $total_time / $runs}")
    local avg_det=$(awk "BEGIN {printf \"%.2f\", $det_time / $runs}")
    local avg_rec=$(awk "BEGIN {printf \"%.2f\", $rec_time / $runs}")

    echo ""
    print_result "$test_name Results (avg of $runs runs):"
    print_result "  Total: ${avg_total} ms"
    print_result "  Detection: ${avg_det} ms"
    print_result "  Recognition: ${avg_rec} ms"
    echo ""

    echo "$test_name,$avg_total,$avg_det,$avg_rec" >> "$OUTPUT_DIR/benchmark.csv"
}

# Initialize CSV
echo "Test,Total(ms),Detection(ms),Recognition(ms)" > "$OUTPUT_DIR/benchmark.csv"

# ==========================================
# CPU benchmark
# ==========================================
echo "========================================="
echo "CPU Benchmark (4 threads)"
echo "========================================="
echo ""

for img in "$TEST_IMAGES_DIR"/*.jpg "$TEST_IMAGES_DIR"/*.png; do
    if [ -f "$img" ]; then
        img_name=$(basename "$img")
        run_benchmark "CPU_$img_name" "$img" "--threads 4"
    fi
done

# ==========================================
# Multi-threading benchmark
# ==========================================
echo "========================================="
echo "Multi-threading Benchmark"
echo "========================================="
echo ""

FIRST_IMG=$(ls "$TEST_IMAGES_DIR"/*.jpg "$TEST_IMAGES_DIR"/*.png 2>/dev/null | head -1)

if [ -f "$FIRST_IMG" ]; then
    run_benchmark "Threads_1" "$FIRST_IMG" "--threads 1"
    run_benchmark "Threads_2" "$FIRST_IMG" "--threads 2"
    run_benchmark "Threads_4" "$FIRST_IMG" "--threads 4"
    run_benchmark "Threads_8" "$FIRST_IMG" "--threads 8"
fi

# ==========================================
# GPU benchmark (if available)
# ==========================================
if "$OCR_CLI" --help 2>&1 | grep -q "\--gpu"; then
    echo "========================================="
    echo "GPU Benchmark"
    echo "========================================="
    echo ""

    if [ -f "$FIRST_IMG" ]; then
        run_benchmark "GPU" "$FIRST_IMG" "--gpu"
    fi
fi

# ==========================================
# Summary
# ==========================================
echo "========================================"
echo "Benchmark Complete!"
echo "========================================"
echo ""
echo "Results saved to: $OUTPUT_DIR/benchmark.csv"
echo ""
echo "Summary:"
echo "--------"

# Pretty-print CSV without requiring the column utility
awk -F',' 'NR==1 {
    printf "%-20s %12s %16s %18s\n", $1, $2, $3, $4
    for (i=0;i<68;i++) printf "-"; printf "\n"
}
NR>1 {
    printf "%-20s %12s %16s %18s\n", $1, $2, $3, $4
}' "$OUTPUT_DIR/benchmark.csv"

echo ""

# Generate Markdown report
REPORT_FILE="$OUTPUT_DIR/benchmark_report.md"

cat > "$REPORT_FILE" << 'EOF'
# PaddleOCR ONNX Benchmark Report

## Test Environment

- **Date**:
- **Model**: PP-OCRv5 Server
- **ONNX Runtime**: 1.20.1

## Results

EOF

echo "| Test | Total (ms) | Detection (ms) | Recognition (ms) |" >> "$REPORT_FILE"
echo "|------|------------|----------------|------------------|" >> "$REPORT_FILE"

tail -n +2 "$OUTPUT_DIR/benchmark.csv" | while IFS=',' read -r test total det rec; do
    echo "| $test | $total | $det | $rec |" >> "$REPORT_FILE"
done

print_info "Report generated: $REPORT_FILE"

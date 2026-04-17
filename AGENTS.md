# Agent Guidelines

## Project
PaddleOCR ONNX Runtime — Cross-platform C++ inference engine for OCR.

## Architecture
- `include/`: Public headers.
- `src/`: Implementation files.
- `scripts/`: Shell/PowerShell helper scripts.
- `.github/workflows/`: CI/CD automation.
- `3rdparty/`: Vendored dependencies (ONNX Runtime, OpenCV, cpp-httplib, cxxopts).

## Before Modifying Code
1. Read the relevant header in `include/` and implementation in `src/`.
2. Check if CI workflows in `.github/workflows/` need updates for the change.
3. Verify script changes remain portable (Linux, macOS, Windows).

## Language Policy
- **Source code comments and script comments: English only.**
- **User-facing documentation (README): Chinese is acceptable.**

## Dependency Policy
- Prefer vendored/header-only libraries over system packages.
- If OpenCV is unavailable on the system, build a minimal static OpenCV locally (`core`, `imgproc`, `imgcodecs`).
- Do not introduce heavy external build dependencies.

## Testing
- There is an industrial test image at `tests/DP_CONSOLE.png`.
- The GitHub Actions `build.yml` runs an end-to-end OCR test using `ocr_cli` on this image.
- When fixing bugs, ensure the CLI and HTTP server paths both compile.

## Common Pitfalls
- `std::vector<std::string>::data()` returns `std::string*`, not `const char* const*`. Always use the dedicated `const char*` pointer vectors for ONNX Runtime `Run()`.
- CTC blank index is **after** the dictionary entries (`dictionary_.size()`).
- Do not use `pause` in `.bat` files intended for CI.
- Do not assume `bc`, `column`, or `brew` are available in scripts.

# Project Instructions for Claude

## Project Overview
This is a cross-platform C++ deployment of PaddleOCR v5 using ONNX Runtime.

## Tech Stack
- C++17
- ONNX Runtime >= 1.20.1
- OpenCV >= 4.0
- CMake >= 3.15
- cpp-httplib (header-only, HTTP server)
- cxxopts (header-only, CLI parsing)

## Coding Conventions
- **All code comments must be in English.** Documentation (README, etc.) can be in Chinese for local users, but source code and script comments are strictly English.
- Use `//` for inline comments, `/* */` for block headers in headers.
- Namespace: `paddleocr::onnx`
- Class naming: `PascalCase` (e.g., `OCRPipeline`)
- File naming: `snake_case.cpp` / `snake_case.h`

## Build & Scripts
- `scripts/setup_deps.sh` / `.bat` / `.ps1`: dependency bootstrap. Must remain CI-friendly (no interactive prompts, no `pause` in CI paths).
- `scripts/download_models.sh` / `.ps1`: download official Paddle models.
- `scripts/convert_to_onnx.sh` / `.ps1`: convert models. Must gracefully skip if `paddle2onnx` is not installed.
- `scripts/benchmark.sh`: performance benchmark. Avoid dependencies like `bc`; use `awk` instead.

## Cross-Platform CI/CD Rules
- Linux/macOS scripts must work without `sudo` where possible. If system OpenCV is missing, fallback to building a minimal OpenCV from source (`core`, `imgproc`, `imgcodecs`).
- Windows batch scripts must not contain `pause` (blocks CI).
- GitHub Actions workflows are in `.github/workflows/`.

## Critical Implementation Details
- ONNX Runtime `Run()` requires `const char* const*` arrays. Maintain parallel `std::vector<const char*>` (`input_name_ptrs_`, `output_name_ptrs_`).
- CTC blank index is `dictionary_.size()` (not `size() - 1`).
- `Ort::Value` read-only access uses `GetTensorData<float>()`.
- Async operations use `std::thread(...).detach()` rather than discarded `std::async`.

## Documentation
- Keep `README.md` accurate. Update project structure diagrams when new files are added.

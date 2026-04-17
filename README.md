# PaddleOCR ONNX Runtime C++ 部署方案

一个轻量级、跨平台的 PaddleOCR C++ 部署方案，基于 ONNX Runtime 实现，支持 CPU/GPU 推理，提供命令行工具和 HTTP 服务两种使用方式。

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)](#)
[![C++](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](#)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.20.1-green.svg)](#)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.10.0-brightgreen.svg)](#)

## 📖 目录

- [特性](#-特性)
- [环境要求](#-环境要求)
- [项目结构](#-项目结构)
- [快速开始](#-快速开始)
  - [Windows](#windows)
  - [Linux](#linux)
  - [macOS](#macos)
- [模型准备](#-模型准备)
- [使用方式](#-使用方式)
  - [命令行工具](#1-命令行工具-ocrcli)
  - [HTTP 服务](#2-http-服务-ocrserver)
- [API 文档](#-api-文档)
- [性能基准](#-性能基准)
- [常见问题](#-常见问题)
- [参考资源](#-参考资源)
- [许可证](#-许可证)

## ✨ 特性

- **轻量级部署**：仅依赖 ONNX Runtime + OpenCV，编译产物约 15-20MB
- **跨平台支持**：Windows 10 / Linux / macOS
- **CPU/GPU 推理**：支持 CUDA 加速（GPU 版本）
- **PP-OCRv5 支持**：完整支持 PP-OCRv5 Server（高精度）和 Mobile（轻量级）版本
- **双模式运行**：
  - 命令行工具：单张/批量图片 OCR 识别
  - HTTP 服务：基于 cpp-httplib 的 RESTful API 服务
- **高性能优化**：支持多线程、内存池、模型量化
- **易于集成**：简洁的 C++ API，CMake 一键构建
- **自包含依赖**：所有第三方库统一管理在 `3rdparty/` 目录，无需系统安装

## 📋 环境要求

| 组件 | 版本要求 | 说明 |
|------|---------|------|
| CMake | ≥ 3.15 | 构建工具 |
| C++ 编译器 | C++17 支持 | MSVC 2022 / GCC 9+ / Clang 10+ |
| OpenCV | ≥ 4.0 | 图像处理（自动下载到 3rdparty/） |
| ONNX Runtime | ≥ 1.20.1 | 推理引擎（自动下载到 3rdparty/） |
| CUDA（可选） | ≥ 11.0 | GPU 加速需要 |
| cpp-httplib | ≥ 0.14.0 | HTTP 服务需要（Header-only） |
| cxxopts | ≥ 3.1.1 | 命令行参数解析（Header-only） |

## 📁 项目结构

```text
paddleocr-onnx/
├── CMakeLists.txt                  # CMake 构建配置
├── README.md                       # 项目文档
├── LICENSE                         # Apache 2.0 许可证
│
├── include/                        # 头文件
│   ├── ocr_common.h                # 公共数据结构
│   ├── ocr_detector.h              # 文本检测器接口
│   ├── ocr_recognizer.h            # 文本识别器接口
│   ├── ocr_pipeline.h              # OCR 流水线接口
│   └── ocr_http_server.h           # HTTP 服务接口
│
├── src/                            # 源代码
│   ├── ocr_cli.cpp                 # 命令行工具入口
│   ├── ocr_server.cpp              # HTTP 服务入口
│   ├── ocr_common.cpp              # 公共工具实现
│   ├── ocr_detector.cpp            # DB 检测算法实现
│   ├── ocr_recognizer.cpp          # CRNN 识别算法实现
│   ├── ocr_pipeline.cpp            # 完整流程串联
│   └── ocr_http_server.cpp         # HTTP 服务实现
│
├── models/                         # 模型目录（CI 自动获取）
│   ├── paddle/                     # Paddle 原始模型
│   ├── onnx/                       # ONNX 转换后模型
│   └── dicts/                      # 字典文件
│
├── examples/                       # 示例图片
├── tests/                          # 测试图片（工业场景）
├── scripts/                        # 辅助脚本
│   ├── setup_deps.sh               # Linux/macOS 依赖配置
│   ├── setup_deps.bat              # Windows 依赖配置 (CMD)
│   ├── setup_deps.ps1              # Windows 依赖配置 (PowerShell)
│   ├── download_models.sh          # 下载 Paddle 模型
│   ├── convert_to_onnx.sh          # 模型转换脚本
│   └── benchmark.sh                # 性能测试脚本
├── .github/workflows/              # CI/CD 工作流
│   ├── build.yml                   # 跨平台编译与测试
│   ├── convert-models.yml          # 模型自动转换
│   └── release.yml                 # Release 自动发布
│
└── 3rdparty/                       # 第三方依赖
    ├── onnxruntime/                # ONNX Runtime
    ├── opencv/                     # OpenCV
    ├── cpp-httplib/                # HTTP 库
    └── cxxopts.hpp                 # 命令行解析

```

## 🚀 快速开始

### Windows 

#### 1. 克隆项目

```powershell
git clone --recursive https://github.com/acechat2026/paddleocr-onnx.git
cd paddleocr-onnx
```

#### 2. 一键配置依赖

通过 CMD 双击运行（无需修改 PowerShell 执行策略）：

```powershell
# 自动下载 ONNX Runtime 和 OpenCV 到 3rdparty/ 目录
.\scripts\setup_deps.bat
```

或者使用 PowerShell（更现代的执行方式）：

```powershell
.\scripts\setup_deps.ps1
```

#### 3. 编译项目

```powershell
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
```

#### 4. 下载模型（见模型准备）

#### 5. 运行测试

```powershell
cd Release
.\ocr_cli.exe ..\..\examples\chinese_test.jpg -d ..\..\models\onnx\PP-OCRv5_server_det.onnx -r ..\..\models\onnx\PP-OCRv5_server_rec.onnx --dict ..\..\models\dicts\ppocr_keys_v1.txt

```

### Linux (Ubuntu/Debian)

#### 1. 克隆项目

```powershell
git clone --recursive https://github.com/acechat2026/paddleocr-onnx.git
cd paddleocr-onnx
```

#### 2. 一键配置依赖


```bash
chmod +x scripts/setup_deps.sh
./scripts/setup_deps.sh
```

#### 3. 编译项目

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

#### 4. 下载模型（见模型准备）

#### 5. 运行测试

```bash
./ocr_cli ../examples/chinese_test.jpg \
    -d ../models/onnx/PP-OCRv5_server_det.onnx \
    -r ../models/onnx/PP-OCRv5_server_rec.onnx \
    --dict ../models/dicts/ppocr_keys_v1.txt
```

### macOS

#### 1. 克隆项目

```powershell
git clone --recursive https://github.com/acechat2026/paddleocr-onnx.git
cd paddleocr-onnx
```

#### 2. 一键配置依赖


```bash
chmod +x scripts/setup_deps.sh
./scripts/setup_deps.sh
```

#### 3. 编译项目

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

#### 4. 下载模型（见模型准备）

#### 5. 运行测试

```bash
./ocr_cli ../examples/chinese_test.jpg \
    -d ../models/onnx/PP-OCRv5_server_det.onnx \
    -r ../models/onnx/PP-OCRv5_server_rec.onnx \
    --dict ../models/dicts/ppocr_keys_v1.txt
```


## 🤖 模型准备

### 模型版本说明

PP-OCRv5 提供两种规格，可根据场景选择：

|版本|检测模型大小|识别模型大小|检测精度 (Hmean)|识别精度|适用场景|
|---|---|---|---|---|---|
|PP-OCRv5 Server|84.3 MB|81 MB|83.8%|86.38%|服务器端，追求最高精度|
|PP-OCRv5 Mobile|4.7 MB|16 MB|79.0%|81.29%|移动端、边缘设备、实时应用|

> ✨ 亮点：PP-OCRv5 Server 版对繁体中文的识别精度可达 93.29%，单模型同时支持简体中文、繁体中文、英文、日文、拼音识别。

### 方式一：使用脚本自动下载和转换（推荐）

```bash
# 下载 Paddle 官方模型
chmod +x scripts/download_models.sh
./scripts/download_models.sh

# 转换为 ONNX 格式（需要安装 paddle2onnx）
pip install paddle2onnx paddlepaddle
chmod +x scripts/convert_to_onnx.sh
./scripts/convert_to_onnx.sh
```

### 方式二：手动下载和转换

#### 1. 下载 Paddle 模型

```bash
# 创建目录
mkdir -p models/paddle models/onnx models/dicts

# 下载 Server 检测模型
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_server_det_infer.tar
tar -xf PP-OCRv5_server_det_infer.tar -C models/paddle/
mv models/paddle/PP-OCRv5_server_det_infer models/paddle/PP-OCRv5_server_det
rm PP-OCRv5_server_det_infer.tar

# 下载 Server 识别模型
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_server_rec_infer.tar
tar -xf PP-OCRv5_server_rec_infer.tar -C models/paddle/
mv models/paddle/PP-OCRv5_server_rec_infer models/paddle/PP-OCRv5_server_rec
rm PP-OCRv5_server_rec_infer.tar

# 下载字典文件
wget https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/release/2.7/ppocr/utils/ppocr_keys_v1.txt -P models/dicts/
wget https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/release/2.7/ppocr/utils/ic15_dict.txt -P models/dicts/
```

#### 2. 转换为 ONNX 格式

```bash
# 安装 paddle2onnx
pip install paddle2onnx paddlepaddle

# 转换检测模型
paddle2onnx \
    --model_dir models/paddle/PP-OCRv5_server_det \
    --model_filename inference.pdmodel \
    --params_filename inference.pdiparams \
    --save_file models/onnx/PP-OCRv5_server_det.onnx \
    --opset_version 11 \
    --enable_onnx_checker True

# 转换识别模型
paddle2onnx \
    --model_dir models/paddle/PP-OCRv5_server_rec \
    --model_filename inference.pdmodel \
    --params_filename inference.pdiparams \
    --save_file models/onnx/PP-OCRv5_server_rec.onnx \
    --opset_version 11 \
    --enable_onnx_checker True
```

### 模型文件清单

|文件|路径|大小|说明|
|---|---|---|---|
|检测模型 (Paddle)|models/paddle/PP-OCRv5_server_det/|84.3 MB|原始格式|
|识别模型 (Paddle)|models/paddle/PP-OCRv5_server_rec/|81 MB	|原始格式|
|检测模型 (ONNX)|models/onnx/PP-OCRv5_server_det.onnx|~84 MB|转换后格式|
|识别模型 (ONNX)|models/onnx/PP-OCRv5_server_rec.onnx|~81 MB|转换后格式|
|中文字典|models/dicts/ppocr_keys_v1.txt|72 KB|6623 字符|
|英文字典|models/dicts/ic15_dict.txt|1.4 KB|36 类|

## 💻 使用方式

### 1. 命令行工具 (ocr_cli)

基本用法

```bash
# 中文图片识别
./ocr_cli input.jpg \
    -d models/onnx/PP-OCRv5_server_det.onnx \
    -r models/onnx/PP-OCRv5_server_rec.onnx \
    --dict models/dicts/ppocr_keys_v1.txt \
    -o output.jpg \
    --json result.json

# 英文图片识别（使用英文模型和字典）
./ocr_cli input.png \
    -d models/onnx/PP-OCRv5_server_det.onnx \
    -r models/onnx/PP-OCRv5_server_rec.onnx \
    --dict models/dicts/ic15_dict.txt \
    -o output.png \
    --json result.json
```

命令行参数

|参数|简写|必需|说明|默认值|
|---|---|---|---|---|
|--detector|-d|是|检测模型路径|-|
|--recognizer|-r|是|识别模型路径|-|
|--dict|-|是|字典文件路径|-|
|--output|-o|否|可视化输出图片路径|-|
|--json|-|否|JSON 结果输出路径|-|
|--gpu|-|否|启用 GPU 推理|false|
|--gpu-id|-|否|GPU 设备 ID|0|
|--threads|-|否|CPU 推理线程数|4|
|--det-thresh|-|否|检测阈值|0.3|
|--det-box-thresh|-|否|检测框阈值|0.6|
|--det-unclip-ratio|-|否|检测框扩张比例|1.5|
|--rec-batch-size|-|否|识别批处理大小|6|
|--help|-h|否|显示帮助信息|-|

输出示例
控制台输出：

```text
=== PaddleOCR ONNX Runtime ===
Image: input.jpg (1920x1080)
Detection Model: PP-OCRv5_server_det.onnx
Recognition Model: PP-OCRv5_server_rec.onnx
GPU: disabled
================================

Initializing models...
Loaded 6623 labels from dictionary
Models initialized in 1066 ms

Running OCR...
[Detector] Input: 1920x1080 -> Model input: 1920x1088
[Detector] Found 156 contours, 23 valid boxes
[Recognizer] Processing 23 text regions...

=== Results (23 text regions found) ===
[ 1] "PaddleOCR ONNX 部署方案" (conf: 0.987) [det: 0.856]
[ 2] "轻量级跨平台 C++ 实现" (conf: 0.995) [det: 0.921]
[ 3] "支持 CPU/GPU 推理" (conf: 0.978) [det: 0.834]

=== Performance ===
Total time: 1250 ms
  - Detection: 420 ms
  - Recognition: 830 ms

Visualization saved to: output.jpg
Results saved to: result.json
```

JSON 输出格式：

```json
{
  "image": "input.jpg",
  "size": [1920, 1080],
  "results": [
    {
      "text": "PaddleOCR ONNX 部署方案",
      "confidence": 0.987,
      "box": [[120, 45], [580, 45], [580, 85], [120, 85]],
      "det_confidence": 0.856
    }
  ],
  "performance": {
    "total_ms": 1250,
    "detection_ms": 420,
    "recognition_ms": 830
  }
}
```

### 2. HTTP 服务 (ocr_server)

启动服务

```bash
# 基本启动（CPU 模式）
./ocr_server \
    -d models/onnx/PP-OCRv5_server_det.onnx \
    -r models/onnx/PP-OCRv5_server_rec.onnx \
    --dict models/dicts/ppocr_keys_v1.txt \
    --port 8080

# GPU 加速启动
./ocr_server \
    -d models/onnx/PP-OCRv5_server_det.onnx \
    -r models/onnx/PP-OCRv5_server_rec.onnx \
    --dict models/dicts/ppocr_keys_v1.txt \
    --gpu \
    --port 8080

# 自定义配置启动
./ocr_server \
    -d models/onnx/PP-OCRv5_server_det.onnx \
    -r models/onnx/PP-OCRv5_server_rec.onnx \
    --dict models/dicts/ppocr_keys_v1.txt \
    --host 0.0.0.0 \
    --port 8080 \
    --threads 4 \
    --rec-batch-size 6
```

服务参数

|参数|简写|必需|说明|默认值|
|---|---|---|---|---|
|--detector|-d|是|检测模型路径|-|
|--recognizer|-r|是|识别模型路径|-|
|--dict|-|是|字典文件路径|-|
|--host|-|否|绑定地址|0.0.0.0|
|--port|-p|否|监听端口|8080|
|--gpu|-|否|启用 GPU 推理|false|
|--gpu-id|-|否|GPU 设备 ID|0|
|--threads|-|否|CPU 推理线程数|4|
|--det-thresh|-|否|检测阈值|0.3|
|--det-box-thresh|-|否|检测框阈值|0.6|
|--rec-batch-size|-|否|识别批处理大小|6|

## API 接口

### 1. 健康检查

```text
GET /health
```

响应：

```json
{
  "status": "ok",
  "models_loaded": true,
  "gpu_enabled": false
}
```

### 2. OCR 识别

```text
POST /ocr
Content-Type: multipart/form-data

file: [图片文件]
```

请求示例 (curl)：

```bash
curl -X POST http://localhost:8080/ocr \
  -F "file=@/path/to/image.jpg"
```

响应：

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "results": [
      {
        "text": "PaddleOCR ONNX 部署方案",
        "confidence": 0.987,
        "box": [[120, 45], [580, 45], [580, 85], [120, 85]]
      }
    ],
    "count": 1,
    "inference_time_ms": 1250
  }
}
```

### 3. Base64 图片识别

```text
POST /ocr/base64
Content-Type: application/json

{
  "image": "base64_encoded_image_string"
}
```

请求示例 (curl)：

```bash
curl -X POST http://localhost:8080/ocr/base64 \
  -H "Content-Type: application/json" \
  -d '{"image": "'$(base64 -w 0 image.jpg)'"}'
```

响应：

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "results": [],
    "count": 5,
    "inference_time_ms": 1250
  }
}
```

### 4. 批量图片识别

```text
POST /ocr/batch
Content-Type: multipart/form-data

files: [多个图片文件]
```

请求示例 (curl)：

```bash
curl -X POST http://localhost:8080/ocr/batch \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "files=@image3.png"
```

响应：

```json
{
  "code": 0,
  "message": "success",
  "data": [
    {
      "filename": "image1.jpg",
      "results": [],
      "inference_time_ms": 1200
    }
  ]
}
```

服务启动日志示例

```text
=== PaddleOCR ONNX HTTP Server ===
Version: 1.0.0
Host: 0.0.0.0
Port: 8080
Detection Model: models/onnx/PP-OCRv5_server_det.onnx
Recognition Model: models/onnx/PP-OCRv5_server_rec.onnx
Dictionary: models/dicts/ppocr_keys_v1.txt (6623 labels)
GPU: disabled
Threads: 4
========================================

Initializing models...
[INFO] Detection model loaded in 520 ms
[INFO] Recognition model loaded in 610 ms
[INFO] Models initialized successfully

Server is running at http://0.0.0.0:8080
Press Ctrl+C to stop...

[2024-01-15 10:23:45] POST /ocr 200 1250ms
[2024-01-15 10:24:12] POST /ocr/base64 200 1180ms
[2024-01-15 10:25:03] GET /health 200 1ms

```

## 📚 API 文档

C++ API 示例

```cpp
#include "ocr_pipeline.h"

int main() {
    // 1. 创建配置
    OCRConfig config;
    config.det_model_path = "models/onnx/PP-OCRv5_server_det.onnx";
    config.rec_model_path = "models/onnx/PP-OCRv5_server_rec.onnx";
    config.dict_path = "models/dicts/ppocr_keys_v1.txt";
    config.use_gpu = false;
    config.num_threads = 4;
    config.det_db_thresh = 0.3f;
    config.det_db_box_thresh = 0.6f;
    
    // 2. 初始化 Pipeline
    OCRPipeline pipeline;
    if (!pipeline.Init(config)) {
        std::cerr << "Failed to initialize OCR pipeline" << std::endl;
        return -1;
    }
    
    // 3. 读取图片
    cv::Mat image = cv::imread("input.jpg");
    
    // 4. 执行 OCR
    std::vector<OCRResult> results = pipeline.Run(image);
    
    // 5. 处理结果
    for (const auto& result : results) {
        std::cout << "Text: " << result.text 
                  << ", Confidence: " << result.confidence << std::endl;
    }
    
    // 6. 可视化
    cv::Mat vis = pipeline.Visualize(image, results);
    cv::imwrite("output.jpg", vis);
    
    return 0;
}
```

高级配置

```cpp
// GPU 配置
config.use_gpu = true;
config.gpu_id = 0;

// 性能调优
config.num_threads = 8;
config.rec_batch_size = 10;

// 检测参数调优（针对不同场景）
config.det_db_thresh = 0.3f;        // 文本框置信度阈值
config.det_db_box_thresh = 0.6f;    // 文本框过滤阈值
config.det_db_unclip_ratio = 1.5f;  // 文本框扩张比例

// 预处理参数
config.det_limit_side_len = 960;    // 检测时图片最大边长
config.rec_image_height = 48;       // 识别时图片高度
```

## 📊 性能基准

### 测试环境

- CPU: Intel Xeon Gold 6271C @ 2.60GHz
- GPU: NVIDIA Tesla T4
- 模型: PP-OCRv5 Server
- 测试图片: 1920×1080，含 15 个文本区域

性能数据

|设备|检测耗时|识别耗时|总耗时|吞吐量 (img/s)|
|---|---|---|---|---|
|CPU (4 线程)|185 ms|52 ms|237 ms|4.2|
|CPU (8 线程)|142 ms|38 ms|180 ms|5.5|
|GPU (CUDA)|70 ms|8.5 ms|78.5 ms|12.7|
|GPU (TensorRT)|45 ms|6 ms|51 ms|19.6

> 注意：实际性能受图片内容、文本数量、硬件配置等因素影响。

模型精度

|模型版本|检测 Hmean|识别精度|繁体中文识别精度|
|---|---|---|---|
|PP-OCRv5 Server|83.8%|86.38%|93.29%|
|PP-OCRv5 Mobile|79.0%|81.29%|88.50%|

## ❓ 常见问题

### Q1: 运行时提示找不到 onnxruntime.dll

解决方案：

- Windows：确保 3rdparty/onnxruntime/bin/onnxruntime.dll 与可执行文件在同一目录
- 或运行 CMake 时会自动复制 DLL 到输出目录

### Q2: CMake 找不到 ONNX Runtime

解决方案：

```bash
# 确认 3rdparty/onnxruntime 目录存在
ls 3rdparty/onnxruntime/

# 如果不存在，运行依赖配置脚本
./scripts/setup_deps.sh       # Linux/macOS
.\scripts\setup_deps.bat      # Windows (CMD)
.\scripts\setup_deps.ps1      # Windows (PowerShell)
```

### Q3: GPU 推理失败

检查项：

1. 确认 CUDA 版本 ≥ 11.0
2. 确认 cuDNN 已正确安装
3. 确认 ONNX Runtime 为 GPU 版本
4. 运行时添加 --gpu 参数

### Q4: 识别结果乱码

解决方案：

- 确保字典文件与模型匹配
- 使用 ppocr_keys_v1.txt 用于中文模型
- 使用 ic15_dict.txt 用于英文模型
- 确保终端编码为 UTF-8

### Q5: 如何提升推理速度？

优化建议：

- 启用 GPU 推理：--gpu
- 增加 CPU 线程数：--threads 8
- 调整识别批处理大小：--rec-batch-size 10
- 使用 Mobile 版本模型（速度优先场景）
- 转换模型为 INT8 量化格式

### Q6: HTTP 服务如何设置并发？

HTTP 服务默认单请求串行处理。如需高并发，建议：

- 使用 Nginx 反向代理 + 多实例部署
- 或修改源码使用线程池处理请求

### Q7: 编译时内存不足

解决方案：

```bash
# 减少并行编译数
make -j2

# 或增加 swap 空间（Linux）
sudo fallocate -l 4G /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

## 🧪 自动化测试

项目包含用于工业软件检查识别的测试图片：

```text
tests/
└── DP_CONSOLE.png          # 工业控制台截图 OCR 测试用例
```

每次推送到 GitHub 后，Actions 会自动运行端到端 OCR 测试：
1. 自动下载并转换 PP-OCRv5 官方模型为 ONNX 格式
2. 在 Linux 环境下编译 `ocr_cli`
3. 使用 `tests/DP_CONSOLE.png` 执行 OCR 识别
4. 输出可视化结果和 JSON 结果作为 artifact 供下载

## 🔁 CI/CD 与模型发布

本项目使用 GitHub Actions 实现完整的持续集成与发布：

### 自动构建与测试 (`build.yml`)
- **触发条件**：push 到 `main`/`develop`、PR、手动触发
- **多平台编译**：Windows (MSVC 2022)、Linux (Ubuntu 22.04)、macOS (Apple Silicon / Intel)
- **OCR 端到端测试**：在 Linux 上自动调用模型测试工业图片

### 模型自动转换 (`convert-models.yml`)
- **触发条件**：手动触发、每月定时检查、workflow 被调用
- **功能**：自动下载 Paddle 官方模型并用 `paddle2onnx` 转换为 ONNX
- **产物**：检测模型、识别模型、中/英文字典的 artifact（保留 90 天）

### 自动发布 (`release.yml`)
- **触发条件**：推送 `v*.*.*` 标签或手动触发
- **发布内容**：
  - `paddleocr-onnx-models-<version>.tar.gz` — ONNX 模型 + 字典
  - `paddleocr-onnx-windows-<version>.zip` — Windows 可执行文件
  - `paddleocr-onnx-linux-<version>.tar.gz` — Linux 可执行文件
  - `paddleocr-onnx-macos-<version>.tar.gz` — macOS 可执行文件

> 💡 **提示**：由于 ONNX 模型文件较大（约 165MB），建议在 GitHub Actions 中完成模型转换，从 Release 或 Artifact 中下载使用，无需在本地安装 `paddlepaddle` 和 `paddle2onnx`。

### macOS 老版本编译注意事项

如果你的 macOS 版本较旧（如 macOS 12），Homebrew 可能无法直接安装 OpenCV，此时可以从源码编译最小化 OpenCV：

```bash
cd 3rdparty
git clone --depth 1 --branch 4.10.0 https://github.com/opencv/opencv.git
cd opencv
mkdir build && cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_LIST=core,imgproc,imgcodecs \
  -DBUILD_SHARED_LIBS=OFF \
  -DBUILD_TESTS=OFF \
  -DBUILD_PERF_TESTS=OFF \
  -DCMAKE_INSTALL_PREFIX=../install
make -j$(sysctl -n hw.ncpu)
make install
```

## 🔗 参考资源

- [PaddleOCR 官方文档](https://github.com/PaddlePaddle/PaddleOCR)
- [PP-OCRv5 模型列表](https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_ch/models_list.md)
- [ONNX Runtime 文档](https://onnxruntime.ai/docs/)
- [OpenCV 文档](https://docs.opencv.org/)
- [Paddle2ONNX 工具](https://github.com/PaddlePaddle/Paddle2ONNX)
- [cpp-httplib 文档](https://github.com/yhirose/cpp-httplib)

## 📄 许可证
本项目采用 Apache License 2.0 许可证。

## 🙏 致谢

- PaddleOCR - 提供优秀的 OCR 算法
- ONNX Runtime - 提供跨平台推理引擎
- cpp-httplib - 提供轻量级 HTTP 库

> ⭐ 如果这个项目对你有帮助，请给一个 Star！






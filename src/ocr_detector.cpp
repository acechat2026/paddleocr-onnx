// Copyright (c) 2024 PaddleOCR ONNX Runtime Project
// Licensed under the Apache License, Version 2.0

#include "ocr_detector.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#ifdef _WIN32
#include <windows.h>
#endif
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

namespace {
#ifdef _WIN32
std::wstring Utf8ToWide(const std::string& utf8) {
    if (utf8.empty()) return std::wstring();
    int size = MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), static_cast<int>(utf8.size()), nullptr, 0);
    std::wstring result(size, 0);
    MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), static_cast<int>(utf8.size()), &result[0], size);
    return result;
}
#endif
} // namespace

namespace paddleocr {
namespace onnx {

// ============================================
// Constructor & Destructor
// ============================================

OCRDetector::OCRDetector() = default;

OCRDetector::~OCRDetector() = default;

OCRDetector::OCRDetector(OCRDetector&& other) noexcept
    : env_(std::move(other.env_))
    , session_options_(std::move(other.session_options_))
    , session_(std::move(other.session_))
    , memory_info_(std::move(other.memory_info_))
    , input_names_(std::move(other.input_names_))
    , output_names_(std::move(other.output_names_))
    , input_shapes_(std::move(other.input_shapes_))
    , output_shapes_(std::move(other.output_shapes_))
    , config_(std::move(other.config_))
    , initialized_(other.initialized_) {
    other.initialized_ = false;
}

OCRDetector& OCRDetector::operator=(OCRDetector&& other) noexcept {
    if (this != &other) {
        env_ = std::move(other.env_);
        session_options_ = std::move(other.session_options_);
        session_ = std::move(other.session_);
        memory_info_ = std::move(other.memory_info_);
        input_names_ = std::move(other.input_names_);
        output_names_ = std::move(other.output_names_);
        input_shapes_ = std::move(other.input_shapes_);
        output_shapes_ = std::move(other.output_shapes_);
        config_ = std::move(other.config_);
        initialized_ = other.initialized_;
        other.initialized_ = false;
    }
    return *this;
}

// ============================================
// Initialization
// ============================================

bool OCRDetector::Init(const DetectorConfig& config) {
    config_ = config;
    
    if (!config_.Validate()) {
        return false;
    }
    
    if (!CreateSession()) {
        return false;
    }
    
    initialized_ = true;
    return true;
}

bool OCRDetector::Init(const OCRConfig& config) {
    DetectorConfig det_config;
    det_config.model_path = config.det_model_path;
    det_config.db_thresh = config.detector.db_thresh;
    det_config.db_box_thresh = config.detector.db_box_thresh;
    det_config.db_unclip_ratio = config.detector.db_unclip_ratio;
    det_config.limit_side_len = config.detector.limit_side_len;
    det_config.batch_size = config.detector.batch_size;
    
    return Init(det_config);
}

bool OCRDetector::CreateSession() {
    try {
        // Create ONNX Runtime environment
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "OCRDetector");
        
        // Create session options
        session_options_ = std::make_unique<Ort::SessionOptions>();
        session_options_->SetIntraOpNumThreads(4);
        session_options_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        // Create session (Windows uses wchar_t paths)
#ifdef _WIN32
        std::wstring model_path_wide = Utf8ToWide(config_.model_path);
        session_ = std::make_unique<Ort::Session>(*env_, model_path_wide.c_str(),
                                                   *session_options_);
#else
        session_ = std::make_unique<Ort::Session>(*env_, config_.model_path.c_str(),
                                                   *session_options_);
#endif
        
        // Get memory info
        memory_info_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        
        // Get input/output names and shapes
        Ort::AllocatorWithDefaultOptions allocator;
        
        size_t num_inputs = session_->GetInputCount();
        input_names_.reserve(num_inputs);
        input_name_ptrs_.reserve(num_inputs);
        input_shapes_.reserve(num_inputs);

        for (size_t i = 0; i < num_inputs; ++i) {
            auto name = session_->GetInputNameAllocated(i, allocator);
            input_names_.push_back(name.get());
            input_name_ptrs_.push_back(input_names_.back().c_str());

            auto type_info = session_->GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            input_shapes_.push_back(tensor_info.GetShape());
        }

        size_t num_outputs = session_->GetOutputCount();
        output_names_.reserve(num_outputs);
        output_name_ptrs_.reserve(num_outputs);
        output_shapes_.reserve(num_outputs);

        for (size_t i = 0; i < num_outputs; ++i) {
            auto name = session_->GetOutputNameAllocated(i, allocator);
            output_names_.push_back(name.get());
            output_name_ptrs_.push_back(output_names_.back().c_str());

            auto type_info = session_->GetOutputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            output_shapes_.push_back(tensor_info.GetShape());
        }
        
        return true;
        
    } catch (const Ort::Exception& e) {
        // Log error: e.what()
        return false;
    }
}

// ============================================
// Configuration
// ============================================

void OCRDetector::SetConfig(const DetectorConfig& config) {
    config_ = config;
}

std::vector<std::pair<std::string, std::vector<int64_t>>> 
OCRDetector::GetInputShapes() const {
    std::vector<std::pair<std::string, std::vector<int64_t>>> result;
    for (size_t i = 0; i < input_names_.size(); ++i) {
        result.emplace_back(input_names_[i], input_shapes_[i]);
    }
    return result;
}

// ============================================
// Detection
// ============================================

std::vector<TextBox> OCRDetector::Detect(const cv::Mat& image) {
    PerfStats stats;
    return Detect(image, stats);
}

std::vector<TextBox> OCRDetector::Detect(const cv::Mat& image, PerfStats& stats) {
    stats.Reset();
    ScopedTimer total_timer(&stats.total_ms);
    
    if (!initialized_ || image.empty()) {
        return {};
    }
    
    original_size_ = image.size();
    
    // Preprocess
    double preprocess_ms = 0;
    std::vector<float> input_data;
    {
        ScopedTimer timer(&preprocess_ms);
        input_data = Preprocess(image);
    }
    stats.preprocess_ms = preprocess_ms;
    
    // Inference
    double inference_ms = 0;
    std::vector<Ort::Value> outputs;
    {
        ScopedTimer timer(&inference_ms);
        outputs = RunInference(input_data);
    }
    stats.inference_ms = inference_ms;
    
    // Postprocess
    double postprocess_ms = 0;
    std::vector<TextBox> boxes;
    {
        ScopedTimer timer(&postprocess_ms);
        boxes = Postprocess(outputs);
    }
    stats.postprocess_ms = postprocess_ms;
    stats.num_text_regions = static_cast<int>(boxes.size());
    
    return boxes;
}

// ============================================
// Preprocessing
// ============================================

std::vector<float> OCRDetector::Preprocess(const cv::Mat& image) {
    // Resize image while keeping aspect ratio
    int src_h = image.rows;
    int src_w = image.cols;
    
    // Calculate resize ratio to limit max side length
    float ratio = 1.0f;
    if (std::max(src_h, src_w) > config_.limit_side_len) {
        ratio = static_cast<float>(config_.limit_side_len) / std::max(src_h, src_w);
    }
    
    int resized_h = static_cast<int>(src_h * ratio);
    int resized_w = static_cast<int>(src_w * ratio);
    
    // Ensure dimensions are multiples of 32 for better performance
    resized_h = std::max(32, (resized_h + 31) / 32 * 32);
    resized_w = std::max(32, (resized_w + 31) / 32 * 32);
    
    scale_x_ = static_cast<float>(src_w) / resized_w;
    scale_y_ = static_cast<float>(src_h) / resized_h;
    
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(resized_w, resized_h));
    
    // Convert to blob with normalization
    std::vector<float> blob = ImageToBlob(resized, config_.mean, config_.scale, true);
    
    // Store padding info for postprocessing
    pad_right_ = resized_w;
    pad_bottom_ = resized_h;
    
    return blob;
}

// ============================================
// Inference
// ============================================

std::vector<Ort::Value> OCRDetector::RunInference(const std::vector<float>& input_data) {
    if (input_shapes_.empty()) {
        return {};
    }
    
    // Get input shape
    auto& shape = input_shapes_[0];
    std::vector<int64_t> actual_shape = shape;
    
    // Update dynamic dimensions
    if (actual_shape.size() == 4) {
        actual_shape[0] = config_.batch_size;
        actual_shape[2] = pad_bottom_;
        actual_shape[3] = pad_right_;
    }
    
    // Create input tensor
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info_,
        const_cast<float*>(input_data.data()),
        input_data.size(),
        actual_shape.data(),
        actual_shape.size()
    );
    
    // Run inference
    std::vector<Ort::Value> output_tensors;
    
    try {
        output_tensors = session_->Run(Ort::RunOptions{nullptr},
                                        input_name_ptrs_.data(),
                                        &input_tensor,
                                        1,
                                        output_name_ptrs_.data(),
                                        output_name_ptrs_.size());
    } catch (const Ort::Exception& e) {
        // Log error
        return {};
    }
    
    return output_tensors;
}

// ============================================
// Postprocessing
// ============================================

std::vector<TextBox> OCRDetector::Postprocess(const std::vector<Ort::Value>& outputs) {
    if (outputs.empty()) {
        return {};
    }
    
    // Get output tensor info
    auto& output = outputs[0];
    auto type_info = output.GetTensorTypeAndShapeInfo();
    auto shape = type_info.GetShape();
    
    if (shape.size() != 4) {
        return {};
    }
    
    int height = static_cast<int>(shape[2]);
    int width = static_cast<int>(shape[3]);
    
    // Get probability map data
    const float* output_data = output.GetTensorData<float>();
    
    // Process first batch item
    cv::Mat prob_map(height, width, CV_32FC1);
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            int idx = h * width + w;
            prob_map.at<float>(h, w) = output_data[idx];
        }
    }
    
    // Extract boxes from bitmap
    return BoxesFromBitmap(prob_map, height, width);
}

std::vector<TextBox> OCRDetector::BoxesFromBitmap(const cv::Mat& prob_map,
                                                    int rows, int cols) {
    // Threshold probability map
    cv::Mat binary;
    cv::threshold(prob_map, binary, config_.db_thresh, 1.0, cv::THRESH_BINARY);
    binary.convertTo(binary, CV_8UC1, 255);
    
    // Find contours
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    
    std::vector<TextBox> boxes;
    
    for (const auto& contour : contours) {
        // Get bounding box
        cv::RotatedRect rect = cv::minAreaRect(contour);
        
        // Filter small boxes
        float min_side = std::min(rect.size.width, rect.size.height);
        if (min_side < 3) {
            continue;
        }
        
        // Get four corners
        std::vector<cv::Point2f> box_points(4);
        rect.points(box_points.data());
        
        // Convert to integer points
        std::vector<cv::Point> points;
        for (const auto& pt : box_points) {
            points.emplace_back(static_cast<int>(pt.x), static_cast<int>(pt.y));
        }
        
        // Sort points clockwise
        std::sort(points.begin(), points.end(),
                  [](const cv::Point& a, const cv::Point& b) {
                      return a.y < b.y;
                  });
        
        if (points[0].x > points[1].x) std::swap(points[0], points[1]);
        if (points[2].x < points[3].x) std::swap(points[2], points[3]);
        
        // Unclip box
        points = UnclipBox(points, config_.db_unclip_ratio);
        
        // Clip to image bounds
        for (auto& pt : points) {
            pt.x = Clip(pt.x, 0, cols - 1);
            pt.y = Clip(pt.y, 0, rows - 1);
        }
        
        // Calculate detection confidence from probability map
        cv::Rect bbox = cv::boundingRect(points);
        bbox &= cv::Rect(0, 0, cols, rows);
        
        if (bbox.area() <= 0) {
            continue;
        }
        
        cv::Mat roi = prob_map(bbox);
        double min_val, max_val;
        cv::minMaxLoc(roi, &min_val, &max_val);
        
        // Filter low confidence boxes
        if (max_val < config_.db_box_thresh) {
            continue;
        }
        
        // Scale points back to original image size
        for (auto& pt : points) {
            pt.x = static_cast<int>(pt.x * scale_x_);
            pt.y = static_cast<int>(pt.y * scale_y_);
        }
        
        TextBox box;
        box.points = points;
        box.confidence = static_cast<float>(max_val);
        box.SortPoints();
        
        boxes.push_back(box);
    }
    
    return boxes;
}

std::vector<cv::Point> OCRDetector::UnclipBox(const std::vector<cv::Point>& box,
                                                float unclip_ratio) {
    if (box.size() != 4) {
        return box;
    }
    
    float area = BoxArea(box);
    if (area <= 0) {
        return box;
    }
    
    float length = 0.0f;
    for (size_t i = 0; i < 4; ++i) {
        length += Distance(box[i], box[(i + 1) % 4]);
    }
    
    float distance = area * unclip_ratio / length;
    
    std::vector<cv::Point> expanded(4);
    for (size_t i = 0; i < 4; ++i) {
        cv::Point p0 = box[i];
        cv::Point p1 = box[(i + 1) % 4];
        cv::Point p2 = box[(i + 2) % 4];
        
        float dx1 = static_cast<float>(p0.x - p1.x);
        float dy1 = static_cast<float>(p0.y - p1.y);
        float len1 = std::sqrt(dx1 * dx1 + dy1 * dy1);
        dx1 /= len1;
        dy1 /= len1;
        
        float dx2 = static_cast<float>(p2.x - p1.x);
        float dy2 = static_cast<float>(p2.y - p1.y);
        float len2 = std::sqrt(dx2 * dx2 + dy2 * dy2);
        dx2 /= len2;
        dy2 /= len2;
        
        float dx = dx1 + dx2;
        float dy = dy1 + dy2;
        float len = std::sqrt(dx * dx + dy * dy);
        
        if (len > 1e-6f) {
            dx /= len;
            dy /= len;
        }
        
        expanded[i].x = static_cast<int>(p1.x + dx * distance);
        expanded[i].y = static_cast<int>(p1.y + dy * distance);
    }
    
    return expanded;
}

float OCRDetector::Distance(const cv::Point& a, const cv::Point& b) {
    float dx = static_cast<float>(a.x - b.x);
    float dy = static_cast<float>(a.y - b.y);
    return std::sqrt(dx * dx + dy * dy);
}

float OCRDetector::BoxArea(const std::vector<cv::Point>& box) {
    if (box.size() < 3) return 0.0f;
    
    float area = 0.0f;
    int n = static_cast<int>(box.size());
    for (int i = 0; i < n; ++i) {
        int j = (i + 1) % n;
        area += static_cast<float>(box[i].x * box[j].y - box[j].x * box[i].y);
    }
    return std::abs(area) * 0.5f;
}

}  // namespace onnx
}  // namespace paddleocr
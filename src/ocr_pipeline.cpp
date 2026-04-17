// Copyright (c) 2024 PaddleOCR ONNX Runtime Project
// Licensed under the Apache License, Version 2.0

#include "ocr_pipeline.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <future>
#include <iomanip>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

namespace paddleocr {
namespace onnx {

// ============================================
// Constructor & Destructor
// ============================================

OCRPipeline::OCRPipeline() = default;

OCRPipeline::~OCRPipeline() = default;

OCRPipeline::OCRPipeline(OCRPipeline&& other) noexcept
    : config_(std::move(other.config_))
    , detector_(std::move(other.detector_))
    , recognizer_(std::move(other.recognizer_))
    , box_color_(other.box_color_)
    , text_color_(other.text_color_)
    , thickness_(other.thickness_)
    , font_scale_(other.font_scale_) {}

OCRPipeline& OCRPipeline::operator=(OCRPipeline&& other) noexcept {
    if (this != &other) {
        config_ = std::move(other.config_);
        detector_ = std::move(other.detector_);
        recognizer_ = std::move(other.recognizer_);
        box_color_ = other.box_color_;
        text_color_ = other.text_color_;
        thickness_ = other.thickness_;
        font_scale_ = other.font_scale_;
    }
    return *this;
}

// ============================================
// Initialization
// ============================================

bool OCRPipeline::Init(const OCRConfig& config) {
    config_ = config;

    if (!config_.Validate()) {
        std::cerr << "OCRPipeline::Init error: OCRConfig validation failed" << std::endl;
        return false;
    }

    // Initialize detector
    if (!detector_.Init(config_)) {
        std::cerr << "OCRPipeline::Init error: detector initialization failed" << std::endl;
        return false;
    }

    // Initialize recognizer
    if (!recognizer_.Init(config_)) {
        std::cerr << "OCRPipeline::Init error: recognizer initialization failed" << std::endl;
        return false;
    }

    return true;
}

bool OCRPipeline::Init(const DetectorConfig& det_config,
                        const RecognizerConfig& rec_config) {
    config_.detector = det_config;
    config_.recognizer = rec_config;
    config_.det_model_path = det_config.model_path;
    config_.rec_model_path = rec_config.model_path;
    config_.dict_path = rec_config.dict_path;

    if (!detector_.Init(det_config)) {
        std::cerr << "OCRPipeline::Init error: detector initialization failed" << std::endl;
        return false;
    }

    if (!recognizer_.Init(rec_config)) {
        std::cerr << "OCRPipeline::Init error: recognizer initialization failed" << std::endl;
        return false;
    }

    return true;
}

bool OCRPipeline::IsInitialized() const {
    return detector_.IsInitialized() && recognizer_.IsInitialized();
}

// ============================================
// Configuration
// ============================================

void OCRPipeline::SetConfig(const OCRConfig& config) {
    config_ = config;
    detector_.SetConfig(config_.detector);
    recognizer_.SetConfig(config_.recognizer);
}

void OCRPipeline::SetVisualizationStyle(cv::Scalar box_color,
                                         cv::Scalar text_color,
                                         int thickness,
                                         double font_scale) {
    box_color_ = box_color;
    text_color_ = text_color;
    thickness_ = thickness;
    font_scale_ = font_scale;
}

// ============================================
// OCR Pipeline
// ============================================

std::vector<OCRResult> OCRPipeline::Run(const cv::Mat& image) {
    PerfStats stats;
    return Run(image, stats);
}

std::vector<OCRResult> OCRPipeline::Run(const cv::Mat& image, PerfStats& stats) {
    stats.Reset();
    
    if (!IsInitialized() || image.empty()) {
        return {};
    }
    
    ScopedTimer total_timer(&stats.total_ms);
    
    // Step 1: Detection
    PerfStats det_stats;
    std::vector<TextBox> boxes = detector_.Detect(image, det_stats);
    last_det_stats_ = det_stats;
    stats.preprocess_ms = det_stats.preprocess_ms;
    stats.inference_ms = det_stats.inference_ms;
    stats.postprocess_ms = det_stats.postprocess_ms;

    if (boxes.empty()) {
        last_rec_stats_ = PerfStats();
        return {};
    }

    // Step 2: Sort boxes by reading order
    boxes = SortByReadingOrder(boxes);

    // Step 3: Filter overlapping boxes
    boxes = FilterOverlappingBoxes(boxes);

    // Step 4: Crop text regions
    std::vector<cv::Mat> cropped_images = CropRegions(image, boxes);

    // Step 5: Recognition
    PerfStats rec_stats;
    auto rec_results = recognizer_.RecognizeBatch(cropped_images, rec_stats);
    last_rec_stats_ = rec_stats;
    stats.inference_ms += rec_stats.inference_ms;
    
    // Step 6: Combine results
    std::vector<OCRResult> results;
    results.reserve(boxes.size());
    
    for (size_t i = 0; i < boxes.size() && i < rec_results.size(); ++i) {
        OCRResult result;
        result.box = boxes[i];
        result.det_confidence = boxes[i].confidence;
        result.text = rec_results[i].first;
        result.confidence = rec_results[i].second;
        results.push_back(result);
    }
    
    stats.num_text_regions = static_cast<int>(results.size());
    
    return results;
}

void OCRPipeline::RunAsync(const cv::Mat& image,
                            std::function<void(std::vector<OCRResult>)> callback) {
    std::thread([this, image, callback]() {
        auto results = Run(image);
        if (callback) {
            callback(results);
        }
    }).detach();
}

// ============================================
// Component Operations
// ============================================

std::vector<TextBox> OCRPipeline::DetectOnly(const cv::Mat& image) {
    if (!detector_.IsInitialized()) {
        return {};
    }
    return detector_.Detect(image);
}

std::vector<std::pair<std::string, float>> OCRPipeline::RecognizeOnly(
    const std::vector<cv::Mat>& images) {
    if (!recognizer_.IsInitialized()) {
        return {};
    }
    return recognizer_.RecognizeBatch(images);
}

// ============================================
// Visualization
// ============================================

cv::Mat OCRPipeline::Visualize(const cv::Mat& image,
                                const std::vector<OCRResult>& results) const {
    cv::Mat vis = image.clone();
    
    for (const auto& result : results) {
        const auto& points = result.box.points;
        
        // Draw bounding box
        if (points.size() == 4) {
            for (int i = 0; i < 4; ++i) {
                cv::line(vis, points[i], points[(i + 1) % 4], box_color_, thickness_);
            }
        }
        
        // Draw text label
        std::string label = result.text;
        if (result.confidence > 0) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(2) << result.confidence;
            label += " (" + oss.str() + ")";
        }
        
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX,
                                              font_scale_, thickness_, &baseline);
        
        cv::Point text_pos(points[0].x, points[0].y - 5);
        if (text_pos.y - text_size.height < 0) {
            text_pos.y = points[2].y + text_size.height + 5;
        }
        
        // Draw text background
        cv::rectangle(vis,
                      cv::Point(text_pos.x, text_pos.y - text_size.height),
                      cv::Point(text_pos.x + text_size.width, text_pos.y + baseline),
                      box_color_, cv::FILLED);
        
        // Draw text
        cv::putText(vis, label, text_pos, cv::FONT_HERSHEY_SIMPLEX,
                    font_scale_, text_color_, thickness_);
    }
    
    return vis;
}

// ============================================
// Export
// ============================================

std::string OCRPipeline::ExportToJson(const std::vector<OCRResult>& results,
                                       const std::string& image_path,
                                       const cv::Size& image_size,
                                       const PerfStats& stats) const {
    std::ostringstream oss;
    oss << "{\n";
    oss << "  \"image\": \"" << image_path << "\",\n";
    oss << "  \"size\": [" << image_size.width << ", " << image_size.height << "],\n";
    oss << "  \"results\": [\n";

    for (size_t i = 0; i < results.size(); ++i) {
        oss << "    " << results[i].ToJson();
        if (i < results.size() - 1) {
            oss << ",";
        }
        oss << "\n";
    }

    oss << "  ],\n";
    oss << "  \"performance\": {\n";
    oss << "    \"total_ms\": " << stats.total_ms << ",\n";
    oss << "    \"detection_ms\": " << last_det_stats_.inference_ms << ",\n";
    oss << "    \"recognition_ms\": " << last_rec_stats_.inference_ms << "\n";
    oss << "  }\n";
    oss << "}\n";

    return oss.str();
}

// ============================================
// Helper Methods
// ============================================

std::vector<cv::Mat> OCRPipeline::CropRegions(const cv::Mat& image,
                                                const std::vector<TextBox>& boxes) {
    std::vector<cv::Mat> crops;
    crops.reserve(boxes.size());
    
    for (const auto& box : boxes) {
        cv::Rect rect = box.GetBoundingRect();
        
        // Expand rect slightly for better recognition
        int padding = 2;
        rect.x = std::max(0, rect.x - padding);
        rect.y = std::max(0, rect.y - padding);
        rect.width = std::min(image.cols - rect.x, rect.width + 2 * padding);
        rect.height = std::min(image.rows - rect.y, rect.height + 2 * padding);
        
        if (rect.width > 0 && rect.height > 0) {
            crops.push_back(image(rect).clone());
        }
    }
    
    return crops;
}

std::vector<TextBox> OCRPipeline::SortByReadingOrder(const std::vector<TextBox>& boxes) {
    if (boxes.size() <= 1) {
        return boxes;
    }
    
    std::vector<TextBox> sorted = boxes;
    
    // Sort primarily by y-coordinate, then by x-coordinate
    std::sort(sorted.begin(), sorted.end(),
              [](const TextBox& a, const TextBox& b) {
                  cv::Rect rect_a = a.GetBoundingRect();
                  cv::Rect rect_b = b.GetBoundingRect();
                  
                  // If vertical overlap is significant, sort by x
                  int overlap_y = std::max(0, std::min(rect_a.y + rect_a.height,
                                                       rect_b.y + rect_b.height) -
                                            std::max(rect_a.y, rect_b.y));
                  int min_height = std::min(rect_a.height, rect_b.height);
                  
                  if (overlap_y > min_height / 2) {
                      return rect_a.x < rect_b.x;
                  }
                  
                  return rect_a.y < rect_b.y;
              });
    
    return sorted;
}

std::vector<TextBox> OCRPipeline::FilterOverlappingBoxes(const std::vector<TextBox>& boxes,
                                                           float iou_threshold) {
    std::vector<TextBox> filtered;
    std::vector<bool> keep(boxes.size(), true);
    
    for (size_t i = 0; i < boxes.size(); ++i) {
        if (!keep[i]) continue;
        
        for (size_t j = i + 1; j < boxes.size(); ++j) {
            if (!keep[j]) continue;
            
            float iou = CalculateIOU(boxes[i], boxes[j]);
            if (iou > iou_threshold) {
                // Keep the one with higher confidence
                if (boxes[i].confidence >= boxes[j].confidence) {
                    keep[j] = false;
                } else {
                    keep[i] = false;
                    break;
                }
            }
        }
    }
    
    for (size_t i = 0; i < boxes.size(); ++i) {
        if (keep[i]) {
            filtered.push_back(boxes[i]);
        }
    }
    
    return filtered;
}

float OCRPipeline::CalculateIOU(const TextBox& a, const TextBox& b) const {
    cv::Rect rect_a = a.GetBoundingRect();
    cv::Rect rect_b = b.GetBoundingRect();
    
    cv::Rect intersection = rect_a & rect_b;
    cv::Rect union_rect = rect_a | rect_b;
    
    if (union_rect.area() == 0) {
        return 0.0f;
    }
    
    return static_cast<float>(intersection.area()) / union_rect.area();
}

}  // namespace onnx
}  // namespace paddleocr
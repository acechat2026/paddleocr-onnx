// Copyright (c) 2024 PaddleOCR ONNX Runtime Project
// Licensed under the Apache License, Version 2.0

#include "ocr_common.h"

#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <cctype>
#include <opencv2/imgproc.hpp>

namespace paddleocr {
namespace onnx {

// ============================================
// TextBox implementation
// ============================================

cv::Rect TextBox::GetBoundingRect() const {
    if (points.size() < 4) {
        return cv::Rect();
    }
    
    int min_x = std::min({points[0].x, points[1].x, points[2].x, points[3].x});
    int min_y = std::min({points[0].y, points[1].y, points[2].y, points[3].y});
    int max_x = std::max({points[0].x, points[1].x, points[2].x, points[3].x});
    int max_y = std::max({points[0].y, points[1].y, points[2].y, points[3].y});
    
    return cv::Rect(min_x, min_y, max_x - min_x, max_y - min_y);
}

void TextBox::SortPoints() {
    if (points.size() != 4) return;
    
    // Sort by y-coordinate
    std::sort(points.begin(), points.end(),
              [](const cv::Point& a, const cv::Point& b) { return a.y < b.y; });
    
    // Top two points: sort by x
    if (points[0].x > points[1].x) {
        std::swap(points[0], points[1]);
    }
    // Bottom two points: sort by x (descending for clockwise order)
    if (points[2].x < points[3].x) {
        std::swap(points[2], points[3]);
    }
}

// ============================================
// OCRResult implementation
// ============================================

std::string OCRResult::ToJson() const {
    std::ostringstream oss;
    oss << "{";
    oss << "\"text\":\"" << text << "\",";
    oss << "\"confidence\":" << confidence << ",";
    oss << "\"box\":[";
    for (size_t i = 0; i < box.points.size(); ++i) {
        if (i > 0) oss << ",";
        oss << "[" << box.points[i].x << "," << box.points[i].y << "]";
    }
    oss << "],";
    oss << "\"det_confidence\":" << det_confidence;
    oss << "}";
    return oss.str();
}

// ============================================
// PerfStats implementation
// ============================================

std::string PerfStats::ToString() const {
    std::ostringstream oss;
    oss << "Total: " << total_ms << " ms, ";
    oss << "Preprocess: " << preprocess_ms << " ms, ";
    oss << "Inference: " << inference_ms << " ms, ";
    oss << "Postprocess: " << postprocess_ms << " ms, ";
    oss << "Regions: " << num_text_regions;
    return oss.str();
}

// ============================================
// DetectorConfig implementation
// ============================================

bool DetectorConfig::Validate() const {
    if (model_path.empty()) {
        return false;
    }
    if (db_thresh < 0.0f || db_thresh > 1.0f) {
        return false;
    }
    if (db_box_thresh < 0.0f || db_box_thresh > 1.0f) {
        return false;
    }
    if (limit_side_len < 32 || limit_side_len > 4096) {
        return false;
    }
    if (batch_size < 1) {
        return false;
    }
    return true;
}

// ============================================
// RecognizerConfig implementation
// ============================================

bool RecognizerConfig::Validate() const {
    if (model_path.empty()) {
        return false;
    }
    if (dict_path.empty()) {
        return false;
    }
    if (image_height < 8 || image_height > 128) {
        return false;
    }
    if (batch_size < 1) {
        return false;
    }
    if (drop_score < 0.0f || drop_score > 1.0f) {
        return false;
    }
    return true;
}

// ============================================
// OCRConfig implementation
// ============================================

OCRConfig OCRConfig::FromPaths(const std::string& det_path,
                                const std::string& rec_path,
                                const std::string& dict) {
    OCRConfig config;
    config.det_model_path = det_path;
    config.rec_model_path = rec_path;
    config.dict_path = dict;
    config.detector.model_path = det_path;
    config.recognizer.model_path = rec_path;
    config.recognizer.dict_path = dict;
    return config;
}

bool OCRConfig::Validate() const {
    if (det_model_path.empty() || rec_model_path.empty() || dict_path.empty()) {
        return false;
    }
    if (!detector.Validate()) {
        return false;
    }
    if (!recognizer.Validate()) {
        return false;
    }
    if (num_threads < 1) {
        return false;
    }
    return true;
}

// ============================================
// Utility functions
// ============================================

std::vector<std::string> LoadDictionary(const std::string& dict_path) {
    std::vector<std::string> dictionary;
    std::ifstream file(dict_path);
    
    if (!file.is_open()) {
        return dictionary;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        // Remove trailing carriage return
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (!line.empty()) {
            dictionary.push_back(line);
        }
    }
    
    file.close();
    return dictionary;
}

std::vector<float> ImageToBlob(const cv::Mat& image,
                                const float mean[3],
                                const float scale[3],
                                bool swap_rb) {
    cv::Mat float_img;
    image.convertTo(float_img, CV_32FC3, 1.0 / 255.0);
    
    std::vector<float> blob;
    int channels = 3;
    int height = float_img.rows;
    int width = float_img.cols;
    blob.resize(channels * height * width);
    
    for (int c = 0; c < channels; ++c) {
        int channel_idx = swap_rb ? (2 - c) : c;
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                float pixel = float_img.at<cv::Vec3f>(h, w)[channel_idx];
                pixel = (pixel - mean[c]) / scale[c];
                blob[c * height * width + h * width + w] = pixel;
            }
        }
    }
    
    return blob;
}

std::pair<cv::Mat, float> ResizeWithAspectRatio(const cv::Mat& image,
                                                 const cv::Size& target_size,
                                                 bool keep_ratio) {
    if (!keep_ratio) {
        cv::Mat resized;
        cv::resize(image, resized, target_size);
        return {resized, 1.0f};
    }
    
    int src_h = image.rows;
    int src_w = image.cols;
    int dst_h = target_size.height;
    int dst_w = target_size.width;
    
    float ratio = std::min(static_cast<float>(dst_w) / src_w,
                           static_cast<float>(dst_h) / src_h);
    
    int new_w = static_cast<int>(src_w * ratio);
    int new_h = static_cast<int>(src_h * ratio);
    
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(new_w, new_h));
    
    return {resized, 1.0f / ratio};
}

std::string BoxToJson(const std::vector<cv::Point>& box) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < box.size(); ++i) {
        if (i > 0) oss << ",";
        oss << "[" << box[i].x << "," << box[i].y << "]";
    }
    oss << "]";
    return oss.str();
}

}  // namespace onnx
}  // namespace paddleocr
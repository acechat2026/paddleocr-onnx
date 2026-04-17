// Copyright (c) 2024 PaddleOCR ONNX Runtime Project
// Licensed under the Apache License, Version 2.0

#ifndef PADDLEOCR_ONNX_INCLUDE_OCR_COMMON_H_
#define PADDLEOCR_ONNX_INCLUDE_OCR_COMMON_H_

#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <opencv2/core.hpp>

namespace paddleocr {
namespace onnx {

// ============================================
// Basic data structures
// ============================================

/**
 * @brief Bounding box for text detection.
 * Points are ordered clockwise starting from top-left.
 */
struct TextBox {
    std::vector<cv::Point> points;  // 4 points for quadrilateral
    float confidence = 0.0f;        // Detection confidence score
    
    /**
     * @brief Get the minimum bounding rectangle.
     * @return cv::Rect representing the bounding box.
     */
    cv::Rect GetBoundingRect() const;
    
    /**
     * @brief Get the four corners as vector of points.
     * @return Vector of 4 points.
     */
    std::vector<cv::Point> GetPoints() const { return points; }
    
    /**
     * @brief Sort points in clockwise order starting from top-left.
     */
    void SortPoints();
};

/**
 * @brief OCR result for a single text region.
 */
struct OCRResult {
    std::string text;               // Recognized text content
    float confidence = 0.0f;        // Recognition confidence score
    TextBox box;                    // Text bounding box
    float det_confidence = 0.0f;    // Detection confidence score
    
    // Convert to JSON string
    std::string ToJson() const;
};

/**
 * @brief Performance statistics.
 */
struct PerfStats {
    double preprocess_ms = 0.0;     // Preprocessing time
    double inference_ms = 0.0;      // Model inference time
    double postprocess_ms = 0.0;    // Postprocessing time
    double total_ms = 0.0;          // Total processing time
    int num_text_regions = 0;       // Number of detected text regions
    
    void Reset() {
        preprocess_ms = 0.0;
        inference_ms = 0.0;
        postprocess_ms = 0.0;
        total_ms = 0.0;
        num_text_regions = 0;
    }
    
    std::string ToString() const;
};

// ============================================
// Configuration structures
// ============================================

/**
 * @brief Detection model configuration.
 */
struct DetectorConfig {
    std::string model_path;         // Path to ONNX model file
    
    // DB algorithm parameters
    float db_thresh = 0.3f;         // Threshold for probability map
    float db_box_thresh = 0.6f;     // Threshold for bounding box
    float db_unclip_ratio = 1.5f;   // Ratio for unclipping box
    bool use_dilation = false;      // Use dilation in post-processing
    
    // Preprocessing parameters
    int limit_side_len = 960;       // Maximum side length for resizing
    float mean[3] = {0.485f, 0.456f, 0.406f};   // Normalization mean
    float scale[3] = {0.229f, 0.224f, 0.225f};   // Normalization std
    
    // Runtime parameters
    int batch_size = 1;             // Batch size for inference
    
    /**
     * @brief Validate configuration.
     * @return true if valid, false otherwise.
     */
    bool Validate() const;
};

/**
 * @brief Recognition model configuration.
 */
struct RecognizerConfig {
    std::string model_path;         // Path to ONNX model file
    std::string dict_path;          // Path to dictionary file
    
    // Preprocessing parameters
    int image_height = 48;          // Target image height
    int image_width = 320;          // Maximum image width (dynamic)
    float mean[3] = {0.5f, 0.5f, 0.5f};    // Normalization mean
    float scale[3] = {0.5f, 0.5f, 0.5f};    // Normalization std
    bool keep_ratio = true;         // Keep aspect ratio when resizing
    
    // Runtime parameters
    int batch_size = 6;             // Batch size for inference
    int max_text_length = 25;       // Maximum text length to decode
    
    // Post-processing parameters
    float drop_score = 0.5f;        // Threshold for dropping low-confidence chars
    
    /**
     * @brief Validate configuration.
     * @return true if valid, false otherwise.
     */
    bool Validate() const;
};

/**
 * @brief Global OCR configuration.
 */
struct OCRConfig {
    // Model paths
    std::string det_model_path;
    std::string rec_model_path;
    std::string dict_path;
    
    // Hardware configuration
    bool use_gpu = false;
    int gpu_id = 0;
    int num_threads = 4;
    
    // Memory optimization
    bool enable_memory_pattern = true;
    bool enable_cpu_mem_arena = true;
    
    // Component configurations
    DetectorConfig detector;
    RecognizerConfig recognizer;
    
    // Logging
    bool verbose = false;
    
    /**
     * @brief Initialize from command line arguments style.
     * @param det_path Detection model path.
     * @param rec_path Recognition model path.
     * @param dict Dictionary file path.
     * @return Configured OCRConfig object.
     */
    static OCRConfig FromPaths(const std::string& det_path,
                                const std::string& rec_path,
                                const std::string& dict);
    
    /**
     * @brief Validate configuration.
     * @return true if valid, false otherwise.
     */
    bool Validate() const;
};

// ============================================
// Utility functions
// ============================================

/**
 * @brief Timer for performance measurement.
 */
class ScopedTimer {
public:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = Clock::time_point;
    
    ScopedTimer(double* output_ms = nullptr)
        : output_ms_(output_ms), start_(Clock::now()) {}
    
    ~ScopedTimer() {
        if (output_ms_) {
            auto end = Clock::now();
            *output_ms_ = std::chrono::duration<double, std::milli>(
                end - start_).count();
        }
    }
    
    /**
     * @brief Get elapsed time in milliseconds.
     */
    double ElapsedMs() const {
        auto end = Clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }
    
private:
    double* output_ms_;
    TimePoint start_;
};

/**
 * @brief Load dictionary from file.
 * @param dict_path Path to dictionary file.
 * @return Vector of character strings.
 */
std::vector<std::string> LoadDictionary(const std::string& dict_path);

/**
 * @brief Convert image to blob for ONNX Runtime.
 * @param image Input image (BGR format).
 * @param mean Normalization mean values.
 * @param scale Normalization scale values.
 * @param swap_rb Swap R and B channels (convert BGR to RGB).
 * @return Blob data as float vector.
 */
std::vector<float> ImageToBlob(const cv::Mat& image,
                                const float mean[3],
                                const float scale[3],
                                bool swap_rb = true);

/**
 * @brief Resize image while keeping aspect ratio.
 * @param image Input image.
 * @param target_size Target size (width, height).
 * @param keep_ratio Keep aspect ratio.
 * @return Resized image and scale factor.
 */
std::pair<cv::Mat, float> ResizeWithAspectRatio(const cv::Mat& image,
                                                 const cv::Size& target_size,
                                                 bool keep_ratio = true);

/**
 * @brief Clip value to range.
 */
template<typename T>
inline T Clip(const T& value, const T& min_val, const T& max_val) {
    return std::max(min_val, std::min(value, max_val));
}

/**
 * @brief Convert box points to JSON array string.
 */
std::string BoxToJson(const std::vector<cv::Point>& box);

}  // namespace onnx
}  // namespace paddleocr

#endif  // PADDLEOCR_ONNX_INCLUDE_OCR_COMMON_H_
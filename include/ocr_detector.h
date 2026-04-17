// Copyright (c) 2024 PaddleOCR ONNX Runtime Project
// Licensed under the Apache License, Version 2.0

#ifndef PADDLEOCR_ONNX_INCLUDE_OCR_DETECTOR_H_
#define PADDLEOCR_ONNX_INCLUDE_OCR_DETECTOR_H_

#include <memory>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

#include "ocr_common.h"

namespace paddleocr {
namespace onnx {

/**
 * @brief Text detector using DB (Differentiable Binarization) algorithm.
 * 
 * This class implements the DB text detection algorithm using ONNX Runtime.
 * It detects text regions in an image and returns bounding boxes.
 */
class OCRDetector {
public:
    OCRDetector();
    ~OCRDetector();
    
    // Disable copy
    OCRDetector(const OCRDetector&) = delete;
    OCRDetector& operator=(const OCRDetector&) = delete;
    
    // Enable move
    OCRDetector(OCRDetector&&) noexcept;
    OCRDetector& operator=(OCRDetector&&) noexcept;
    
    /**
     * @brief Initialize the detector with configuration.
     * @param config Detector configuration.
     * @return true if initialization successful, false otherwise.
     */
    bool Init(const DetectorConfig& config);
    
    /**
     * @brief Initialize the detector with global OCR configuration.
     * @param config Global OCR configuration.
     * @return true if initialization successful, false otherwise.
     */
    bool Init(const OCRConfig& config);
    
    /**
     * @brief Check if detector is initialized.
     * @return true if initialized, false otherwise.
     */
    bool IsInitialized() const { return initialized_; }
    
    /**
     * @brief Detect text regions in an image.
     * @param image Input image (BGR format).
     * @return Vector of detected text boxes.
     */
    std::vector<TextBox> Detect(const cv::Mat& image);
    
    /**
     * @brief Detect text regions with performance statistics.
     * @param image Input image (BGR format).
     * @param stats Output performance statistics.
     * @return Vector of detected text boxes.
     */
    std::vector<TextBox> Detect(const cv::Mat& image, PerfStats& stats);
    
    /**
     * @brief Get the detector configuration.
     * @return Current configuration.
     */
    const DetectorConfig& GetConfig() const { return config_; }
    
    /**
     * @brief Update detector configuration.
     * @param config New configuration.
     */
    void SetConfig(const DetectorConfig& config);
    
    /**
     * @brief Get input tensor dimensions.
     * @return Vector of dimension names and values.
     */
    std::vector<std::pair<std::string, std::vector<int64_t>>> 
    GetInputShapes() const;
    
private:
    // ONNX Runtime components
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::SessionOptions> session_options_;
    std::unique_ptr<Ort::Session> session_;
    Ort::MemoryInfo memory_info_{nullptr};
    
    // Model metadata
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<const char*> input_name_ptrs_;
    std::vector<const char*> output_name_ptrs_;
    std::vector<std::vector<int64_t>> input_shapes_;
    std::vector<std::vector<int64_t>> output_shapes_;
    
    // Configuration
    DetectorConfig config_;
    bool initialized_ = false;
    
    // Preprocessing state
    cv::Size original_size_;
    float scale_x_ = 1.0f;
    float scale_y_ = 1.0f;
    int pad_right_ = 0;
    int pad_bottom_ = 0;
    
    /**
     * @brief Preprocess image for inference.
     * @param image Input image.
     * @return Preprocessed blob data.
     */
    std::vector<float> Preprocess(const cv::Mat& image);
    
    /**
     * @brief Run inference on preprocessed data.
     * @param input_data Preprocessed input blob.
     * @return Vector of output tensors.
     */
    std::vector<Ort::Value> RunInference(const std::vector<float>& input_data);
    
    /**
     * @brief Postprocess inference outputs to get text boxes.
     * @param outputs Inference output tensors.
     * @return Vector of detected text boxes.
     */
    std::vector<TextBox> Postprocess(const std::vector<Ort::Value>& outputs);
    
    /**
     * @brief Extract boxes from probability map.
     * @param prob_map Probability map from model output.
     * @param rows Number of rows.
     * @param cols Number of columns.
     * @return Vector of text boxes.
     */
    std::vector<TextBox> BoxesFromBitmap(const cv::Mat& prob_map,
                                          int rows, int cols);
    
    /**
     * @brief Unclip box for better coverage.
     * @param box Input box points.
     * @param unclip_ratio Ratio to expand.
     * @return Expanded box points.
     */
    std::vector<cv::Point> UnclipBox(const std::vector<cv::Point>& box,
                                      float unclip_ratio);
    
    /**
     * @brief Calculate distance between two points.
     */
    static float Distance(const cv::Point& a, const cv::Point& b);
    
    /**
     * @brief Calculate box area.
     */
    static float BoxArea(const std::vector<cv::Point>& box);
    
    /**
     * @brief Create ONNX Runtime session.
     */
    bool CreateSession();
};

}  // namespace onnx
}  // namespace paddleocr

#endif  // PADDLEOCR_ONNX_INCLUDE_OCR_DETECTOR_H_
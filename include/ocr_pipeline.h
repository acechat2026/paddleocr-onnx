// Copyright (c) 2024 PaddleOCR ONNX Runtime Project
// Licensed under the Apache License, Version 2.0

#ifndef PADDLEOCR_ONNX_INCLUDE_OCR_PIPELINE_H_
#define PADDLEOCR_ONNX_INCLUDE_OCR_PIPELINE_H_

#include <memory>
#include <vector>
#include <functional>
#include <opencv2/core.hpp>

#include "ocr_common.h"
#include "ocr_detector.h"
#include "ocr_recognizer.h"

namespace paddleocr {
namespace onnx {

/**
 * @brief Complete OCR pipeline combining detection and recognition.
 * 
 * This class orchestrates the full OCR workflow:
 * 1. Detect text regions using OCRDetector
 * 2. Crop detected regions
 * 3. Recognize text using OCRRecognizer
 * 4. Aggregate and return results
 */
class OCRPipeline {
public:
    OCRPipeline();
    ~OCRPipeline();
    
    // Disable copy
    OCRPipeline(const OCRPipeline&) = delete;
    OCRPipeline& operator=(const OCRPipeline&) = delete;
    
    // Enable move
    OCRPipeline(OCRPipeline&&) noexcept;
    OCRPipeline& operator=(OCRPipeline&&) noexcept;
    
    /**
     * @brief Initialize the pipeline with configuration.
     * @param config Global OCR configuration.
     * @return true if initialization successful, false otherwise.
     */
    bool Init(const OCRConfig& config);
    
    /**
     * @brief Initialize with separate detector and recognizer configs.
     * @param det_config Detector configuration.
     * @param rec_config Recognizer configuration.
     * @return true if initialization successful, false otherwise.
     */
    bool Init(const DetectorConfig& det_config,
              const RecognizerConfig& rec_config);
    
    /**
     * @brief Check if pipeline is initialized.
     * @return true if initialized, false otherwise.
     */
    bool IsInitialized() const;
    
    /**
     * @brief Run OCR on an image.
     * @param image Input image (BGR format).
     * @return Vector of OCR results.
     */
    std::vector<OCRResult> Run(const cv::Mat& image);
    
    /**
     * @brief Run OCR with performance statistics.
     * @param image Input image (BGR format).
     * @param stats Output performance statistics.
     * @return Vector of OCR results.
     */
    std::vector<OCRResult> Run(const cv::Mat& image, PerfStats& stats);
    
    /**
     * @brief Run OCR asynchronously with callback.
     * @param image Input image (BGR format).
     * @param callback Callback function called when processing completes.
     */
    void RunAsync(const cv::Mat& image,
                  std::function<void(std::vector<OCRResult>)> callback);
    
    /**
     * @brief Run detection only (skip recognition).
     * @param image Input image (BGR format).
     * @return Vector of detected text boxes.
     */
    std::vector<TextBox> DetectOnly(const cv::Mat& image);
    
    /**
     * @brief Run recognition only on pre-cropped regions.
     * @param images Vector of cropped text region images.
     * @return Vector of recognized text with confidence.
     */
    std::vector<std::pair<std::string, float>> RecognizeOnly(
        const std::vector<cv::Mat>& images);
    
    /**
     * @brief Visualize OCR results on image.
     * @param image Original image.
     * @param results OCR results to visualize.
     * @return Image with visualized results.
     */
    cv::Mat Visualize(const cv::Mat& image,
                      const std::vector<OCRResult>& results) const;
    
    /**
     * @brief Export results to JSON string.
     * @param results OCR results.
     * @param image_path Original image path.
     * @param image_size Original image size.
     * @param stats Performance statistics.
     * @return JSON string.
     */
    std::string ExportToJson(const std::vector<OCRResult>& results,
                              const std::string& image_path,
                              const cv::Size& image_size,
                              const PerfStats& stats) const;
    
    /**
     * @brief Get the detector instance.
     * @return Reference to detector.
     */
    OCRDetector& GetDetector() { return detector_; }
    const OCRDetector& GetDetector() const { return detector_; }
    
    /**
     * @brief Get the recognizer instance.
     * @return Reference to recognizer.
     */
    OCRRecognizer& GetRecognizer() { return recognizer_; }
    const OCRRecognizer& GetRecognizer() const { return recognizer_; }
    
    /**
     * @brief Get the pipeline configuration.
     * @return Current configuration.
     */
    const OCRConfig& GetConfig() const { return config_; }
    
    /**
     * @brief Update pipeline configuration.
     * @param config New configuration.
     */
    void SetConfig(const OCRConfig& config);
    
    /**
     * @brief Set the visualization style.
     * @param box_color Box color (BGR).
     * @param text_color Text color (BGR).
     * @param thickness Line thickness.
     * @param font_scale Font scale factor.
     */
    void SetVisualizationStyle(cv::Scalar box_color,
                               cv::Scalar text_color,
                               int thickness = 2,
                               double font_scale = 0.6);
    
private:
    OCRConfig config_;
    OCRDetector detector_;
    OCRRecognizer recognizer_;

    // Last run statistics
    PerfStats last_det_stats_;
    PerfStats last_rec_stats_;

    // Visualization style
    cv::Scalar box_color_ = cv::Scalar(0, 255, 0);      // Green
    cv::Scalar text_color_ = cv::Scalar(255, 0, 0);     // Blue
    int thickness_ = 2;
    double font_scale_ = 0.6;
    
    /**
     * @brief Crop detected text regions from image.
     * @param image Original image.
     * @param boxes Detected text boxes.
     * @return Vector of cropped images.
     */
    std::vector<cv::Mat> CropRegions(const cv::Mat& image,
                                      const std::vector<TextBox>& boxes);
    
    /**
     * @brief Sort boxes by reading order (top-to-bottom, left-to-right).
     * @param boxes Input boxes.
     * @return Sorted boxes.
     */
    std::vector<TextBox> SortByReadingOrder(const std::vector<TextBox>& boxes);
    
    /**
     * @brief Filter overlapping boxes.
     * @param boxes Input boxes.
     * @param iou_threshold IOU threshold for filtering.
     * @return Filtered boxes.
     */
    std::vector<TextBox> FilterOverlappingBoxes(const std::vector<TextBox>& boxes,
                                                  float iou_threshold = 0.5f);
    
    /**
     * @brief Calculate IOU between two boxes.
     */
    float CalculateIOU(const TextBox& a, const TextBox& b) const;
};

}  // namespace onnx
}  // namespace paddleocr

#endif  // PADDLEOCR_ONNX_INCLUDE_OCR_PIPELINE_H_
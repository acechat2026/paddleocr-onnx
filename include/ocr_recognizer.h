// Copyright (c) 2024 PaddleOCR ONNX Runtime Project
// Licensed under the Apache License, Version 2.0

#ifndef PADDLEOCR_ONNX_INCLUDE_OCR_RECOGNIZER_H_
#define PADDLEOCR_ONNX_INCLUDE_OCR_RECOGNIZER_H_

#include <memory>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

#include "ocr_common.h"

namespace paddleocr {
namespace onnx {

/**
 * @brief Text recognizer using CRNN algorithm.
 * 
 * This class implements the CRNN text recognition algorithm using ONNX Runtime.
 * It recognizes text content from cropped image regions.
 */
class OCRRecognizer {
public:
    OCRRecognizer();
    ~OCRRecognizer();
    
    // Disable copy
    OCRRecognizer(const OCRRecognizer&) = delete;
    OCRRecognizer& operator=(const OCRRecognizer&) = delete;
    
    // Enable move
    OCRRecognizer(OCRRecognizer&&) noexcept;
    OCRRecognizer& operator=(OCRRecognizer&&) noexcept;
    
    /**
     * @brief Initialize the recognizer with configuration.
     * @param config Recognizer configuration.
     * @return true if initialization successful, false otherwise.
     */
    bool Init(const RecognizerConfig& config);
    
    /**
     * @brief Initialize the recognizer with global OCR configuration.
     * @param config Global OCR configuration.
     * @return true if initialization successful, false otherwise.
     */
    bool Init(const OCRConfig& config);
    
    /**
     * @brief Check if recognizer is initialized.
     * @return true if initialized, false otherwise.
     */
    bool IsInitialized() const { return initialized_; }
    
    /**
     * @brief Recognize text from a single image region.
     * @param image Cropped text region image.
     * @return Recognized text with confidence.
     */
    std::pair<std::string, float> Recognize(const cv::Mat& image);
    
    /**
     * @brief Recognize text from multiple image regions (batch processing).
     * @param images Vector of cropped text region images.
     * @return Vector of recognized text with confidence scores.
     */
    std::vector<std::pair<std::string, float>> RecognizeBatch(
        const std::vector<cv::Mat>& images);
    
    /**
     * @brief Recognize text from multiple image regions with performance stats.
     * @param images Vector of cropped text region images.
     * @param stats Output performance statistics.
     * @return Vector of recognized text with confidence scores.
     */
    std::vector<std::pair<std::string, float>> RecognizeBatch(
        const std::vector<cv::Mat>& images, PerfStats& stats);
    
    /**
     * @brief Get the recognizer configuration.
     * @return Current configuration.
     */
    const RecognizerConfig& GetConfig() const { return config_; }
    
    /**
     * @brief Update recognizer configuration.
     * @param config New configuration.
     */
    void SetConfig(const RecognizerConfig& config);
    
    /**
     * @brief Get the loaded dictionary.
     * @return Vector of character strings.
     */
    const std::vector<std::string>& GetDictionary() const { return dictionary_; }
    
    /**
     * @brief Get the number of classes (dictionary size).
     * @return Number of classes.
     */
    size_t GetNumClasses() const { return dictionary_.size(); }
    
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
    
    // Configuration and data
    RecognizerConfig config_;
    std::vector<std::string> dictionary_;
    bool initialized_ = false;
    
    // CTC decoder parameters
    int blank_index_ = 0;
    
    /**
     * @brief Preprocess a single image for inference.
     * @param image Input image.
     * @return Preprocessed blob data.
     */
    std::vector<float> Preprocess(const cv::Mat& image);
    
    /**
     * @brief Preprocess batch of images for inference.
     * @param images Vector of input images.
     * @return Preprocessed blob data.
     */
    std::vector<float> PreprocessBatch(const std::vector<cv::Mat>& images);
    
    /**
     * @brief Run inference on preprocessed data.
     * @param input_data Preprocessed input blob.
     * @param batch_size Number of images in batch.
     * @return Vector of output tensors.
     */
    std::vector<Ort::Value> RunInference(const std::vector<float>& input_data,
                                          int batch_size);
    
    /**
     * @brief Decode CTC output to text.
     * @param output Inference output tensor.
     * @param batch_size Batch size.
     * @param batch_index Index in batch.
     * @return Decoded text and average confidence.
     */
    std::pair<std::string, float> DecodeCTC(const Ort::Value& output,
                                             int batch_size,
                                             int batch_index);
    
    /**
     * @brief Create ONNX Runtime session.
     */
    bool CreateSession();
    
    /**
     * @brief Load dictionary from file.
     */
    bool LoadDictionary();
};

}  // namespace onnx
}  // namespace paddleocr

#endif  // PADDLEOCR_ONNX_INCLUDE_OCR_RECOGNIZER_H_
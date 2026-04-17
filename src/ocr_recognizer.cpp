// Copyright (c) 2024 PaddleOCR ONNX Runtime Project
// Licensed under the Apache License, Version 2.0

#include "ocr_recognizer.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#ifdef _WIN32
#include <windows.h>
#endif
#include <opencv2/imgproc.hpp>

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

OCRRecognizer::OCRRecognizer() = default;

OCRRecognizer::~OCRRecognizer() = default;

OCRRecognizer::OCRRecognizer(OCRRecognizer&& other) noexcept
    : env_(std::move(other.env_))
    , session_options_(std::move(other.session_options_))
    , session_(std::move(other.session_))
    , memory_info_(std::move(other.memory_info_))
    , input_names_(std::move(other.input_names_))
    , output_names_(std::move(other.output_names_))
    , input_shapes_(std::move(other.input_shapes_))
    , output_shapes_(std::move(other.output_shapes_))
    , config_(std::move(other.config_))
    , dictionary_(std::move(other.dictionary_))
    , initialized_(other.initialized_)
    , blank_index_(other.blank_index_) {
    other.initialized_ = false;
}

OCRRecognizer& OCRRecognizer::operator=(OCRRecognizer&& other) noexcept {
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
        dictionary_ = std::move(other.dictionary_);
        initialized_ = other.initialized_;
        blank_index_ = other.blank_index_;
        other.initialized_ = false;
    }
    return *this;
}

// ============================================
// Initialization
// ============================================

bool OCRRecognizer::Init(const RecognizerConfig& config) {
    config_ = config;
    
    if (!config_.Validate()) {
        return false;
    }
    
    if (!LoadDictionary()) {
        return false;
    }
    
    if (!CreateSession()) {
        return false;
    }
    
    initialized_ = true;
    return true;
}

bool OCRRecognizer::Init(const OCRConfig& config) {
    RecognizerConfig rec_config;
    rec_config.model_path = config.rec_model_path;
    rec_config.dict_path = config.dict_path;
    rec_config.image_height = config.recognizer.image_height;
    rec_config.batch_size = config.recognizer.batch_size;
    rec_config.drop_score = config.recognizer.drop_score;
    
    return Init(rec_config);
}

bool OCRRecognizer::CreateSession() {
    try {
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "OCRRecognizer");
        
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
        
        memory_info_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        
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

        // Set blank index (last class in CTC, after all dictionary chars)
        blank_index_ = static_cast<int>(dictionary_.size());
        
        return true;
        
    } catch (const Ort::Exception& e) {
        return false;
    }
}

bool OCRRecognizer::LoadDictionary() {
    dictionary_ = paddleocr::onnx::LoadDictionary(config_.dict_path);
    return !dictionary_.empty();
}

// ============================================
// Configuration
// ============================================

void OCRRecognizer::SetConfig(const RecognizerConfig& config) {
    config_ = config;
}

std::vector<std::pair<std::string, std::vector<int64_t>>> 
OCRRecognizer::GetInputShapes() const {
    std::vector<std::pair<std::string, std::vector<int64_t>>> result;
    for (size_t i = 0; i < input_names_.size(); ++i) {
        result.emplace_back(input_names_[i], input_shapes_[i]);
    }
    return result;
}

// ============================================
// Recognition
// ============================================

std::pair<std::string, float> OCRRecognizer::Recognize(const cv::Mat& image) {
    auto results = RecognizeBatch({image});
    if (results.empty()) {
        return {"", 0.0f};
    }
    return results[0];
}

std::vector<std::pair<std::string, float>> OCRRecognizer::RecognizeBatch(
    const std::vector<cv::Mat>& images) {
    PerfStats stats;
    return RecognizeBatch(images, stats);
}

std::vector<std::pair<std::string, float>> OCRRecognizer::RecognizeBatch(
    const std::vector<cv::Mat>& images, PerfStats& stats) {
    stats.Reset();
    
    if (!initialized_ || images.empty()) {
        return {};
    }
    
    std::vector<std::pair<std::string, float>> results;
    
    // Process in batches
    for (size_t i = 0; i < images.size(); i += config_.batch_size) {
        size_t batch_end = std::min(i + static_cast<size_t>(config_.batch_size), images.size());
        std::vector<cv::Mat> batch(images.begin() + i, images.begin() + batch_end);
        
        // Preprocess
        double preprocess_ms = 0;
        std::vector<float> input_data;
        {
            ScopedTimer timer(&preprocess_ms);
            input_data = PreprocessBatch(batch);
        }
        stats.preprocess_ms += preprocess_ms;
        
        // Inference
        double inference_ms = 0;
        std::vector<Ort::Value> outputs;
        {
            ScopedTimer timer(&inference_ms);
            outputs = RunInference(input_data, static_cast<int>(batch.size()));
        }
        stats.inference_ms += inference_ms;
        
        // Postprocess
        double postprocess_ms = 0;
        {
            ScopedTimer timer(&postprocess_ms);
            for (size_t j = 0; j < batch.size(); ++j) {
                auto result = DecodeCTC(outputs[0], static_cast<int>(batch.size()), 
                                         static_cast<int>(j));
                results.push_back(result);
            }
        }
        stats.postprocess_ms += postprocess_ms;
    }
    
    stats.num_text_regions = static_cast<int>(results.size());
    stats.total_ms = stats.preprocess_ms + stats.inference_ms + stats.postprocess_ms;
    
    return results;
}

// ============================================
// Preprocessing
// ============================================

std::vector<float> OCRRecognizer::Preprocess(const cv::Mat& image) {
    return PreprocessBatch({image});
}

std::vector<float> OCRRecognizer::PreprocessBatch(const std::vector<cv::Mat>& images) {
    int batch_size = static_cast<int>(images.size());
    int channels = 3;
    int height = config_.image_height;
    int width = config_.image_width;
    
    std::vector<float> blob(batch_size * channels * height * width, 0.0f);
    
    for (int b = 0; b < batch_size; ++b) {
        cv::Mat img = images[b];
        
        if (img.channels() == 1) {
            cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
        } else if (img.channels() == 4) {
            cv::cvtColor(img, img, cv::COLOR_BGRA2BGR);
        }
        
        // Resize keeping aspect ratio
        float aspect_ratio = static_cast<float>(img.cols) / img.rows;
        int new_width = static_cast<int>(height * aspect_ratio);
        new_width = std::min(new_width, width);
        
        cv::Mat resized;
        cv::resize(img, resized, cv::Size(new_width, height));
        
        // Convert to float and normalize
        resized.convertTo(resized, CV_32FC3, 1.0f / 255.0f);
        
        // Apply mean and scale normalization
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < new_width; ++w) {
                cv::Vec3f& pixel = resized.at<cv::Vec3f>(h, w);
                for (int c = 0; c < channels; ++c) {
                    pixel[c] = (pixel[c] - config_.mean[c]) / config_.scale[c];
                }
            }
        }
        
        // Copy to blob (CHW format)
        int base_offset = b * channels * height * width;
        for (int c = 0; c < channels; ++c) {
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < new_width; ++w) {
                    int idx = base_offset + c * height * width + h * width + w;
                    blob[idx] = resized.at<cv::Vec3f>(h, w)[c];
                }
            }
        }
    }
    
    return blob;
}

// ============================================
// Inference
// ============================================

std::vector<Ort::Value> OCRRecognizer::RunInference(const std::vector<float>& input_data,
                                                      int batch_size) {
    if (input_shapes_.empty()) {
        return {};
    }
    
    auto& shape = input_shapes_[0];
    std::vector<int64_t> actual_shape = shape;
    
    if (actual_shape.size() == 4) {
        actual_shape[0] = batch_size;
        actual_shape[2] = config_.image_height;
        actual_shape[3] = config_.image_width;
    }
    
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info_,
        const_cast<float*>(input_data.data()),
        input_data.size(),
        actual_shape.data(),
        actual_shape.size()
    );
    
    std::vector<Ort::Value> output_tensors;
    
    try {
        output_tensors = session_->Run(Ort::RunOptions{nullptr},
                                        input_name_ptrs_.data(),
                                        &input_tensor,
                                        1,
                                        output_name_ptrs_.data(),
                                        output_name_ptrs_.size());
    } catch (const Ort::Exception& e) {
        return {};
    }
    
    return output_tensors;
}

// ============================================
// CTC Decoding
// ============================================

std::pair<std::string, float> OCRRecognizer::DecodeCTC(const Ort::Value& output,
                                                         int batch_size,
                                                         int batch_index) {
    auto type_info = output.GetTensorTypeAndShapeInfo();
    auto shape = type_info.GetShape();
    
    if (shape.size() != 3) {
        return {"", 0.0f};
    }
    
    int time_steps = static_cast<int>(shape[1]);
    int num_classes = static_cast<int>(shape[2]);
    
    const float* output_data = output.GetTensorData<float>();
    
    // Get predictions for this batch item
    int batch_offset = batch_index * time_steps * num_classes;
    
    // Greedy CTC decoding
    std::vector<int> indices;
    std::vector<float> confidences;
    
    int prev_idx = blank_index_;
    
    for (int t = 0; t < time_steps; ++t) {
        // Find max class for this time step
        const float* time_step_data = output_data + batch_offset + t * num_classes;
        
        int max_idx = 0;
        float max_val = time_step_data[0];
        for (int c = 1; c < num_classes; ++c) {
            if (time_step_data[c] > max_val) {
                max_val = time_step_data[c];
                max_idx = c;
            }
        }
        
        // Apply CTC merge rules
        if (max_idx != blank_index_ && max_idx != prev_idx) {
            indices.push_back(max_idx);
            confidences.push_back(max_val);
        }
        
        prev_idx = max_idx;
    }
    
    // Build text string from indices
    std::string text;
    float total_conf = 0.0f;
    int valid_chars = 0;
    
    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        if (idx >= 0 && static_cast<size_t>(idx) < dictionary_.size()) {
            // Filter low confidence characters
            if (confidences[i] >= config_.drop_score) {
                text += dictionary_[idx];
                total_conf += confidences[i];
                ++valid_chars;
            }
        }
    }
    
    float avg_confidence = valid_chars > 0 ? total_conf / valid_chars : 0.0f;
    
    return {text, avg_confidence};
}

}  // namespace onnx
}  // namespace paddleocr
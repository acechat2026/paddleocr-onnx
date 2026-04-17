// Copyright (c) 2024 PaddleOCR ONNX Runtime Project
// Licensed under the Apache License, Version 2.0

#include "ocr_http_server.h"

#include <fstream>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <filesystem>

// cpp-httplib is header-only
#include "httplib.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace paddleocr {
namespace onnx {

namespace fs = std::filesystem;

// ============================================
// Base64 encoding/decoding (internal helpers)
// ============================================

namespace detail {

static const std::string base64_chars =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/";

std::vector<uint8_t> Base64Decode(const std::string& encoded_string) {
    int in_len = static_cast<int>(encoded_string.size());
    int i = 0;
    int j = 0;
    int in_ = 0;
    uint8_t char_array_4[4], char_array_3[3];
    std::vector<uint8_t> ret;
    
    while (in_len-- && (encoded_string[in_] != '=') &&
           (std::isalnum(static_cast<unsigned char>(encoded_string[in_])) ||
            (encoded_string[in_] == '+') ||
            (encoded_string[in_] == '/'))) {
        char_array_4[i++] = encoded_string[in_];
        in_++;
        if (i == 4) {
            for (i = 0; i < 4; i++) {
                char_array_4[i] = static_cast<uint8_t>(
                    base64_chars.find(char_array_4[i]));
            }
            
            char_array_3[0] = (char_array_4[0] << 2) +
                              ((char_array_4[1] & 0x30) >> 4);
            char_array_3[1] = ((char_array_4[1] & 0xf) << 4) +
                              ((char_array_4[2] & 0x3c) >> 2);
            char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];
            
            for (i = 0; i < 3; i++) {
                ret.push_back(char_array_3[i]);
            }
            i = 0;
        }
    }
    
    if (i) {
        for (j = i; j < 4; j++) {
            char_array_4[j] = 0;
        }
        for (j = 0; j < 4; j++) {
            char_array_4[j] = static_cast<uint8_t>(
                base64_chars.find(char_array_4[j]));
        }
        
        char_array_3[0] = (char_array_4[0] << 2) +
                          ((char_array_4[1] & 0x30) >> 4);
        char_array_3[1] = ((char_array_4[1] & 0xf) << 4) +
                          ((char_array_4[2] & 0x3c) >> 2);
        char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];
        
        for (j = 0; j < i - 1; j++) {
            ret.push_back(char_array_3[j]);
        }
    }
    
    return ret;
}

std::string Base64Encode(const std::vector<uint8_t>& data) {
    std::string ret;
    int i = 0;
    int j = 0;
    uint8_t char_array_3[3];
    uint8_t char_array_4[4];
    size_t in_len = data.size();
    const uint8_t* bytes_to_encode = data.data();
    
    while (in_len--) {
        char_array_3[i++] = *(bytes_to_encode++);
        if (i == 3) {
            char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
            char_array_4[1] = ((char_array_3[0] & 0x03) << 4) +
                              ((char_array_3[1] & 0xf0) >> 4);
            char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) +
                              ((char_array_3[2] & 0xc0) >> 6);
            char_array_4[3] = char_array_3[2] & 0x3f;
            
            for (i = 0; i < 4; i++) {
                ret += base64_chars[char_array_4[i]];
            }
            i = 0;
        }
    }
    
    if (i) {
        for (j = i; j < 3; j++) {
            char_array_3[j] = '\0';
        }
        
        char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
        char_array_4[1] = ((char_array_3[0] & 0x03) << 4) +
                          ((char_array_3[1] & 0xf0) >> 4);
        char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) +
                          ((char_array_3[2] & 0xc0) >> 6);
        char_array_4[3] = char_array_3[2] & 0x3f;
        
        for (j = 0; j < i + 1; j++) {
            ret += base64_chars[char_array_4[j]];
        }
        
        while ((i++ < 3)) {
            ret += '=';
        }
    }
    
    return ret;
}

}  // namespace detail

// ============================================
// Base64 helpers for class methods
// ============================================

std::vector<uint8_t> OCRHttpServer::DecodeBase64(const std::string& base64_str) {
    return detail::Base64Decode(base64_str);
}

std::string OCRHttpServer::EncodeBase64(const cv::Mat& image,
                                         const std::string& format) {
    std::vector<uint8_t> buf;
    std::vector<int> params;
    if (format == "jpg" || format == "jpeg") {
        params = {cv::IMWRITE_JPEG_QUALITY, 95};
        cv::imencode(".jpg", image, buf, params);
    } else {
        params = {cv::IMWRITE_PNG_COMPRESSION, 3};
        cv::imencode(".png", image, buf, params);
    }
    return detail::Base64Encode(buf);
}

// ============================================
// Constructor & Destructor
// ============================================

OCRHttpServer::OCRHttpServer() = default;

OCRHttpServer::~OCRHttpServer() {
    Stop();
}

// ============================================
// Initialization
// ============================================

bool OCRHttpServer::Init(std::shared_ptr<OCRPipeline> pipeline,
                          const HttpServerConfig& config) {
    if (!pipeline || !pipeline->IsInitialized()) {
        return false;
    }
    
    pipeline_ = pipeline;
    config_ = config;
    
    // Create upload directory
    if (!fs::exists(config_.upload_dir)) {
        fs::create_directories(config_.upload_dir);
    }
    
    // Create server
    server_ = std::make_unique<httplib::Server>();
    
    // Setup routes
    SetupRoutes();
    
    return true;
}

// ============================================
// Route Setup
// ============================================

void OCRHttpServer::SetupRoutes() {
    if (!server_) return;
    
    // CORS middleware
    if (config_.enable_cors) {
        server_->set_pre_routing_handler([](const httplib::Request& req,
                                             httplib::Response& res) {
            res.set_header("Access-Control-Allow-Origin", "*");
            res.set_header("Access-Control-Allow-Methods",
                          "GET, POST, PUT, DELETE, OPTIONS");
            res.set_header("Access-Control-Allow-Headers",
                          "Content-Type, Authorization");
            if (req.method == "OPTIONS") {
                res.status = 204;
                return httplib::Server::HandlerResponse::Handled;
            }
            return httplib::Server::HandlerResponse::Unhandled;
        });
    }
    
    // Health check
    if (config_.enable_health_check) {
        server_->Get("/health", [this](const httplib::Request& req,
                                        httplib::Response& res) {
            HandleHealth(req, res);
        });
    }
    
    // Statistics
    server_->Get("/stats", [this](const httplib::Request& req,
                                   httplib::Response& res) {
        HandleStats(req, res);
    });
    
    // OCR endpoints
    server_->Post("/ocr", [this](const httplib::Request& req,
                                  httplib::Response& res) {
        HandleOCR(req, res);
    });
    
    server_->Post("/ocr/base64", [this](const httplib::Request& req,
                                         httplib::Response& res) {
        HandleBase64OCR(req, res);
    });
    
    server_->Post("/ocr/batch", [this](const httplib::Request& req,
                                        httplib::Response& res) {
        HandleBatchOCR(req, res);
    });
}

// ============================================
// Start/Stop
// ============================================

bool OCRHttpServer::Start(bool blocking) {
    if (!server_ || running_) {
        return false;
    }
    
    // Start worker threads
    stop_workers_ = false;
    for (int i = 0; i < config_.num_threads; ++i) {
        worker_threads_.emplace_back(&OCRHttpServer::WorkerThread, this);
    }
    
    running_ = true;
    
    if (blocking) {
        server_->listen(config_.host.c_str(), config_.port);
        running_ = false;
    } else {
        server_thread_ = std::make_unique<std::thread>([this]() {
            server_->listen(config_.host.c_str(), config_.port);
            running_ = false;
        });
    }
    
    return true;
}

void OCRHttpServer::Stop() {
    if (server_) {
        server_->stop();
    }
    
    // Stop worker threads
    stop_workers_ = true;
    queue_cv_.notify_all();
    
    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    worker_threads_.clear();
    
    if (server_thread_ && server_thread_->joinable()) {
        server_thread_->join();
    }
    
    running_ = false;
}

// ============================================
// Request Handlers
// ============================================

void OCRHttpServer::HandleHealth(const httplib::Request& req,
                                  httplib::Response& res) {
    std::ostringstream oss;
    oss << "{";
    oss << "\"status\":\"ok\",";
    oss << "\"models_loaded\":" << (pipeline_ && pipeline_->IsInitialized()) << ",";
    oss << "\"gpu_enabled\":" << (pipeline_ ? pipeline_->GetConfig().use_gpu : false);
    oss << "}";
    
    res.set_content(oss.str(), "application/json");
}

void OCRHttpServer::HandleStats(const httplib::Request& req,
                                 httplib::Response& res) {
    HttpStats stats = GetStats();
    
    std::ostringstream oss;
    oss << "{";
    oss << "\"total_requests\":" << stats.total_requests << ",";
    oss << "\"successful_requests\":" << stats.successful_requests << ",";
    oss << "\"failed_requests\":" << stats.failed_requests << ",";
    oss << "\"avg_inference_time_ms\":" << stats.GetAverageInferenceTime();
    oss << "}";

    res.set_content(oss.str(), "application/json");
}

void OCRHttpServer::HandleOCR(const httplib::Request& req,
                               httplib::Response& res) {
    stats_.total_requests++;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Check for file upload
    auto file = req.get_file_value("file");
    if (file.content.empty()) {
        res.set_content(ErrorResponse(400, "No file uploaded"), "application/json");
        stats_.failed_requests++;
        return;
    }
    
    // Process image
    std::string result = ProcessOCRRequest(file.content);
    
    // Update stats
    stats_.successful_requests++;
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time).count();
    stats_.total_inference_time_ms += duration;
    
    res.set_content(result, "application/json");
}

void OCRHttpServer::HandleBase64OCR(const httplib::Request& req,
                                     httplib::Response& res) {
    stats_.total_requests++;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Parse JSON body
    std::string body = req.body;
    
    // Simple JSON parsing (in production, use a proper JSON library)
    std::string image_key = "\"image\":\"";
    size_t start = body.find(image_key);
    if (start == std::string::npos) {
        res.set_content(ErrorResponse(400, "Missing 'image' field"), "application/json");
        stats_.failed_requests++;
        return;
    }
    
    start += image_key.length();
    size_t end = body.find("\"", start);
    if (end == std::string::npos) {
        res.set_content(ErrorResponse(400, "Invalid JSON format"), "application/json");
        stats_.failed_requests++;
        return;
    }
    
    std::string base64_data = body.substr(start, end - start);
    
    // Decode base64
    std::vector<uint8_t> image_data = DecodeBase64(base64_data);
    std::string image_str(image_data.begin(), image_data.end());
    
    // Process image
    std::string result = ProcessOCRRequest(image_str);
    
    // Update stats
    stats_.successful_requests++;
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time).count();
    stats_.total_inference_time_ms += duration;
    
    res.set_content(result, "application/json");
}

void OCRHttpServer::HandleBatchOCR(const httplib::Request& req,
                                    httplib::Response& res) {
    stats_.total_requests++;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Get all files
    std::vector<std::string> images;
    for (const auto& file : req.files) {
        if (!file.second.content.empty()) {
            images.push_back(file.second.content);
        }
    }
    
    if (images.empty()) {
        res.set_content(ErrorResponse(400, "No files uploaded"), "application/json");
        stats_.failed_requests++;
        return;
    }
    
    // Process batch
    std::string result = ProcessBatchOCRRequest(images);
    
    // Update stats
    stats_.successful_requests++;
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time).count();
    stats_.total_inference_time_ms += duration;
    
    res.set_content(result, "application/json");
}

// ============================================
// OCR Processing
// ============================================

std::string OCRHttpServer::ProcessOCRRequest(const std::string& image_data) {
    if (!pipeline_) {
        return ErrorResponse(500, "Pipeline not initialized");
    }
    
    // Decode image
    std::vector<uint8_t> buffer(image_data.begin(), image_data.end());
    cv::Mat image = cv::imdecode(buffer, cv::IMREAD_COLOR);
    
    if (image.empty()) {
        return ErrorResponse(400, "Invalid image data");
    }
    
    // Run OCR
    PerfStats stats;
    auto results = pipeline_->Run(image, stats);
    
    // Build response
    std::ostringstream oss;
    oss << "{";
    oss << "\"code\":0,";
    oss << "\"message\":\"success\",";
    oss << "\"data\":{";
    oss << "\"results\":[";
    
    for (size_t i = 0; i < results.size(); ++i) {
        if (i > 0) oss << ",";
        oss << results[i].ToJson();
    }
    
    oss << "],";
    oss << "\"count\":" << results.size() << ",";
    oss << "\"inference_time_ms\":" << stats.total_ms;
    oss << "}";
    oss << "}";
    
    return oss.str();
}

std::string OCRHttpServer::ProcessBatchOCRRequest(
    const std::vector<std::string>& images) {
    if (!pipeline_) {
        return ErrorResponse(500, "Pipeline not initialized");
    }
    
    std::ostringstream oss;
    oss << "{";
    oss << "\"code\":0,";
    oss << "\"message\":\"success\",";
    oss << "\"data\":[";
    
    for (size_t i = 0; i < images.size(); ++i) {
        if (i > 0) oss << ",";
        
        // Decode image
        std::vector<uint8_t> buffer(images[i].begin(), images[i].end());
        cv::Mat image = cv::imdecode(buffer, cv::IMREAD_COLOR);
        
        if (image.empty()) {
            oss << "{\"error\":\"Invalid image data\"}";
            continue;
        }
        
        // Run OCR
        PerfStats stats;
        auto results = pipeline_->Run(image, stats);
        
        oss << "{";
        oss << "\"filename\":\"image_" << i << ".jpg\",";
        oss << "\"results\":[";
        
        for (size_t j = 0; j < results.size(); ++j) {
            if (j > 0) oss << ",";
            oss << results[j].ToJson();
        }
        
        oss << "],";
        oss << "\"count\":" << results.size() << ",";
        oss << "\"inference_time_ms\":" << stats.total_ms;
        oss << "}";
    }
    
    oss << "]";
    oss << "}";
    
    return oss.str();
}

// ============================================
// Response Helpers
// ============================================

std::string OCRHttpServer::ErrorResponse(int code, const std::string& message) {
    std::ostringstream oss;
    oss << "{";
    oss << "\"code\":" << code << ",";
    oss << "\"message\":\"" << message << "\",";
    oss << "\"data\":null";
    oss << "}";
    return oss.str();
}

std::string OCRHttpServer::SuccessResponse(const std::string& data) {
    std::ostringstream oss;
    oss << "{";
    oss << "\"code\":0,";
    oss << "\"message\":\"success\",";
    oss << "\"data\":" << data;
    oss << "}";
    return oss.str();
}

// ============================================
// Worker Thread
// ============================================

void OCRHttpServer::WorkerThread() {
    while (!stop_workers_) {
        RequestTask task;
        
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [this] {
                return !request_queue_.empty() || stop_workers_;
            });
            
            if (stop_workers_ && request_queue_.empty()) {
                break;
            }
            
            task = std::move(request_queue_.front());
            request_queue_.pop();
        }
        
        // Process task
        if (task.callback) {
            std::string result = ProcessOCRRequest(task.image_data);
            task.callback(result);
        }
    }
}

// ============================================
// File Utilities
// ============================================

std::string OCRHttpServer::SaveUploadFile(const std::string& data,
                                           const std::string& filename) {
    fs::path file_path = fs::path(config_.upload_dir) / filename;
    
    std::ofstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        return "";
    }
    
    file.write(data.c_str(), data.size());
    file.close();
    
    return file_path.string();
}

void OCRHttpServer::CleanupOldFiles() {
    // Cleanup files older than 1 hour
    (void)std::chrono::system_clock::now();

    for (const auto& entry : fs::directory_iterator(config_.upload_dir)) {
        (void)fs::last_write_time(entry);
        // Cleanup logic would go here with proper time comparison
        // This is a simplified placeholder
    }
}

}  // namespace onnx
}  // namespace paddleocr
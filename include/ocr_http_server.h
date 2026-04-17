// Copyright (c) 2024 PaddleOCR ONNX Runtime Project
// Licensed under the Apache License, Version 2.0

#ifndef PADDLEOCR_ONNX_INCLUDE_OCR_HTTP_SERVER_H_
#define PADDLEOCR_ONNX_INCLUDE_OCR_HTTP_SERVER_H_

#include <string>
#include <memory>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>

#include "ocr_pipeline.h"

// Forward declaration for httplib
namespace httplib {
class Server;
struct Request;
struct Response;
}  // namespace httplib

namespace paddleocr {
namespace onnx {

/**
 * @brief HTTP server configuration.
 */
struct HttpServerConfig {
    std::string host = "0.0.0.0";
    int port = 8080;
    int num_threads = 4;
    int max_queue_size = 100;
    int request_timeout_seconds = 30;
    bool enable_cors = true;
    bool enable_health_check = true;
    std::string upload_dir = "/tmp/ocr_uploads";
};

/**
 * @brief HTTP request statistics.
 */
struct HttpStats {
    uint64_t total_requests = 0;
    uint64_t successful_requests = 0;
    uint64_t failed_requests = 0;
    uint64_t total_inference_time_ms = 0;

    double GetAverageInferenceTime() const {
        return total_requests > 0 ? static_cast<double>(total_inference_time_ms) / total_requests : 0.0;
    }

    void Reset() {
        total_requests = 0;
        successful_requests = 0;
        failed_requests = 0;
        total_inference_time_ms = 0;
    }
};

/**
 * @brief HTTP server for OCR service.
 * 
 * Provides RESTful API endpoints:
 * - GET  /health          Health check
 * - POST /ocr             OCR with file upload
 * - POST /ocr/base64      OCR with base64 encoded image
 * - POST /ocr/batch       Batch OCR with multiple files
 * - GET  /stats           Server statistics
 */
class OCRHttpServer {
public:
    OCRHttpServer();
    ~OCRHttpServer();
    
    // Disable copy
    OCRHttpServer(const OCRHttpServer&) = delete;
    OCRHttpServer& operator=(const OCRHttpServer&) = delete;
    
    /**
     * @brief Initialize server with OCR pipeline and configuration.
     * @param pipeline Initialized OCR pipeline.
     * @param config Server configuration.
     * @return true if initialization successful, false otherwise.
     */
    bool Init(std::shared_ptr<OCRPipeline> pipeline,
              const HttpServerConfig& config);
    
    /**
     * @brief Start the HTTP server.
     * @param blocking If true, blocks until server stops.
     * @return true if started successfully, false otherwise.
     */
    bool Start(bool blocking = true);
    
    /**
     * @brief Stop the HTTP server.
     */
    void Stop();
    
    /**
     * @brief Check if server is running.
     * @return true if running, false otherwise.
     */
    bool IsRunning() const { return running_.load(); }
    
    /**
     * @brief Get server statistics.
     * @return HTTP statistics.
     */
    HttpStats GetStats() const {
        HttpStats copy;
        copy.total_requests = stats_.total_requests.load();
        copy.successful_requests = stats_.successful_requests.load();
        copy.failed_requests = stats_.failed_requests.load();
        copy.total_inference_time_ms = stats_.total_inference_time_ms.load();
        return copy;
    }

    /**
     * @brief Reset server statistics.
     */
    void ResetStats() {
        stats_.total_requests = 0;
        stats_.successful_requests = 0;
        stats_.failed_requests = 0;
        stats_.total_inference_time_ms = 0;
    }
    
    /**
     * @brief Get server configuration.
     * @return Current configuration.
     */
    const HttpServerConfig& GetConfig() const { return config_; }
    
    /**
     * @brief Get the underlying OCR pipeline.
     * @return Shared pointer to pipeline.
     */
    std::shared_ptr<OCRPipeline> GetPipeline() const { return pipeline_; }
    
private:
    std::shared_ptr<OCRPipeline> pipeline_;
    HttpServerConfig config_;
    std::unique_ptr<httplib::Server> server_;
    std::unique_ptr<std::thread> server_thread_;
    std::atomic<bool> running_{false};

    struct AtomicHttpStats {
        std::atomic<uint64_t> total_requests{0};
        std::atomic<uint64_t> successful_requests{0};
        std::atomic<uint64_t> failed_requests{0};
        std::atomic<uint64_t> total_inference_time_ms{0};
    } stats_;

    // Request queue for async processing
    struct RequestTask {
        std::string image_data;
        std::string content_type;
        std::function<void(const std::string&)> callback;
        std::chrono::steady_clock::time_point enqueue_time;
    };
    
    std::queue<RequestTask> request_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::vector<std::thread> worker_threads_;
    std::atomic<bool> stop_workers_{false};
    
    /**
     * @brief Setup HTTP route handlers.
     */
    void SetupRoutes();
    
    /**
     * @brief Handle health check request.
     */
    void HandleHealth(const httplib::Request& req, httplib::Response& res);
    
    /**
     * @brief Handle OCR request (file upload).
     */
    void HandleOCR(const httplib::Request& req, httplib::Response& res);
    
    /**
     * @brief Handle base64 OCR request.
     */
    void HandleBase64OCR(const httplib::Request& req, httplib::Response& res);
    
    /**
     * @brief Handle batch OCR request.
     */
    void HandleBatchOCR(const httplib::Request& req, httplib::Response& res);
    
    /**
     * @brief Handle statistics request.
     */
    void HandleStats(const httplib::Request& req, httplib::Response& res);
    
    /**
     * @brief Process a single OCR request.
     * @param image_data Raw image data.
     * @return JSON response string.
     */
    std::string ProcessOCRRequest(const std::string& image_data);
    
    /**
     * @brief Process a batch OCR request.
     * @param images Vector of raw image data.
     * @return JSON response string.
     */
    std::string ProcessBatchOCRRequest(const std::vector<std::string>& images);
    
    /**
     * @brief Decode base64 image.
     * @param base64_str Base64 encoded string.
     * @return Decoded image data or empty vector on error.
     */
    std::vector<uint8_t> DecodeBase64(const std::string& base64_str);
    
    /**
     * @brief Encode image to base64.
     * @param image Image to encode.
     * @param format Image format (jpg, png).
     * @return Base64 encoded string.
     */
    std::string EncodeBase64(const cv::Mat& image, const std::string& format = "jpg");
    
    /**
     * @brief Create error JSON response.
     * @param code Error code.
     * @param message Error message.
     * @return JSON string.
     */
    std::string ErrorResponse(int code, const std::string& message);
    
    /**
     * @brief Create success JSON response.
     * @param data Response data.
     * @return JSON string.
     */
    std::string SuccessResponse(const std::string& data);
    
    /**
     * @brief Worker thread function for async processing.
     */
    void WorkerThread();
    
    /**
     * @brief Save uploaded file to disk.
     * @param data File data.
     * @param filename Output filename.
     * @return Path to saved file.
     */
    std::string SaveUploadFile(const std::string& data, const std::string& filename);
    
    /**
     * @brief Clean up old uploaded files.
     */
    void CleanupOldFiles();
};

}  // namespace onnx
}  // namespace paddleocr

#endif  // PADDLEOCR_ONNX_INCLUDE_OCR_HTTP_SERVER_H_
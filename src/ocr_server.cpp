// Copyright (c) 2024 PaddleOCR ONNX Runtime Project
// Licensed under the Apache License, Version 2.0

#include <iostream>
#include <string>
#include <csignal>
#include <cxxopts.hpp>

#include "ocr_http_server.h"

using namespace paddleocr::onnx;

// Global server instance for signal handling
std::unique_ptr<OCRHttpServer> g_server;

// ============================================
// Signal handler
// ============================================

void SignalHandler(int signal) {
    std::cout << "\nReceived signal " << signal << ", shutting down..." << std::endl;
    if (g_server) {
        g_server->Stop();
    }
}

// ============================================
// Command line options parsing
// ============================================

struct ServerOptions {
    std::string det_model_path;
    std::string rec_model_path;
    std::string dict_path;
    std::string host = "0.0.0.0";
    int port = 8080;
    bool use_gpu = false;
    int gpu_id = 0;
    int num_threads = 4;
    float det_db_thresh = 0.3f;
    float det_db_box_thresh = 0.6f;
    int rec_batch_size = 6;
    int max_queue_size = 100;
    bool verbose = false;
    bool print_version = false;
};

bool ParseCommandLine(int argc, char* argv[], ServerOptions& opts) {
    try {
        cxxopts::Options options("ocr_server", "PaddleOCR ONNX Runtime HTTP Server");
        
        options.add_options()
            ("d,detector", "Detection model path", cxxopts::value<std::string>())
            ("r,recognizer", "Recognition model path", cxxopts::value<std::string>())
            ("dict", "Dictionary file path", cxxopts::value<std::string>())
            ("host", "Server host address", cxxopts::value<std::string>()->default_value("0.0.0.0"))
            ("p,port", "Server port", cxxopts::value<int>()->default_value("8080"))
            ("gpu", "Enable GPU inference", cxxopts::value<bool>()->default_value("false"))
            ("gpu-id", "GPU device ID", cxxopts::value<int>()->default_value("0"))
            ("threads", "Number of CPU threads", cxxopts::value<int>()->default_value("4"))
            ("det-thresh", "Detection threshold", cxxopts::value<float>()->default_value("0.3"))
            ("det-box-thresh", "Detection box threshold", cxxopts::value<float>()->default_value("0.6"))
            ("rec-batch-size", "Recognition batch size", cxxopts::value<int>()->default_value("6"))
            ("max-queue", "Maximum request queue size", cxxopts::value<int>()->default_value("100"))
            ("v,verbose", "Verbose output", cxxopts::value<bool>()->default_value("false"))
            ("version", "Print version information")
            ("h,help", "Print help");
        
        auto result = options.parse(argc, argv);
        
        if (result.count("help")) {
            std::cout << options.help() << std::endl;
            return false;
        }

        if (result.count("version")) {
            opts.print_version = true;
            return true;
        }
        
        // Required arguments
        if (!result.count("detector") || !result.count("recognizer") || !result.count("dict")) {
            std::cerr << "Error: Missing required arguments" << std::endl;
            std::cout << options.help() << std::endl;
            return false;
        }
        
        opts.det_model_path = result["detector"].as<std::string>();
        opts.rec_model_path = result["recognizer"].as<std::string>();
        opts.dict_path = result["dict"].as<std::string>();
        opts.host = result["host"].as<std::string>();
        opts.port = result["port"].as<int>();
        opts.use_gpu = result["gpu"].as<bool>();
        opts.gpu_id = result["gpu-id"].as<int>();
        opts.num_threads = result["threads"].as<int>();
        opts.det_db_thresh = result["det-thresh"].as<float>();
        opts.det_db_box_thresh = result["det-box-thresh"].as<float>();
        opts.rec_batch_size = result["rec-batch-size"].as<int>();
        opts.max_queue_size = result["max-queue"].as<int>();
        opts.verbose = result["verbose"].as<bool>();
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error parsing arguments: " << e.what() << std::endl;
        return false;
    }
}

// ============================================
// Main function
// ============================================

int main(int argc, char* argv[]) {
    // Parse command line options
    ServerOptions opts;
    if (!ParseCommandLine(argc, argv, opts)) {
        return 1;
    }

    if (opts.print_version) {
        std::cout << "PaddleOCR ONNX HTTP Server " << PROJECT_VERSION
                  << " (git: " << GIT_COMMIT_HASH << ")" << std::endl;
        return 0;
    }

    // Setup signal handlers
    std::signal(SIGINT, SignalHandler);
    std::signal(SIGTERM, SignalHandler);

    // Print header
    std::cout << "=== PaddleOCR ONNX HTTP Server ===" << std::endl;
    std::cout << "Version: " << PROJECT_VERSION << " (git: " << GIT_COMMIT_HASH << ")" << std::endl;
    std::cout << "Host: " << opts.host << std::endl;
    std::cout << "Port: " << opts.port << std::endl;
    std::cout << "Detection Model: " << opts.det_model_path << std::endl;
    std::cout << "Recognition Model: " << opts.rec_model_path << std::endl;
    std::cout << "Dictionary: " << opts.dict_path << std::endl;
    std::cout << "GPU: " << (opts.use_gpu ? "enabled" : "disabled") << std::endl;
    std::cout << "Threads: " << opts.num_threads << std::endl;
    std::cout << "========================================" << std::endl << std::endl;
    
    // Configure OCR pipeline
    OCRConfig config;
    config.det_model_path = opts.det_model_path;
    config.rec_model_path = opts.rec_model_path;
    config.dict_path = opts.dict_path;
    config.use_gpu = opts.use_gpu;
    config.gpu_id = opts.gpu_id;
    config.num_threads = opts.num_threads;
    config.verbose = opts.verbose;
    config.detector.db_thresh = opts.det_db_thresh;
    config.detector.db_box_thresh = opts.det_db_box_thresh;
    config.recognizer.batch_size = opts.rec_batch_size;
    
    // Initialize OCR pipeline
    std::cout << "Initializing models..." << std::endl;
    auto init_start = std::chrono::high_resolution_clock::now();
    
    auto pipeline = std::make_shared<OCRPipeline>();
    if (!pipeline->Init(config)) {
        std::cerr << "Error: Failed to initialize OCR pipeline" << std::endl;
        return 1;
    }
    
    auto init_end = std::chrono::high_resolution_clock::now();
    auto init_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        init_end - init_start).count();
    std::cout << "[INFO] Detection model loaded" << std::endl;
    std::cout << "[INFO] Recognition model loaded" << std::endl;
    std::cout << "[INFO] Dictionary loaded: " 
              << pipeline->GetRecognizer().GetDictionary().size() << " labels" << std::endl;
    std::cout << "[INFO] Models initialized in " << init_ms << " ms" << std::endl;
    
    // Configure HTTP server
    HttpServerConfig server_config;
    server_config.host = opts.host;
    server_config.port = opts.port;
    server_config.num_threads = opts.num_threads;
    server_config.max_queue_size = opts.max_queue_size;
    server_config.enable_cors = true;
    server_config.enable_health_check = true;
    
    // Initialize and start server
    g_server = std::make_unique<OCRHttpServer>();
    if (!g_server->Init(pipeline, server_config)) {
        std::cerr << "Error: Failed to initialize HTTP server" << std::endl;
        return 1;
    }
    
    std::cout << "\nServer is running at http://" << opts.host << ":" << opts.port << std::endl;
    std::cout << "Press Ctrl+C to stop..." << std::endl << std::endl;
    
    // Start server (blocking)
    if (!g_server->Start(true)) {
        std::cerr << "Error: Failed to start HTTP server" << std::endl;
        return 1;
    }
    
    std::cout << "Server stopped." << std::endl;
    
    return 0;
}
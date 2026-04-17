// Copyright (c) 2024 PaddleOCR ONNX Runtime Project
// Licensed under the Apache License, Version 2.0

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <cxxopts.hpp>

#include <opencv2/imgcodecs.hpp>
#include "ocr_pipeline.h"

using namespace paddleocr::onnx;

// ============================================
// Command line options parsing
// ============================================

struct CommandLineOptions {
    std::string image_path;
    std::string det_model_path;
    std::string rec_model_path;
    std::string dict_path;
    std::string output_path;
    std::string json_path;
    bool use_gpu = false;
    int gpu_id = 0;
    int num_threads = 4;
    float det_db_thresh = 0.3f;
    float det_db_box_thresh = 0.6f;
    float det_db_unclip_ratio = 1.5f;
    int rec_batch_size = 6;
    bool verbose = false;
};

bool ParseCommandLine(int argc, char* argv[], CommandLineOptions& opts) {
    try {
        cxxopts::Options options("ocr_cli", "PaddleOCR ONNX Runtime Command Line Tool");
        
        options.add_options()
            ("i,input", "Input image path", cxxopts::value<std::string>())
            ("d,detector", "Detection model path", cxxopts::value<std::string>())
            ("r,recognizer", "Recognition model path", cxxopts::value<std::string>())
            ("dict", "Dictionary file path", cxxopts::value<std::string>())
            ("o,output", "Output visualization image path", cxxopts::value<std::string>()->default_value(""))
            ("json", "Output JSON result path", cxxopts::value<std::string>()->default_value(""))
            ("gpu", "Enable GPU inference", cxxopts::value<bool>()->default_value("false"))
            ("gpu-id", "GPU device ID", cxxopts::value<int>()->default_value("0"))
            ("threads", "Number of CPU threads", cxxopts::value<int>()->default_value("4"))
            ("det-thresh", "Detection threshold", cxxopts::value<float>()->default_value("0.3"))
            ("det-box-thresh", "Detection box threshold", cxxopts::value<float>()->default_value("0.6"))
            ("det-unclip-ratio", "Detection unclip ratio", cxxopts::value<float>()->default_value("1.5"))
            ("rec-batch-size", "Recognition batch size", cxxopts::value<int>()->default_value("6"))
            ("v,verbose", "Verbose output", cxxopts::value<bool>()->default_value("false"))
            ("h,help", "Print help");
        
        auto result = options.parse(argc, argv);
        
        if (result.count("help")) {
            std::cout << options.help() << std::endl;
            return false;
        }
        
        // Required arguments
        if (!result.count("input") || !result.count("detector") ||
            !result.count("recognizer") || !result.count("dict")) {
            std::cerr << "Error: Missing required arguments" << std::endl;
            std::cout << options.help() << std::endl;
            return false;
        }
        
        opts.image_path = result["input"].as<std::string>();
        opts.det_model_path = result["detector"].as<std::string>();
        opts.rec_model_path = result["recognizer"].as<std::string>();
        opts.dict_path = result["dict"].as<std::string>();
        opts.output_path = result["output"].as<std::string>();
        opts.json_path = result["json"].as<std::string>();
        opts.use_gpu = result["gpu"].as<bool>();
        opts.gpu_id = result["gpu-id"].as<int>();
        opts.num_threads = result["threads"].as<int>();
        opts.det_db_thresh = result["det-thresh"].as<float>();
        opts.det_db_box_thresh = result["det-box-thresh"].as<float>();
        opts.det_db_unclip_ratio = result["det-unclip-ratio"].as<float>();
        opts.rec_batch_size = result["rec-batch-size"].as<int>();
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
    CommandLineOptions opts;
    if (!ParseCommandLine(argc, argv, opts)) {
        return 1;
    }
    
    // Print header
    std::cout << "=== PaddleOCR ONNX Runtime ===" << std::endl;
    std::cout << "Image: " << opts.image_path << std::endl;
    std::cout << "Detection Model: " << opts.det_model_path << std::endl;
    std::cout << "Recognition Model: " << opts.rec_model_path << std::endl;
    std::cout << "Dictionary: " << opts.dict_path << std::endl;
    std::cout << "GPU: " << (opts.use_gpu ? "enabled" : "disabled") << std::endl;
    std::cout << "Threads: " << opts.num_threads << std::endl;
    std::cout << "================================" << std::endl << std::endl;
    
    // Load image
    std::cout << "Loading image..." << std::endl;
    cv::Mat image = cv::imread(opts.image_path);
    if (image.empty()) {
        std::cerr << "Error: Failed to load image: " << opts.image_path << std::endl;
        return 1;
    }
    std::cout << "Image size: " << image.cols << "x" << image.rows << std::endl;
    
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
    config.detector.db_unclip_ratio = opts.det_db_unclip_ratio;
    config.recognizer.batch_size = opts.rec_batch_size;
    
    // Initialize pipeline
    std::cout << "\nInitializing models..." << std::endl;
    auto init_start = std::chrono::high_resolution_clock::now();
    
    OCRPipeline pipeline;
    if (!pipeline.Init(config)) {
        std::cerr << "Error: Failed to initialize OCR pipeline" << std::endl;
        return 1;
    }
    
    auto init_end = std::chrono::high_resolution_clock::now();
    auto init_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        init_end - init_start).count();
    std::cout << "Models initialized in " << init_ms << " ms" << std::endl;
    
    // Run OCR
    std::cout << "\nRunning OCR..." << std::endl;
    PerfStats stats;
    auto results = pipeline.Run(image, stats);
    
    // Print results
    std::cout << "\n=== Results (" << results.size() << " text regions found) ===" << std::endl;
    
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& result = results[i];
        std::cout << "[" << std::setw(2) << (i + 1) << "] ";
        std::cout << "\"" << result.text << "\" ";
        std::cout << "(conf: " << std::fixed << std::setprecision(3) << result.confidence << ") ";
        std::cout << "[det: " << result.det_confidence << "]" << std::endl;
    }
    
    // Print performance
    std::cout << "\n=== Performance ===" << std::endl;
    std::cout << "Total time: " << stats.total_ms << " ms" << std::endl;
    std::cout << "  - Preprocess: " << stats.preprocess_ms << " ms" << std::endl;
    std::cout << "  - Inference: " << stats.inference_ms << " ms" << std::endl;
    std::cout << "  - Postprocess: " << stats.postprocess_ms << " ms" << std::endl;
    
    // Save visualization
    if (!opts.output_path.empty()) {
        cv::Mat vis = pipeline.Visualize(image, results);
        cv::imwrite(opts.output_path, vis);
        std::cout << "\nVisualization saved to: " << opts.output_path << std::endl;
    }
    
    // Save JSON
    if (!opts.json_path.empty()) {
        std::string json_str = pipeline.ExportToJson(results, opts.image_path,
                                                      cv::Size(image.cols, image.rows),
                                                      stats);
        std::ofstream json_file(opts.json_path);
        if (json_file.is_open()) {
            json_file << json_str;
            json_file.close();
            std::cout << "Results saved to JSON: " << opts.json_path << std::endl;
        }
    }
    
    return 0;
}
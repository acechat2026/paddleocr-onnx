// Copyright (c) 2024 PaddleOCR ONNX Runtime Project
// Licensed under the Apache License, Version 2.0

#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <cxxopts.hpp>

#include <opencv2/imgcodecs.hpp>
#include "ocr_pipeline.h"

using namespace paddleocr::onnx;

struct BenchmarkOptions {
    std::string image_path;
    std::string det_model_path;
    std::string rec_model_path;
    std::string dict_path;
    int num_iterations = 10;
    bool use_gpu = false;
    int gpu_id = 0;
    int num_threads = 4;
    bool verbose = false;
};

bool ParseCommandLine(int argc, char* argv[], BenchmarkOptions& opts) {
    try {
        cxxopts::Options options("ocr_benchmark", "PaddleOCR ONNX Runtime Benchmark Tool");

        options.add_options()
            ("i,input", "Input image path", cxxopts::value<std::string>())
            ("d,detector", "Detection model path", cxxopts::value<std::string>())
            ("r,recognizer", "Recognition model path", cxxopts::value<std::string>())
            ("dict", "Dictionary file path", cxxopts::value<std::string>())
            ("n,iterations", "Number of iterations", cxxopts::value<int>()->default_value("10"))
            ("gpu", "Enable GPU inference", cxxopts::value<bool>()->default_value("false"))
            ("gpu-id", "GPU device ID", cxxopts::value<int>()->default_value("0"))
            ("threads", "Number of CPU threads", cxxopts::value<int>()->default_value("4"))
            ("v,verbose", "Verbose output", cxxopts::value<bool>()->default_value("false"))
            ("version", "Print version information")
            ("h,help", "Print help");

        auto result = options.parse(argc, argv);

        if (result.count("help")) {
            std::cout << options.help() << std::endl;
            return false;
        }

        if (result.count("version")) {
            std::cout << "PaddleOCR ONNX Benchmark " << PROJECT_VERSION
                      << " (git: " << GIT_COMMIT_HASH << ")" << std::endl;
            return false;
        }

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
        opts.num_iterations = result["iterations"].as<int>();
        opts.use_gpu = result["gpu"].as<bool>();
        opts.gpu_id = result["gpu-id"].as<int>();
        opts.num_threads = result["threads"].as<int>();
        opts.verbose = result["verbose"].as<bool>();

        return true;

    } catch (const std::exception& e) {
        std::cerr << "Error parsing arguments: " << e.what() << std::endl;
        return false;
    }
}

int main(int argc, char* argv[]) {
    BenchmarkOptions opts;
    if (!ParseCommandLine(argc, argv, opts)) {
        return 1;
    }

    std::cout << "=== PaddleOCR ONNX Benchmark ===" << std::endl;
    std::cout << "Version: " << PROJECT_VERSION << " (git: " << GIT_COMMIT_HASH << ")" << std::endl;
    std::cout << "Image: " << opts.image_path << std::endl;
    std::cout << "Iterations: " << opts.num_iterations << std::endl;
    std::cout << "=================================" << std::endl << std::endl;

    cv::Mat image = cv::imread(opts.image_path);
    if (image.empty()) {
        std::cerr << "Error: Failed to load image: " << opts.image_path << std::endl;
        return 1;
    }

    OCRConfig config;
    config.det_model_path = opts.det_model_path;
    config.rec_model_path = opts.rec_model_path;
    config.dict_path = opts.dict_path;
    config.use_gpu = opts.use_gpu;
    config.gpu_id = opts.gpu_id;
    config.num_threads = opts.num_threads;
    config.verbose = opts.verbose;

    OCRPipeline pipeline;
    if (!pipeline.Init(config)) {
        std::cerr << "Error: Failed to initialize OCR pipeline" << std::endl;
        return 1;
    }

    // Warmup
    std::cout << "Running warmup..." << std::endl;
    PerfStats warmup_stats;
    pipeline.Run(image, warmup_stats);

    // Benchmark
    std::cout << "Running benchmark..." << std::endl;
    std::vector<double> total_times;
    std::vector<double> preprocess_times;
    std::vector<double> inference_times;
    std::vector<double> postprocess_times;
    std::vector<int> region_counts;

    total_times.reserve(opts.num_iterations);
    preprocess_times.reserve(opts.num_iterations);
    inference_times.reserve(opts.num_iterations);
    postprocess_times.reserve(opts.num_iterations);
    region_counts.reserve(opts.num_iterations);

    for (int i = 0; i < opts.num_iterations; ++i) {
        PerfStats stats;
        auto results = pipeline.Run(image, stats);

        total_times.push_back(stats.total_ms);
        preprocess_times.push_back(stats.preprocess_ms);
        inference_times.push_back(stats.inference_ms);
        postprocess_times.push_back(stats.postprocess_ms);
        region_counts.push_back(stats.num_text_regions);
    }

    auto mean = [](const std::vector<double>& v) -> double {
        if (v.empty()) return 0.0;
        return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    };

    auto median = [](std::vector<double> v) -> double {
        if (v.empty()) return 0.0;
        size_t n = v.size() / 2;
        std::nth_element(v.begin(), v.begin() + n, v.end());
        return v[n];
    };

    auto minmax = [](const std::vector<double>& v) -> std::pair<double, double> {
        if (v.empty()) return {0.0, 0.0};
        auto [min_it, max_it] = std::minmax_element(v.begin(), v.end());
        return {*min_it, *max_it};
    };

    std::cout << "\n=== Benchmark Results ===" << std::endl;
    std::cout << std::fixed << std::setprecision(2);

    auto print_stat = [&](const std::string& name, const std::vector<double>& times) {
        auto [min_val, max_val] = minmax(times);
        std::cout << name << ":" << std::endl;
        std::cout << "  Mean:   " << mean(times) << " ms" << std::endl;
        std::cout << "  Median: " << median(times) << " ms" << std::endl;
        std::cout << "  Min:    " << min_val << " ms" << std::endl;
        std::cout << "  Max:    " << max_val << " ms" << std::endl;
    };

    print_stat("Total", total_times);
    print_stat("Preprocess", preprocess_times);
    print_stat("Inference", inference_times);
    print_stat("Postprocess", postprocess_times);

    double avg_regions = mean(std::vector<double>(region_counts.begin(), region_counts.end()));
    std::cout << "Avg text regions: " << avg_regions << std::endl;

    return 0;
}

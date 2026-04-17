# ============================================
# PaddleOCR ONNX Runtime C++ Deployment
# CMake Build Configuration
# ============================================

cmake_minimum_required(VERSION 3.15)
project(paddleocr_onnx 
    VERSION 1.0.0
    DESCRIPTION "Lightweight PaddleOCR ONNX Runtime C++ deployment"
    LANGUAGES CXX
)

# ============================================
# Compiler Options
# ============================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set default build type to Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Generate compile_commands.json for IDE intellisense
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ============================================
# Project Options
# ============================================
option(BUILD_HTTP_SERVER "Build HTTP server" ON)
option(USE_CUDA "Enable CUDA GPU acceleration" OFF)
option(BUILD_TESTS "Build unit tests" OFF)
option(BUILD_BENCHMARK "Build benchmark tool" OFF)

# ============================================
# Output Directories
# ============================================
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# ============================================
# Platform Detection
# ============================================
if(WIN32)
    set(PLATFORM_WINDOWS TRUE)
    add_definitions(-DPLATFORM_WINDOWS)
    add_definitions(-DNOMINMAX)  # Avoid Windows.h min/max macros
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
elseif(APPLE)
    set(PLATFORM_MACOS TRUE)
    add_definitions(-DPLATFORM_MACOS)
elseif(UNIX)
    set(PLATFORM_LINUX TRUE)
    add_definitions(-DPLATFORM_LINUX)
endif()

message(STATUS "Platform: ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# ============================================
# Third-party Dependencies Directory
# ============================================
set(THIRDPARTY_DIR ${CMAKE_SOURCE_DIR}/3rdparty CACHE PATH 
    "Third-party dependencies directory")

# ============================================
# Check Header-Only Dependencies
# ============================================

# 1. cxxopts - Command line parser (Required)
set(CXXOPTS_HEADER ${THIRDPARTY_DIR}/cxxopts.hpp)
if(NOT EXISTS ${CXXOPTS_HEADER})
    message(FATAL_ERROR 
        "cxxopts.hpp not found at ${CXXOPTS_HEADER}\n"
        "Please download it from: https://github.com/jarro2783/cxxopts\n"
        "Or run: ./scripts/setup_deps.sh")
endif()
message(STATUS "Found cxxopts: ${CXXOPTS_HEADER}")

# 2. cpp-httplib - HTTP server library (Required only if BUILD_HTTP_SERVER=ON)
if(BUILD_HTTP_SERVER)
    set(HTTPLIB_HEADER ${THIRDPARTY_DIR}/cpp-httplib/httplib.h)
    if(NOT EXISTS ${HTTPLIB_HEADER})
        message(WARNING 
            "httplib.h not found at ${HTTPLIB_HEADER}\n"
            "HTTP server will be disabled.\n"
            "Please download it from: https://github.com/yhirose/cpp-httplib")
        set(BUILD_HTTP_SERVER OFF)
    else()
        message(STATUS "Found cpp-httplib: ${HTTPLIB_HEADER}")
        find_package(Threads REQUIRED)
    endif()
endif()

# ============================================
# Find ONNX Runtime
# ============================================
set(ONNXRUNTIME_ROOT ${THIRDPARTY_DIR}/onnxruntime CACHE PATH 
    "ONNX Runtime root directory")

# Find header files
find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_c_api.h
    PATHS 
        ${ONNXRUNTIME_ROOT}/include
        ${ONNXRUNTIME_ROOT}
    NO_DEFAULT_PATH
)

# Find library files (platform-specific)
if(WIN32)
    find_library(ONNXRUNTIME_LIBRARY
        NAMES onnxruntime
        PATHS 
            ${ONNXRUNTIME_ROOT}/lib
            ${ONNXRUNTIME_ROOT}
        NO_DEFAULT_PATH
    )
    # Windows also needs the DLL
    find_file(ONNXRUNTIME_DLL
        NAMES onnxruntime.dll
        PATHS 
            ${ONNXRUNTIME_ROOT}/bin
            ${ONNXRUNTIME_ROOT}/lib
            ${ONNXRUNTIME_ROOT}
        NO_DEFAULT_PATH
    )
elseif(APPLE)
    find_library(ONNXRUNTIME_LIBRARY
        NAMES libonnxruntime.dylib onnxruntime
        PATHS 
            ${ONNXRUNTIME_ROOT}/lib
            ${ONNXRUNTIME_ROOT}
        NO_DEFAULT_PATH
    )
else()
    find_library(ONNXRUNTIME_LIBRARY
        NAMES libonnxruntime.so onnxruntime
        PATHS 
            ${ONNXRUNTIME_ROOT}/lib
            ${ONNXRUNTIME_ROOT}
        NO_DEFAULT_PATH
    )
endif()

# Validate ONNX Runtime
if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR 
        "ONNX Runtime not found in ${ONNXRUNTIME_ROOT}\n"
        "Please run: ./scripts/setup_deps.sh (or .bat)")
endif()

message(STATUS "Found ONNX Runtime:")
message(STATUS "  Include: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "  Library: ${ONNXRUNTIME_LIBRARY}")
if(ONNXRUNTIME_DLL)
    message(STATUS "  DLL: ${ONNXRUNTIME_DLL}")
endif()

# ============================================
# Find OpenCV
# ============================================
set(OpenCV_DIR ${THIRDPARTY_DIR}/opencv/build/install/lib/cmake/opencv4 CACHE PATH
    "OpenCV config directory")

# Try local OpenCV first
find_package(OpenCV QUIET
    PATHS
        ${OpenCV_DIR}
        ${THIRDPARTY_DIR}/opencv/build/install/lib/cmake/opencv4
        ${THIRDPARTY_DIR}/opencv/build/x64/vc15/lib
        ${THIRDPARTY_DIR}/opencv/share/OpenCV
    NO_DEFAULT_PATH
)

# Fall back to system OpenCV
if(NOT OpenCV_FOUND)
    find_package(OpenCV REQUIRED
        HINTS
            /usr/local
            /opt/homebrew/opt/opencv
            /usr
            /opt/local
    )
endif()

if(NOT OpenCV_FOUND)
    message(FATAL_ERROR 
        "OpenCV not found!\n"
        "Please run: ./scripts/setup_deps.sh (or .bat)")
endif()

message(STATUS "Found OpenCV: ${OpenCV_VERSION}")
message(STATUS "  Include: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "  Libraries: ${OpenCV_LIBS}")

# ============================================
# Compiler-specific Options
# ============================================
if(MSVC)
    # Visual Studio compiler options
    add_compile_options(/W3)           # Warning level 3
    add_compile_options(/MP)           # Multi-processor compilation
    add_compile_options(/utf-8)        # UTF-8 source encoding
    add_compile_options(/bigobj)       # Support large object files
    
    # Release optimization flags
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /Ob2 /Oi /Ot")
else()
    # GCC/Clang compiler options
    add_compile_options(-Wall -Wextra)
    add_compile_options(-Wno-unused-parameter)
    add_compile_options(-Wno-sign-compare)
    
    # Release optimization flags
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -mtune=native")
    
    # Debug flags
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
    
    # Linker flags for Linux
    if(PLATFORM_LINUX)
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pthread")
    endif()
endif()

# ============================================
# CUDA Support (Optional)
# ============================================
if(USE_CUDA)
    find_package(CUDA QUIET)
    if(CUDA_FOUND)
        message(STATUS "Found CUDA: ${CUDA_VERSION}")
        add_definitions(-DUSE_CUDA)
    else()
        message(WARNING "CUDA not found, falling back to CPU")
        set(USE_CUDA OFF)
    endif()
endif()

# ============================================
# Source Files
# ============================================
set(SOURCES
    src/ocr_common.cpp
    src/ocr_detector.cpp
    src/ocr_recognizer.cpp
    src/ocr_pipeline.cpp
)

set(HEADERS
    include/ocr_common.h
    include/ocr_detector.h
    include/ocr_recognizer.h
    include/ocr_pipeline.h
)

# ============================================
# Common Include Directories
# ============================================
set(COMMON_INCLUDES
    ${CMAKE_SOURCE_DIR}/include
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${THIRDPARTY_DIR}                    # For cxxopts.hpp
)

if(BUILD_HTTP_SERVER)
    list(APPEND COMMON_INCLUDES ${THIRDPARTY_DIR}/cpp-httplib)
endif()

# ============================================
# Command Line Tool (ocr_cli)
# ============================================
add_executable(ocr_cli
    src/ocr_cli.cpp
    ${SOURCES}
    ${HEADERS}
)

set_target_properties(ocr_cli PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    OUTPUT_NAME ocr_cli
)

target_include_directories(ocr_cli PRIVATE ${COMMON_INCLUDES})
target_link_libraries(ocr_cli
    ${ONNXRUNTIME_LIBRARY}
    ${OpenCV_LIBS}
)

# Windows-specific: Copy DLLs to output directory
if(WIN32 AND ONNXRUNTIME_DLL)
    add_custom_command(TARGET ocr_cli POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_DLL}"
            $<TARGET_FILE_DIR:ocr_cli>
        COMMENT "Copying onnxruntime.dll to output directory"
    )
endif()

# macOS-specific: Set runtime path
if(APPLE)
    set_target_properties(ocr_cli PROPERTIES
        INSTALL_RPATH "@executable_path/../lib"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()

# ============================================
# HTTP Server (ocr_server)
# ============================================
if(BUILD_HTTP_SERVER)
    add_executable(ocr_server
        src/ocr_server.cpp
        src/ocr_http_server.cpp
        ${SOURCES}
        ${HEADERS}
        include/ocr_http_server.h
    )
    
    set_target_properties(ocr_server PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        OUTPUT_NAME ocr_server
    )
    
    target_include_directories(ocr_server PRIVATE ${COMMON_INCLUDES})
    target_link_libraries(ocr_server
        ${ONNXRUNTIME_LIBRARY}
        ${OpenCV_LIBS}
        Threads::Threads
    )
    
    # Platform-specific libraries for cpp-httplib
    if(WIN32)
        target_link_libraries(ocr_server ws2_32 crypt32)
        
        # Copy DLL for Windows
        if(ONNXRUNTIME_DLL)
            add_custom_command(TARGET ocr_server POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${ONNXRUNTIME_DLL}"
                    $<TARGET_FILE_DIR:ocr_server>
                COMMENT "Copying onnxruntime.dll to output directory"
            )
        endif()
    elseif(APPLE)
        target_link_libraries(ocr_server 
            "-framework CoreFoundation"
            "-framework Security"
        )
    endif()
    
    message(STATUS "HTTP server will be built")
endif()

# ============================================
# Unit Tests (Optional)
# ============================================
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ============================================
# Benchmark Tool (Optional)
# ============================================
if(BUILD_BENCHMARK)
    add_executable(ocr_benchmark
        src/ocr_benchmark.cpp
        ${SOURCES}
    )
    
    target_include_directories(ocr_benchmark PRIVATE ${COMMON_INCLUDES})
    target_link_libraries(ocr_benchmark
        ${ONNXRUNTIME_LIBRARY}
        ${OpenCV_LIBS}
    )
endif()

# ============================================
# Installation Rules
# ============================================
install(TARGETS ocr_cli
    RUNTIME DESTINATION bin
)

if(BUILD_HTTP_SERVER)
    install(TARGETS ocr_server
        RUNTIME DESTINATION bin
    )
endif()

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)

# ============================================
# Print Configuration Summary
# ============================================
message(STATUS "")
message(STATUS "=========================================")
message(STATUS "PaddleOCR ONNX Configuration Summary")
message(STATUS "=========================================")
message(STATUS "Version:          ${PROJECT_VERSION}")
message(STATUS "Build type:       ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard:     ${CMAKE_CXX_STANDARD}")
message(STATUS "CUDA enabled:     ${USE_CUDA}")
message(STATUS "HTTP Server:      ${BUILD_HTTP_SERVER}")
message(STATUS "Build tests:      ${BUILD_TESTS}")
message(STATUS "Build benchmark:  ${BUILD_BENCHMARK}")
message(STATUS "=========================================")
message(STATUS "")
message(STATUS "To build the project:")
message(STATUS "  cmake --build . --config ${CMAKE_BUILD_TYPE}")
message(STATUS "  # or make -j$(nproc) on Linux/macOS")
message(STATUS "")
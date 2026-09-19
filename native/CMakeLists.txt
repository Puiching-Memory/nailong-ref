cmake_minimum_required(VERSION 3.20)
project(nailong_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(MSVC)
    add_compile_options(/utf-8 /W4 /permissive- /Zc:__cplusplus /EHsc /MP)
    add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
else()
    add_compile_options(-Wall -Wextra)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# ---------------------------------------------------------------------------
# 依赖：TensorRT（官方 Windows x64 zip，解压到 third_party/TensorRT-* 下）
#
# 走 TensorRT C++ API，不用 ONNX Runtime：社区项目（GPT-SoVITS_minimal_inference
# 的 run_trt_inference.py）把全部 8 个模型都建为 .engine 直接跑，已验证可行。
# 好处是摆脱 ORT 与 TRT 的版本硬绑定，且能自己掌控 AR 循环的显存/流/零拷贝。
#
# 用官方自带的 CMake config，不手写 find 模块：它把 DLL（bin/）与导入库（lib/）
# 分得很清楚，并把按架构拆分的 builder resource 也建模成了一组独立 target
# （TRT::tensorrt_builder_resource_sm89 等）——这正是决定分发体积的东西。
#
# 下载（需接受许可）：https://developer.nvidia.com/tensorrt/download/11x
#   选 11.3.0 → Windows x64 → Zip，且必须是 CUDA 13.x 那个变体
#   （官方支持矩阵 f10：Built with CUDA Toolkit 13.4, compatible with CUDA 13.x only）
# ---------------------------------------------------------------------------
file(GLOB _trt_cmake_dirs "${CMAKE_CURRENT_SOURCE_DIR}/third_party/TensorRT-*/cmake")
list(SORT _trt_cmake_dirs)
list(REVERSE _trt_cmake_dirs)
find_package(TensorRT-Enterprise CONFIG REQUIRED
    PATHS ${_trt_cmake_dirs}
    NO_DEFAULT_PATH)
message(STATUS "TensorRT ${TensorRT_VERSION} (CUDA ${TensorRT_CUDA_VERSION}) @ ${TensorRT_DIR}")

# ---------------------------------------------------------------------------
# 目标
# ---------------------------------------------------------------------------
add_subdirectory(tools)

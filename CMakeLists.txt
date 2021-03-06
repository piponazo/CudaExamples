# https://devblogs.nvidia.com/building-cuda-applications-cmake/

cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(CudaExamples
    LANGUAGES CXX CUDA
    VERSION 0.0.1
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

include(cmake_stuff/findDependencies.cmake      REQUIRED)
include(cmake_stuff/compileroptions.cmake       REQUIRED)

add_subdirectory(lib)
add_subdirectory(apps)

#include(cmake_stuff/packageGenerator.cmake      REQUIRED)
include(cmake_stuff/printInfo.cmake             REQUIRED)

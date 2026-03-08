cmake_minimum_required(VERSION 3.20)

project(hf_diffusion LANGUAGES CXX)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

if(NOT DEFINED PYBIND11_INCLUDE_DIR)
  message(FATAL_ERROR "PYBIND11_INCLUDE_DIR must be provided")
endif()

if(NOT DEFINED OUTPUT_DIR)
  message(FATAL_ERROR "OUTPUT_DIR must be provided")
endif()

if(APPLE)
  find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
else()
  find_library(LAPACK_LIBRARY NAMES openblas lapack REQUIRED)
endif()

Python_add_library(_hf_diffusion_native MODULE diffusion.cpp WITH_SOABI)
target_compile_features(_hf_diffusion_native PRIVATE cxx_std_17)
target_compile_options(_hf_diffusion_native PRIVATE -O3)
target_include_directories(_hf_diffusion_native PRIVATE "${PYBIND11_INCLUDE_DIR}")
if(APPLE)
  target_link_libraries(_hf_diffusion_native PRIVATE "${ACCELERATE_FRAMEWORK}")
else()
  target_link_libraries(_hf_diffusion_native PRIVATE "${LAPACK_LIBRARY}")
endif()
set_target_properties(
  _hf_diffusion_native
  PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
)

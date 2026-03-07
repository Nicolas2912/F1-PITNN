cmake_minimum_required(VERSION 3.20)

project(hf_simulator_kernels LANGUAGES CXX)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

if(NOT DEFINED PYBIND11_INCLUDE_DIR)
  message(FATAL_ERROR "PYBIND11_INCLUDE_DIR must be provided")
endif()

if(NOT DEFINED OUTPUT_DIR)
  message(FATAL_ERROR "OUTPUT_DIR must be provided")
endif()

Python_add_library(_hf_simulator_native MODULE simulator_kernels.cpp WITH_SOABI)
target_compile_features(_hf_simulator_native PRIVATE cxx_std_17)
target_compile_options(_hf_simulator_native PRIVATE -O3)
target_include_directories(_hf_simulator_native PRIVATE "${PYBIND11_INCLUDE_DIR}")
set_target_properties(
  _hf_simulator_native
  PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
)

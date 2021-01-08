cmake_minimum_required(VERSION 3.0)
project(NeuralNetwork)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

add_executable(NeuralNetwork src/main.cpp src/DenseLayer.cpp)
target_include_directories(NeuralNetwork PUBLIC "/usr/local/lib/eigen-3.3.7")

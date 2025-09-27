cmake_minimum_required(VERSION 3.16)
project(cpp_cuda)  # REMOVE CUDA for now - just regular C++

set(CMAKE_CXX_STANDARD 17)

add_executable(cpp_cuda main.cpp)  # Regular C++ file
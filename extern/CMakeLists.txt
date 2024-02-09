cmake_minimum_required(VERSION 3.5...3.26)
project(depth_utils LANGUAGES CXX)


add_subdirectory(pybind11)
pybind11_add_module(depth_utils depth_utils.cpp)
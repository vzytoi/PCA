cmake_minimum_required(VERSION 3.18)
project(pca LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(pca src/pca.cpp)

# Ensure the extension is installed into the wheel
install(TARGETS pca
  LIBRARY DESTINATION .
  RUNTIME DESTINATION .
)

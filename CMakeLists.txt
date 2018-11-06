cmake_minimum_required(VERSION 3.7)
project(PSGDC__)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp Test.cpp SGD.cpp DataSet.cpp Util.cpp)
add_executable(PSGDC__ ${SOURCE_FILES})
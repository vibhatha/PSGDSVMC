cmake_minimum_required(VERSION 3.7)
project(PSGDC__)

#add_compile_options(-std=c++11)
set(CMAKE_CXX_STANDARD 11)
#set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp Test.cpp SGD.cpp DataSet.cpp Util.cpp Initializer.cpp Matrix.cpp OptArgs.cpp ArgReader.cpp Predict.cpp ResourceManager.cpp)
add_executable(PSGDC__ ${SOURCE_FILES})
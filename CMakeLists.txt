cmake_minimum_required(VERSION 3.7)
project(PSGDC__)

# set the path to the library folder
link_directories(/home/vibhatha/tools/blass/build/lib/)

find_package(BLAS)


#add_compile_options(-std=c++11)
set(CMAKE_CXX_STANDARD 11)
#set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fopenmp -lcblas")

set(CMAKE_CXX_COMPILER /home/vibhatha/ds/openmpi-3/build/bin/mpic++)
set(SOURCE_FILES main.cpp Test.cpp SGD.cpp DataSet.cpp Util.cpp Initializer.cpp Matrix.cpp OptArgs.cpp ArgReader.cpp Predict.cpp ResourceManager.cpp PSGD.cpp Matrix1.cpp)
add_executable(PSGDC__ ${SOURCE_FILES})

# link the libraries to the executable


cmake_minimum_required(VERSION 2.8)

set(TARGET_NAME CaffeScoring)

# Get caffe locations
set(CAFFE_ROOT $ENV{CAFFE_ROOT})

set(NO_OPENCV TRUE)
# Set flags for compiler
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCPU_ONLY -DNO_OPENCV -std=c++11")

add_definitions(-std=c++11)

include_directories(src)
include_directories(${CAFFE_ROOT}/include ${CAFFE_ROOT}/build/src)

if(APPLE)
    include_directories("/usr/local/opt/openblas/include")
endif()

file(GLOB MAIN_SRC
        main_float.cpp
        )

add_executable(${TARGET_NAME} ${MAIN_SRC})

find_library(CAFFE_LIBRARY
        caffe
        HINTS
        ${CAFFE_ROOT}/build/lib)

target_link_libraries(${TARGET_NAME} ${GFLAGS_LIBRARIES} ${GLOG_LIBRARIES} ${CAFFE_LIBRARY} protobuf boost_system)
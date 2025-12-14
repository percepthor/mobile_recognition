cmake_minimum_required(VERSION 3.10)
project(image_recognition)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Source files
set(SOURCES
    src/image_recognition.cpp
    src/image_decode_stb.cpp
    src/image_preprocess.cpp
    src/softmax.cpp
    src/config_parse.cpp
    src/timing.cpp
    third_party/cJSON.c
)

# Create shared library
add_library(image_recognition SHARED ${SOURCES})

# Include directories
target_include_directories(image_recognition PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tflite_headers
)

# Compiler flags
target_compile_options(image_recognition PRIVATE
    -Wall
    -Wextra
    -O3
    -ffast-math
)

# Platform-specific configuration
if(ANDROID)
    target_link_libraries(image_recognition
        log
        android
    )
    # Note: libtensorflowlite_c.so will be loaded dynamically from jniLibs
elseif(APPLE)
    # iOS build
    set_target_properties(image_recognition PROPERTIES
        FRAMEWORK FALSE
        MACOSX_RPATH ON
    )
endif()

# Build type optimizations
if(CMAKE_BUILD_TYPE MATCHES Release)
    target_compile_definitions(image_recognition PRIVATE NDEBUG)
endif()

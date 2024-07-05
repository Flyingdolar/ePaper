# Set CMake Required Version
cmake_minimum_required(VERSION 3.5.0)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Collect all files in the directory
file(GLOB_RECURSE SOURCES "*.cpp")

# This is the subdirectory that will be added as a library
project(functions)
add_library(functions ${SOURCES})

# Get Function Names saved in the variable
foreach(SRC ${SOURCES})
    get_filename_component(SRC_NAME ${SRC} NAME_WE)
    list(APPEND FUNCTION_NAMES ${SRC_NAME})
endforeach()

message(STATUS "Creating Function Libraries: ${FUNCTION_NAMES}")

# Link to Libraries
target_link_libraries(functions
    ${OpenCV_LIBS}
    NumCpp::NumCpp
    cjson
)

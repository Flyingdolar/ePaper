cmake_minimum_required(VERSION 3.5.0)
project(IPP-Design LANGUAGES CXX)

# Setting C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenCV REQUIRED QUIET)
find_package(NumCpp REQUIRED QUIET)
find_package(cJSON REQUIRED QUIET)

message(STATUS "Found Packages: [OpenCV: ${OpenCV_VERSION}] [NumCpp: ${NumCpp_VERSION}] [cJSON: ${cJSON_VERSION}]")

# Copy image data to build directory
message(STATUS "Copy [data] to build directory")
file(COPY data DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
message(STATUS "Copy [image] to build directory")
file(COPY image DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# Main source files
file(GLOB_RECURSE PROJECT_SOURCES "source/*.cpp")
file(GLOB_RECURSE TEST_SOURCES "test/*.cpp")

# Functions library
add_subdirectory(functions)
target_include_directories(functions PUBLIC include)    # Functions include

# Set Include directories & Link libraries
include_directories(include ${OpenCV_INCLUDE_DIRS} ${NumCpp_INCLUDE_DIRS} ${cJSON_INCLUDE_DIRS})
link_libraries(${OpenCV_LIBS} NumCpp::NumCpp cjson functions)

# Get Project Names, Test Names saved in the variable
foreach(SRC ${PROJECT_SOURCES})
    get_filename_component(SRC_NAME ${SRC} NAME_WE)
    list(APPEND PROJECT_NAMES ${SRC_NAME})
endforeach()
foreach(TEST ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST} NAME_WE)
    list(APPEND TEST_NAMES ${TEST_NAME})
endforeach()

# Create executable files
message(STATUS "Creating Project Executables: ${PROJECT_NAMES}")
foreach(SRC ${PROJECT_SOURCES})
    get_filename_component(SRC_NAME ${SRC} NAME_WE)
    add_executable(${SRC_NAME} ${SRC})
endforeach()

# Create test files
enable_testing()
message(STATUS "Creating Test Executables: ${TEST_NAMES}")
foreach(TEST ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST} NAME_WE)
    add_executable(${TEST_NAME} ${TEST})
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    set_tests_properties(${TEST_NAME} PROPERTIES LABELS "ISP")
endforeach()

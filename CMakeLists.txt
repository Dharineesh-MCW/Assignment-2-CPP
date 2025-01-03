cmake_minimum_required(VERSION 3.10)
# Project name
project(Conv2DProject)
# Setting  C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(VCPKG_INCLUDE_DIRS "O:/Assignment2/Project_Root/vcpkg")

# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/Operators/include
    ${PROJECT_SOURCE_DIR}/Test_Operators/include
    ${PROJECT_SOURCE_DIR}/utilities/include
    ${PROJECT_SOURCE_DIR}/include
    ${VCPKG_INCLUDE_DIRS}/installed
    ${PROJECT_SOURCE_DIR}/.vscode
    ${PROJECT_SOURCE_DIR}/vcpkg/installed/x64-windows/include
    ${json_SOURCE_DIR}/include
)
# Source files
file(GLOB_RECURSE OPERATORS_SRC
    ${PROJECT_SOURCE_DIR}/Operators/src/*.cpp
)
file(GLOB_RECURSE TEST_OPERATORS_SRC
    ${PROJECT_SOURCE_DIR}/Test_Operators/src/*.cpp
)
file(GLOB_RECURSE UTILITIES_SRC
    ${PROJECT_SOURCE_DIR}/utilities/src/*.cpp
)
file(GLOB_RECURSE MAIN_SRC
    ${PROJECT_SOURCE_DIR}/src/*.cpp
)
# file(GLOB_RECURSE VCPKG
# )
file(GLOB_RECURSE NLOHMANN_JSON_HEADERS
    ${VCPKG_DIR}/installed/x64-windows/include/nlohmann/*.hpp
)
# Add executable
add_executable(Conv2DApp
    ${OPERATORS_SRC}
    ${TEST_OPERATORS_SRC}
    ${UTILITIES_SRC}
    ${MAIN_SRC}
)
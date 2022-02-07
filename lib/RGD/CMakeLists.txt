cmake_minimum_required(VERSION 3.0)
project(RGD)
SET(CMAKE_CXX_FLAGS "-std=c++0x")
add_library(RGD_core STATIC RGD_core.cpp)

#add_executable(test_first_derivative tests/test_first_derivative.cpp)
target_include_directories(RGD_core PUBLIC "${CMAKE_CURRENT_LIST_DIR}/include")
#target_include_directories(test_first_derivative PUBLIC ../catch2)
#target_link_libraries(test_first_derivative indicators_core)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
 
#add_executable (RGD_core RGD_core.cpp)
target_link_libraries (RGD_core Eigen3::Eigen)

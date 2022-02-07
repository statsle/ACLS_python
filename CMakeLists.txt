# CMakeLists.txt
cmake_minimum_required(VERSION 3.0)
project(ACLS)
SET(CMAKE_CXX_FLAGS "-std=c++0x")
add_subdirectory(lib/RGD)
add_subdirectory(lib/pybind11)
#add_subdirectory(lib/Eigen)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
 

set(SOURCE_DIR "src/ACLS")
set(SOURCES "${SOURCE_DIR}/RGD_bindings.cpp")
pybind11_add_module(RGD_bindings ${SOURCES})
target_link_libraries(RGD_bindings PRIVATE RGD_core)
#add_executable (RGD_core RGD_core.cpp)
#target_link_libraries (RGD_core Eigen3::Eigen)
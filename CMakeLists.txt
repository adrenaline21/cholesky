cmake_minimum_required(VERSION 3.18)

project(TCpotrs LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(CUDAToolkit REQUIRED)
add_library(cuda_target INTERFACE)
target_link_libraries(cuda_target INTERFACE CUDA::cudart CUDA::cublas)

find_package(LAPACK)

add_executable(NDtest src/NDtest.cpp src/utils.cpp src/sparse.cu)
target_link_libraries(NDtest ${LAPACK_LIBRARIES} cuda_target)
set_target_properties(NDtest PROPERTIES CUDA_ARCHITECTURES "75")

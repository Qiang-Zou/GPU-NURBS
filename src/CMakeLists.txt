# Make sure cuda is installed and in your shell path
# Use command nvcc -V to check your cuda installation
# If you have installed cuda but fail the nvcc command, add these two lines to your shell profile, ex. ~/.zshrc
# export PATH="/usr/local/cuda-11.8/bin:$PATH"
# export LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH"
cmake_minimum_required(VERSION 3.14)

project(NURBS LANGUAGES CXX CUDA)
if(CUDA_ENABLED)
    enable_language(CUDA)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_CUDA_FLAGS "-DFIX1 --gpu-architecture=compute_86 -Xptxas -dlcm=ca -O0") 
# set_directory_properties( PROPERTIES COMPILE_DEFINITIONS "" )

include_directories (kernel NURBS test util)
aux_source_directory (test SRC_LIST1)
aux_source_directory (kernel SRC_LIST2)
aux_source_directory (NURBS SRC_LIST3)
aux_source_directory (util SRC_LIST4)

set(SRC_LIST main.cu ${SRC_LIST1} ${SRC_LIST2} ${SRC_LIST3} ${SRC_LIST4})
add_executable(NURBS ${SRC_LIST})

# solve cuda link problem
find_package(CUDA)
target_link_libraries(NURBS ${CUDA_cusparse_LIBRARY} ${CUDA_cublas_LIBRARY} )

# allow separable compilation
# generate relocatable code
set_property(TARGET NURBS PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET NURBS PROPERTY OUTPUT_NAME "test.o")
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/../)
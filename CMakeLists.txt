cmake_minimum_required(VERSION 3.10)
project(SystolicArray_Conv)

set(CMAKE_CXX_STANDARD 17)

# Update this path to where SystemC is installed on your machine/WSL
set(SYSTEMC_ROOT "/usr/local/systemc" CACHE PATH "SystemC installation directory")

# Include directories
include_directories(${SYSTEMC_ROOT}/include)
include_directories(src/dialted_convolution)
include_directories(src/grouped_convolution)
include_directories(Depthwise_convolution)

link_directories(${SYSTEMC_ROOT}/lib64) # Change to lib-macosx if on Mac

# Targets
add_executable(dilated_conv src/dialted_convolution/dilated_conv_tb.cpp)
target_link_libraries(dilated_conv systemc)

add_executable(depthwise_conv Depthwise_convolution/depthwise_conv_tb.cpp)
target_link_libraries(depthwise_conv systemc)

add_executable(grouped_conv src/grouped_convolution/grouped_conv_tb.cpp)
target_link_libraries(grouped_conv systemc)

# Original sim target
add_executable(sim src/main.cpp)
target_link_libraries(sim systemc)
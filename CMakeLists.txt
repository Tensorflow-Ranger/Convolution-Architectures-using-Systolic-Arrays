cmake_minimum_required(VERSION 3.10)
project(SystolicArray_Conv)

set(CMAKE_CXX_STANDARD 14)

# Update this path to where SystemC is installed on your machine/WSL
set(SYSTEMC_ROOT "/usr/local/systemc-2.3.3" CACHE PATH "SystemC installation directory")

include_directories(${SYSTEMC_ROOT}/include)
link_directories(${SYSTEMC_ROOT}/lib-linux64) # Change to lib-macosx if on Mac

# Add your C++ files here as you create them
add_executable(sim 
    src/main.cpp 
)

target_link_libraries(sim systemc)
cmake_minimum_required(VERSION 3.12.0)
project(wave-equation LANGUAGES CXX C)

include(../common/cmake-common.cmake)

add_executable(wmain src/wmain.cpp src/wave.cpp)
deal_ii_setup_target(wmain)

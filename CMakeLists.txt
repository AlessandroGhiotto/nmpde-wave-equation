cmake_minimum_required(VERSION 3.12.0)
project(wave-equation LANGUAGES CXX C)

### COMMON
# include(../common/cmake-common.cmake)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED "ON")

# Set default build type to Release.
if(NOT CMAKE_BUILD_TYPE OR "${CMAKE_BUILD_TYPE}" STREQUAL "")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()
message(STATUS)
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS)
if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  add_definitions(-DBUILD_TYPE_DEBUG)
endif()

# Locate MPI compiler.
find_package(MPI REQUIRED)
set(CMAKE_CXX_COMPILER "${MPI_CXX_COMPILER}")

# Locate Boost.
find_package(Boost 1.72.0 REQUIRED
  COMPONENTS filesystem iostreams serialization
  HINTS ${BOOST_DIR} $ENV{BOOST_DIR} $ENV{mkBoostPrefix})
message(STATUS "Using the Boost-${Boost_VERSION} configuration found at ${Boost_DIR}")
message(STATUS)
include_directories(${Boost_INCLUDE_DIRS})

# Locate deal.II and initialize its variables.
find_package(deal.II 9.3.1 REQUIRED
  HINTS ${DEAL_II_DIR} $ENV{DEAL_II_DIR} $ENV{mkDealiiPrefix})
deal_ii_initialize_cached_variables()

# Add useful compiler flags.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wfloat-conversion -Wmissing-braces -Wnon-virtual-dtor")

###########################################

# Add the base class library
add_library(WaveEquationBase STATIC src/WaveEquationBase.cpp)
deal_ii_setup_target(WaveEquationBase)

add_library(WaveTheta STATIC src/WaveTheta.cpp)
target_link_libraries(WaveTheta WaveEquationBase)
deal_ii_setup_target(WaveTheta)

add_library(WaveNewmark STATIC src/WaveNewmark.cpp)
target_link_libraries(WaveNewmark WaveEquationBase)
deal_ii_setup_target(WaveNewmark)

add_library(ParameterReader STATIC src/ParameterReader.cpp)
deal_ii_setup_target(ParameterReader)

add_executable(main-theta src/main-theta.cpp)
target_link_libraries(main-theta WaveTheta ParameterReader)
deal_ii_setup_target(main-theta)

add_executable(main-newmark src/main-newmark.cpp)
target_link_libraries(main-newmark WaveNewmark ParameterReader)
deal_ii_setup_target(main-newmark)

cmake_minimum_required(VERSION 3.21)

project(plaquette_simulator_tests)

set(CMAKE_CXX_STANDARD 20)

# Default build type for test code is Debug
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

option(PLAQUETTE_SIMULATOR_ENABLE_NATIVE "Enable native CPU build tuning" OFF)

Include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v2.13.7
)

FetchContent_MakeAvailable(Catch2)

# Required for catch_discover_tests().
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/contrib)

# Modify `ctest` to only run the supported subset of tests.
include(CTest)
include(Catch)

add_executable(test_runner runner.cpp )
target_link_libraries(test_runner PUBLIC Catch2::Catch2 kokkos)
target_include_directories(test_runner PUBLIC ${CMAKE_SOURCE_DIR}/plaquette_simulator/src)
target_include_directories(test_runner PUBLIC ${PLAQUETTE_SIMULATOR_KOKKOS_INC_DIR} )

message("CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")
target_compile_options(test_runner PRIVATE "$<$<CONFIG:DEBUG>:-Wall>")
catch_discover_tests(test_runner)


# add_library(plaquette_simulator INTERFACE)
# target_link_libraries(plaquette_simulator INTERFACE kokkos)
# target_include_directories(plaquette_simulator INTERFACE "plaquette_simulator/src")
# target_include_directories(plaquette_simulator INTERFACE ${PLAQUETTE_SIMULATOR_KOKKOS_INC_DIR} )

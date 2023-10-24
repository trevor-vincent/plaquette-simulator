cmake_minimum_required(VERSION 3.21)

project(plaquette_simulator_benchmarks)

set(CMAKE_CXX_STANDARD 20)

# Default build type for test code is Debug
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(PLAQUETTE_SIMULATOR_ENABLE_NATIVE "Enable native CPU build tuning" ON)

add_executable(run_circuit run_circuit.cpp )
target_include_directories(run_circuit PUBLIC ${CMAKE_SOURCE_DIR}/plaquette_simulator/src)
target_include_directories(run_circuit PUBLIC ${PLAQUETTE_SIMULATOR_KOKKOS_INC_DIR} )
target_link_libraries(run_circuit PUBLIC kokkos)
message(STATUS "PLAQUETTE_SIMULATOR_KOKKOS_INC_DIR: ${PLAQUETTE_SIMULATOR_KOKKOS_INC_DIR}")

##########################
## Set Project version
##########################
cmake_minimum_required(VERSION 3.20)
set(CMAKE_OSX_DEPLOYMENT_TARGET "10.15" CACHE STRING "Minimum OS X deployment version")

project(plaquette_simulator
    DESCRIPTION "Lightning-Kokkos bindings for PennyLane. Backed by the Kokkos library."
    LANGUAGES CXX C
)

##########################
## Enforce C++ Standard
##########################
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

##########################
## Set Default Options
##########################
# Compiler options
option(PLAQUETTE_SIMULATOR_ENABLE_NATIVE "Enable native CPU build tuning" OFF)
option(PLAQUETTE_SIMULATOR_BUILD_TESTS "Build cpp tests" OFF)
option(PLAQUETTE_SIMULATOR_ENABLE_WARNINGS "Enable warnings" OFF)
option(PLAQUETTE_SIMULATOR_ENABLE_CLANG_TIDY "Enable clang-tidy build checks" OFF)
option(PLAQUETTE_SIMULATOR_ENABLE_SANITIZER "Enable address sanitizer" OFF)
option(PLAQUETTE_SIMULATOR_BUILD_BINDINGS "Build Python bindings" ON)


# Kokkos options
if(Kokkos_ENABLE_OPENMP)
  set(PLAQUETTE_SIMULATOR_ENABLE_OPENMP ON)
endif()

if(Kokkos_ENABLE_THREADS)
  set(PLAQUETTE_SIMULATOR_ENABLE_THREADS ON)
endif()

# Build options
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

# Ensure the libraries can see additional libs at same level;
# Required for external deps when loading in Python
set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)
set(CMAKE_SKIP_BUILD_RPATH FALSE)

if(PLAQUETTE_SIMULATOR_ENABLE_CLANG_TIDY)
    if (NOT DEFINED CLANG_TIDY_BINARY)
        set(CLANG_TIDY_BINARY clang-tidy)
    endif()
    set(CMAKE_CXX_CLANG_TIDY ${CLANG_TIDY_BINARY};
                            -extra-arg=-std=c++20;
    )
endif()

if(PLAQUETTE_SIMULATOR_ENABLE_SANITIZER)
    add_compile_options(-fsanitize=address)
    if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        add_link_options(-fsanitize=address)
    else ()
        add_link_options(-fsanitize=address -static-libasan)
    endif()
    set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
    set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
ENDIF()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
##########################
## Fetch dependencies
##########################

option(Kokkos_ENABLE_DEPRECATION_WARNINGS "Enable Kokkos deprecation warnings" OFF)

include(FetchContent)
FetchContent_Declare(kokkos
  GIT_REPOSITORY https://github.com/kokkos/kokkos.git
  # GIT_TAG        main
)
FetchContent_MakeAvailable(kokkos)
FetchContent_GetProperties(kokkos)
if(NOT kokkos_POPULATED)
  FetchContent_Populate(kokkos)
endif()

# Set the include directory into a variable
set(PLAQUETTE_SIMULATOR_KOKKOS_INC_DIR ${kokkos_SOURCE_DIR}/core/src)
message(STATUS "Kokkos include dir: ${PLAQUETTE_SIMULATOR_KOKKOS_INC_DIR}")

if(PLAQUETTE_SIMULATOR_ENABLE_WARNINGS)
    target_compile_options(plaquette_simulator INTERFACE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Werror;>
    )
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(plaquette_simulator INTERFACE
            $<$<COMPILE_LANGUAGE:CXX>:-Wno-error=volatile;-Wno-error=deprecated-declarations>
        )
    endif()
    # Explicitly disable warnings for Kokkos-specific library issues.
endif()

if(PLAQUETTE_SIMULATOR_ENABLE_NATIVE)
    message(STATUS "ENABLE_NATIVE is ON. Using -march=native")
    add_compile_options(-march=native)
    target_compile_options(plaquette_simulator INTERFACE -march=native)
endif()

if(PLAQUETTE_SIMULATOR_BUILD_BINDINGS)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.10.1
)
FetchContent_MakeAvailable(pybind11)
find_package (Python COMPONENTS Interpreter Development)
pybind11_add_module(plaquette_simulator_bindings "plaquette_simulator/src/Bindings.cpp")
target_include_directories(plaquette_simulator_bindings PUBLIC ${PLAQUETTE_SIMULATOR_KOKKOS_INC_DIR} )
target_link_libraries(plaquette_simulator_bindings PUBLIC kokkos)
endif()


if(PLAQUETTE_SIMULATOR_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()


if (PLAQUETTE_SIMULATOR_BUILD_TESTS)
  add_subdirectory("plaquette_simulator/src/tests")
endif()



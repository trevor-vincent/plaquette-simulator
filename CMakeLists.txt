##########################
## Set Project version
##########################
cmake_minimum_required(VERSION 3.20)
set(CMAKE_OSX_DEPLOYMENT_TARGET "10.15" CACHE STRING "Minimum OS X deployment version")

project(plaquette_hpc_simulator
    DESCRIPTION "Lightning-Kokkos bindings for PennyLane. Backed by the Kokkos library."
    LANGUAGES CXX C
)

##########################
## Utility methods
##########################

# Read and set plaquette_hpc_simulator version
function(set_plaquette_hpc_simulator_version VERSION_FILE_PATH)
    file(STRINGS ${VERSION_FILE_PATH} VERSION_FILE_STR)
    foreach (LINE IN LISTS VERSION_FILE_STR)
        if("${LINE}" MATCHES "__version__.*")
            set(VERSION_LINE_STR "${LINE}")
        endif()
    endforeach()
    string(REGEX REPLACE "__version__ = \"(.*)\"" "\\1" VERSION_STRING ${VERSION_LINE_STR})
    set(VERSION_STRING ${VERSION_STRING} PARENT_SCOPE)
endfunction()

set_plaquette_hpc_simulator_version(${PROJECT_SOURCE_DIR}/plaquette_hpc_simulator/_version.py)

message(STATUS "plaquette_hpc_simulator version ${VERSION_STRING}")
set(PROJECT_VERSION ${VERSION_STRING})

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
option(PLAQUETTE_HPC_SIMULATOR_ENABLE_NATIVE "Enable native CPU build tuning" OFF)
option(PLAQUETTE_HPC_SIMULATOR_BUILD_TESTS "Build cpp tests" OFF)
option(PLAQUETTE_HPC_SIMULATOR_ENABLE_WARNINGS "Enable warnings" ON)
option(PLAQUETTE_HPC_SIMULATOR_ENABLE_CLANG_TIDY "Enable clang-tidy build checks" OFF)
option(PLAQUETTE_HPC_SIMULATOR_ENABLE_PYTHON "Build Python bindings" ON)
option(PLAQUETTE_HPC_SIMULATOR_ENABLE_SANITIZER "Enable address sanitizer" OFF)

# Kokkos options
if(Kokkos_ENABLE_OPENMP)
  set(PLAQUETTE_HPC_SIMULATOR_ENABLE_OPENMP ON)
endif()

if(Kokkos_ENABLE_THREADS)
  set(PLAQUETTE_HPC_SIMULATOR_ENABLE_THREADS ON)
endif()

# Build options
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

# Ensure the libraries can see additional libs at same level;
# Required for external deps when loading in Python
set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)
set(CMAKE_SKIP_BUILD_RPATH FALSE)

if(PLAQUETTE_HPC_SIMULATOR_ENABLE_CLANG_TIDY)
    if (NOT DEFINED CLANG_TIDY_BINARY)
        set(CLANG_TIDY_BINARY clang-tidy)
    endif()
    set(CMAKE_CXX_CLANG_TIDY ${CLANG_TIDY_BINARY};
                            -extra-arg=-std=c++20;
    )
endif()

if(PLAQUETTE_HPC_SIMULATOR_ENABLE_SANITIZER)
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

find_package(Kokkos 3.7)
if(Kokkos_FOUND)
    message(STATUS "Found existing Kokkos libraries")
    add_library(kokkos ALIAS Kokkos::kokkos)
else()
    message(STATUS "Building Kokkos from source.")
    include(FetchContent)
    FetchContent_Declare(kokkos
    GIT_REPOSITORY https://github.com/kokkos/kokkos.git
    GIT_TAG        3.7.00
    )
    FetchContent_MakeAvailable(kokkos)
    get_target_property(PLAQUETTE_HPC_SIMULATOR_KOKKOS_INC_DIR kokkos INTERFACE_INCLUDE_DIRECTORIES)
    set_target_properties(kokkos PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${PLAQUETTE_HPC_SIMULATOR_KOKKOS_INC_DIR}")
endif()

############################
## Create libraries
############################
add_subdirectory(plaquette_hpc_simulator/src)
add_library(plaquette_hpc_simulator INTERFACE)
target_link_libraries(plaquette_hpc_simulator INTERFACE kokkos)
target_include_directories(plaquette_hpc_simulator INTERFACE "plaquette_hpc_simulator/src")
target_include_directories(plaquette_hpc_simulator INTERFACE ${PLAQUETTE_HPC_SIMULATOR_KOKKOS_INC_DIR} )

# Create binding module
if(PLAQUETTE_HPC_SIMULATOR_BUILD_BINDINGS)
include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.10.1
)
FetchContent_MakeAvailable(pybind11)

find_package (Python COMPONENTS Interpreter Development)
    if(Kokkos_ENABLE_CUDA)
        add_library(plaquette_hpc_simulator_bindings MODULE "plaquette_hpc_simulator/src/bindings/Bindings.cpp")
        target_link_libraries(plaquette_hpc_simulator_bindings PUBLIC pybind11::module)
        pybind11_extension(plaquette_hpc_simulator_bindings)
    else()
        pybind11_add_module(plaquette_hpc_simulator_bindings "plaquette_hpc_simulator/src/bindings/Bindings.cpp" )
    endif() 
    target_link_libraries(plaquette_hpc_simulator_bindings PRIVATE plaquette_hpc_simulator)
    set_target_properties(plaquette_hpc_simulator_bindings PROPERTIES CXX_VISIBILITY_PRESET "hidden" CUDA_VISIBILITY_PRESET "hidden")
    target_compile_options(plaquette_hpc_simulator_bindings PRIVATE "$<$<CONFIG:RELEASE>:-W>")
    target_include_directories(plaquette_hpc_simulator_bindings PRIVATE ${PLAQUETTE_HPC_SIMULATOR_KOKKOS_INC_DIR})
    target_compile_definitions(plaquette_hpc_simulator_bindings PRIVATE VERSION_INFO=${VERSION_STRING})
    target_link_libraries(plaquette_hpc_simulator_bindings PRIVATE kokkos)
endif()


if(PLAQUETTE_HPC_SIMULATOR_ENABLE_WARNINGS)
    target_compile_options(plaquette_hpc_simulator INTERFACE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Werror;>
    )
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(plaquette_hpc_simulator INTERFACE
            $<$<COMPILE_LANGUAGE:CXX>:-Wno-error=volatile;-Wno-error=deprecated-declarations>
        )
    endif()
    # Explicitly disable warnings for Kokkos-specific library issues.
endif()

if(PLAQUETTE_HPC_SIMULATOR_ENABLE_NATIVE)
    message(STATUS "ENABLE_NATIVE is ON. Using -march=native")
    add_compile_options(-march=native)
    target_compile_options(plaquette_hpc_simulator INTERFACE -march=native)
endif()

install(TARGETS plaquette_hpc_simulator
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        INCLUDES DESTINATION include
        PUBLIC_HEADER DESTINATION include
)

install(DIRECTORY 
    ${PROJECT_SOURCE_DIR}/plaquette_hpc_simulator/src
    DESTINATION include/plaquette_hpc_simulator
)

if (PLAQUETTE_HPC_SIMULATOR_BUILD_TESTS)
  add_subdirectory("plaquette_hpc_simulator/src/tests")
endif()



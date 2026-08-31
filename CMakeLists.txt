cmake_minimum_required(VERSION 3.20)
project(cascade VERSION 0.1.0 LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(CASCADE_ENABLE_CUDA   "Build the CUDA execution backend"      OFF)
option(CASCADE_ENABLE_OPENMP "Build the OpenMP execution backend"    ON)
option(CASCADE_BUILD_TESTS   "Build the unit / verification tests"   ON)
option(CASCADE_WERROR        "Treat compiler warnings as errors"     OFF)

set(CASCADE_REAL "float" CACHE STRING "Solver floating point type: float or double")
set_property(CACHE CASCADE_REAL PROPERTY STRINGS float double)

set(CASCADE_NDIM "2" CACHE STRING "Spatial dimension: 2 or 3")
set_property(CACHE CASCADE_NDIM PROPERTY STRINGS 2 3)

if(NOT CASCADE_REAL MATCHES "^(float|double)$")
  message(FATAL_ERROR "CASCADE_REAL must be 'float' or 'double' (got '${CASCADE_REAL}')")
endif()
if(NOT CASCADE_NDIM MATCHES "^(2|3)$")
  message(FATAL_ERROR "CASCADE_NDIM must be 2 or 3 (got '${CASCADE_NDIM}')")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---------------------------------------------------------------------------
# Core library
# ---------------------------------------------------------------------------
add_library(cascade_core
  src/core/Config.cpp
)
add_library(cascade::core ALIAS cascade_core)

target_include_directories(cascade_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_compile_definitions(cascade_core PUBLIC
  CASCADE_REAL_TYPE=${CASCADE_REAL}
  CASCADE_NDIM=${CASCADE_NDIM}
  CASCADE_VERSION_STRING="${PROJECT_VERSION}"
)

if(MSVC)
  target_compile_options(cascade_core PUBLIC /W4 /permissive- /Zc:__cplusplus)
  if(CASCADE_WERROR)
    target_compile_options(cascade_core PUBLIC /WX)
  endif()
else()
  target_compile_options(cascade_core PUBLIC -Wall -Wextra -Wpedantic)
  if(CASCADE_WERROR)
    target_compile_options(cascade_core PUBLIC -Werror)
  endif()
endif()

# ---------------------------------------------------------------------------
# OpenMP backend
# ---------------------------------------------------------------------------
if(CASCADE_ENABLE_OPENMP)
  find_package(OpenMP COMPONENTS CXX)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(cascade_core PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(cascade_core PUBLIC CASCADE_HAVE_OPENMP=1)
    message(STATUS "cascade: OpenMP backend enabled")
  else()
    message(STATUS "cascade: OpenMP requested but not found -- Serial backend only")
  endif()
endif()

# ---------------------------------------------------------------------------
# CUDA backend (compiles without a GPU present; running needs one)
# ---------------------------------------------------------------------------
if(CASCADE_ENABLE_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)  # RTX 3090; 89 for RTX 4090
  endif()
  target_compile_definitions(cascade_core PUBLIC CASCADE_HAVE_CUDA=1)
  target_compile_options(cascade_core PUBLIC
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  message(STATUS "cascade: CUDA backend enabled (arch ${CMAKE_CUDA_ARCHITECTURES})")
endif()

# ---------------------------------------------------------------------------
# Executable
# ---------------------------------------------------------------------------
add_executable(cascade src/main.cpp)
target_link_libraries(cascade PRIVATE cascade_core)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(CASCADE_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

message(STATUS "cascade: Real=${CASCADE_REAL}  NDIM=${CASCADE_NDIM}  build=${CMAKE_BUILD_TYPE}")

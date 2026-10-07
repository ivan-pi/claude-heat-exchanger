cmake_minimum_required(VERSION 3.22)
# C is enabled only because the preCICE CMake package calls find_dependency(Threads).
project(solid-fdm LANGUAGES C Fortran)

# --------------------------------------------------------------------------------------
# Options
#   SOLID_OFFLOAD        ON: compile OpenMP target regions for the GPU (NVHPC: -mp=gpu)
#                        OFF: host OpenMP only (target regions execute on the host)
#   SOLID_GPU_FLAGS      NVHPC -gpu= sub-options, e.g. "cc90" or "cc90,mem:unified"
#   SOLID_GNU_OFFLOAD    extra flags for gfortran offloading, e.g. "-foffload=nvptx-none"
#
# The preCICE Fortran module (precice/fortran-module) is fetched with FetchContent.
# Offline: -DFETCHCONTENT_SOURCE_DIR_PRECICE_FORTRAN_MODULE=/path/to/fortran-module
# --------------------------------------------------------------------------------------
option(SOLID_OFFLOAD "Offload OpenMP target regions to the GPU" ON)
set(SOLID_GPU_FLAGS "cc90" CACHE STRING "NVHPC -gpu= sub-options")
set(SOLID_GNU_OFFLOAD "" CACHE STRING "Extra gfortran offload flags")
set(PRECICE_FORTRAN_MODULE_TAG "v3.4.0" CACHE STRING "Tag of precice/fortran-module")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(precice 3 REQUIRED CONFIG)

include(FetchContent)
FetchContent_Declare(precice_fortran_module
  GIT_REPOSITORY https://github.com/precice/fortran-module.git
  GIT_TAG        ${PRECICE_FORTRAN_MODULE_TAG}
  GIT_SHALLOW    TRUE)
FetchContent_MakeAvailable(precice_fortran_module) # no CMakeLists.txt there: populate only

add_library(precice_fortran_module STATIC ${precice_fortran_module_SOURCE_DIR}/precice.f90)
set_target_properties(precice_fortran_module PROPERTIES
  Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/mod)
target_include_directories(precice_fortran_module PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/mod)
target_link_libraries(precice_fortran_module PUBLIC precice::precice)

add_executable(solid-fdm
  src/case_params.f90
  src/heat_solver.f90
  src/solid_main.f90)
set_target_properties(solid-fdm PROPERTIES
  Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/mod
  LINKER_LANGUAGE Fortran)
target_link_libraries(solid-fdm PRIVATE precice_fortran_module)

if(CMAKE_Fortran_COMPILER_ID STREQUAL "NVHPC")
  if(SOLID_OFFLOAD)
    set(_mp -mp=gpu -gpu=${SOLID_GPU_FLAGS})
  else()
    set(_mp -mp=multicore)
  endif()
  target_compile_options(solid-fdm PRIVATE ${_mp} -Minfo=mp)
  target_link_options(solid-fdm PRIVATE ${_mp})
else()
  find_package(OpenMP REQUIRED COMPONENTS Fortran)
  target_link_libraries(solid-fdm PRIVATE OpenMP::OpenMP_Fortran)
  if(SOLID_OFFLOAD AND SOLID_GNU_OFFLOAD)
    separate_arguments(_gnu_offload UNIX_COMMAND "${SOLID_GNU_OFFLOAD}")
    target_compile_options(solid-fdm PRIVATE ${_gnu_offload})
    target_link_options(solid-fdm PRIVATE ${_gnu_offload})
  endif()
  if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
    target_compile_options(solid-fdm PRIVATE -Wall -Wno-unused-dummy-argument -Wno-maybe-uninitialized)
  endif()
endif()

include(GNUInstallDirs)
install(TARGETS solid-fdm RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

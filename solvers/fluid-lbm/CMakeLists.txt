cmake_minimum_required(VERSION 3.22)
project(fluid-lbm LANGUAGES CXX)

# --------------------------------------------------------------------------------------
# Options
#   LBM_BACKEND   STDPAR (default) | CUDA | SERIAL
#   LBM_GPU_FLAGS NVHPC -gpu= sub-options, e.g. "cc90,mem:managed" (Grace Hopper default)
#                 or "cc90,mem:unified" to use the GH system-allocated unified memory.
#
# Typical builds
#   GCC (testing, multithreaded via TBB):
#       cmake -S . -B build -DLBM_BACKEND=STDPAR
#   NVHPC, std::for_each offloaded to the GPU, managed memory:
#       cmake -S . -B build -DCMAKE_CXX_COMPILER=nvc++ -DLBM_BACKEND=STDPAR
#   NVHPC, CUDA C++ kernels launching the same lambdas:
#       cmake -S . -B build -DCMAKE_CXX_COMPILER=nvc++ -DLBM_BACKEND=CUDA
# --------------------------------------------------------------------------------------
set(LBM_BACKEND "STDPAR" CACHE STRING "Execution backend: STDPAR, CUDA or SERIAL")
set_property(CACHE LBM_BACKEND PROPERTY STRINGS STDPAR CUDA SERIAL)
set(LBM_GPU_FLAGS "cc90,mem:managed" CACHE STRING "NVHPC -gpu= sub-options")
option(LBM_STDPAR_MULTICORE "NVHPC only: use -stdpar=multicore instead of -stdpar=gpu" OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(precice 3 REQUIRED CONFIG)

add_executable(fluid-lbm src/main.cpp)
target_compile_features(fluid-lbm PRIVATE cxx_std_17)
target_link_libraries(fluid-lbm PRIVATE precice::precice)

if(LBM_BACKEND STREQUAL "STDPAR")
  target_compile_definitions(fluid-lbm PRIVATE LBM_BACKEND_STDPAR)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
    if(LBM_STDPAR_MULTICORE)
      set(_stdpar -stdpar=multicore)
    else()
      set(_stdpar -stdpar=gpu -gpu=${LBM_GPU_FLAGS})
    endif()
    target_compile_options(fluid-lbm PRIVATE ${_stdpar} -Minfo=stdpar)
    target_link_options(fluid-lbm PRIVATE ${_stdpar})
  else()
    # libstdc++ implements the parallel policies on top of oneTBB; without it they
    # silently fall back to sequential execution.
    find_package(TBB QUIET)
    if(TBB_FOUND)
      target_link_libraries(fluid-lbm PRIVATE TBB::tbb)
      message(STATUS "fluid-lbm: stdpar on the host via TBB")
    else()
      message(WARNING "fluid-lbm: TBB not found, std::execution::par_unseq will run sequentially")
    endif()
  endif()

elseif(LBM_BACKEND STREQUAL "CUDA")
  target_compile_definitions(fluid-lbm PRIVATE LBM_BACKEND_CUDA)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
    target_compile_options(fluid-lbm PRIVATE -cuda -gpu=${LBM_GPU_FLAGS})
    target_link_options(fluid-lbm PRIVATE -cuda -gpu=${LBM_GPU_FLAGS})
  else()
    # nvcc path: compile the translation unit as CUDA with extended lambdas
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set_source_files_properties(src/main.cpp PROPERTIES LANGUAGE CUDA)
    set_target_properties(fluid-lbm PROPERTIES CUDA_ARCHITECTURES 90)
    target_compile_options(fluid-lbm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
    target_link_libraries(fluid-lbm PRIVATE CUDA::cudart)
  endif()

elseif(LBM_BACKEND STREQUAL "SERIAL")
  # nothing to add
else()
  message(FATAL_ERROR "Unknown LBM_BACKEND '${LBM_BACKEND}'")
endif()

include(GNUInstallDirs)
install(TARGETS fluid-lbm RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

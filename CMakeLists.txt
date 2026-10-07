# Superbuild for both participants. Each solver is also a standalone CMake project
# (solvers/fluid-lbm, solvers/solid-fdm), like the solvers in precice/tutorials.
#
#   cmake -S . -B build -DCMAKE_PREFIX_PATH=<precice-install>        # GCC: testing
#   cmake -S . -B build -DCMAKE_CXX_COMPILER=nvc++ -DCMAKE_Fortran_COMPILER=nvfortran \
#         -DCMAKE_C_COMPILER=nvc                                      # NVHPC: Grace Hopper
#   cmake --build build -j
cmake_minimum_required(VERSION 3.22)
project(precice-heat-exchanger LANGUAGES C CXX Fortran)

add_subdirectory(solvers/fluid-lbm)
add_subdirectory(solvers/solid-fdm)

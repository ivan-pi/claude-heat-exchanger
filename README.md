# preCICE conjugate-heat-transfer prototype: LBM fluid ↔ implicit FD wall

A 2-D counterflow heat exchanger with baffled (serpentine) channels, split into **two
preCICE participants** coupled across two surfaces. The Fluid executable advances both
channels as two lattices; the Solid is the insulating wall between them.

```
 y = 2H+d  ───────────────────────────────────────────  adiabatic
           hot channel   ──►  u_hot,  T_in_hot = 350 K  (baffles) ┐
 y = H+d   ══════ Fluid-Hot-Mesh  ⇄  Solid-Hot-Mesh ══════       │ Code A: fluid-lbm
           insulating polymer wall  (Code B: solid-fdm)          │ (C++, D2Q9 + D2Q5)
 y = H     ══════ Fluid-Cold-Mesh ⇄  Solid-Cold-Mesh ══════      │
           cold channel  ◄──  u_cold, T_in_cold = 290 K (baffles) ┘
 y = 0     ───────────────────────────────────────────  adiabatic
           x = 0                                    x = L
```

Default case (`case/params.txt`): water channels H = 1 mm, L = 6 mm, a 0.5 mm wall with
k = 0.2 W/(m K), Re ≈ 3.1 / 2.2, Pr = 5.4.
- **Baffles:** 8 per channel, 2 cells (62.5 µm) thick and 0.65 H tall. They alternate between the outer and the coupled wall, so the flow runs as a serpentine with inverted-U loops over the plates standing on each channel's lower wall. The two channels are staggered by half a pitch, and the solid fraction is 5.4 % (the code refuses > 10 %).
- **Straight channels:** set `baffle_count 0` (and use `u_mean_hot 6e-3`, `u_mean_cold 4e-3` to reproduce the straight-channel run below).
- **Physics:** temperature is a passive scalar (forced convection, no buoyancy, constant properties).

## Layout

```
CMakeLists.txt               superbuild (both solvers); each solver also builds standalone
solvers/fluid-lbm/           Code A: C++17, kernels as lambdas behind lbm::parallel_for
  src/parallel_for.hpp       backends: STDPAR (std::for_each), CUDA, SERIAL
  src/channel.hpp            one channel: D2Q9 flow + D2Q5 heat, BCs, checkpoints
  src/main.cpp               preCICE adapter, energy budget, VTK output
solvers/solid-fdm/           Code B: Fortran 2008 + OpenMP target offload
  src/heat_solver.f90        backward Euler, matrix-free Jacobi-PCG, flux recovery
  src/solid_main.f90         preCICE adapter (official Fortran module)
case/                        precice-config.xml, params.txt, run scripts
  plot_energy.py, animate.py post-processing (shared helpers in hx_post.py)
```

## Numerics

**Code A: fluid (`fluid-lbm`).**
- **Masked LBM:** a per-node solid flag. Every link from a fluid node to a solid node (or to a channel wall) uses half-way bounce-back for D2Q9 and adiabatic bounce-back for D2Q5. Solid nodes are skipped. Any geometry can be drawn into the mask (`make_baffles()` in `main.cpp` is the generator).
- **Baffle roots on the coupled wall:** these cover interface faces (8 of 192 per channel here). Heat that the solid delivers under a root goes to the nearest wetted face, and the root face reports that face's wall temperature. That's a zero-height, perfectly conducting fin, and it keeps the coupling exactly conservative. Resolving conduction inside the baffles would mean modelling them as conducting solids (another participant, or conjugate LBM).
- **Flow:** D2Q9 BGK with the incompressible equilibrium of He & Luo, which makes the steady velocity divergence free. Walls use half-way bounce-back. The inlet is a parabolic velocity profile and the outlet is held at ρ = 1, both via Guo's non-equilibrium extrapolation.
- **Heat:** D2Q5 BGK for θ = T − T_ref, advected by the D2Q9 velocity.
- **Coupled wall:** a prescribed-flux bounce-back, `g_in = g_out + q·Δt/(ρc_p Δx)`. The wall temperature sent to preCICE is `T_w = T_0 + q Δx/(2k)`.
- **Grid:** the inlet and outlet boundary columns sit half a lattice spacing outside [0, L], so all interface heat enters interior nodes.
- **Start-up:** before coupling starts, the flow is developed for `lbm_flow_init_time` (2 s, 8000 steps). The thermal populations are then reset to the uniform initial temperature, in equilibrium with the developed velocity. Without this phase, the start-up pressure waves around the baffles (the initial profile isn't divergence free) act as spurious sources of θ in the conservative D2Q5 equation. The hot stream briefly reached 358 K, against a 350 K inlet.
- **Time stepping:** the fluid subcycles 200 LBM steps (Δt = 0.25 ms) per 0.05 s coupling window.
- **Kernels:** two launches per channel and step: the fused pull stream+collide kernel (which also books the scalar flux through the inlet/outlet faces for the energy budget) and the inlet/outlet columns. The macroscopic fields are not stored every step; they are derived from the populations once per window for the budget and the output. Per window there are also the wall-temperature gather, the checkpoint copies and one budget reduction per channel.

**Code B: wall (`solid-fdm`).**
- **Discretization:** vertex-centred finite differences in finite-volume form, with half cells on every boundary. The matrix is SPD and the interface flux comes from the boundary half-cell energy balance, so the wall is discretely conservative.
- **Time stepping:** backward Euler, one step per window.
- **Linear solver:** matrix-free CG with a Jacobi preconditioner (the diagonal depends on the column and the step size only, so it is a 1-D array per step). The dot product p·Ap is formed inside the matrix-vector kernel, so one CG iteration is three target regions. It stops when the residual has dropped by `cg_tol` relative to the initial residual (the flux imbalance driving the step), with an absolute floor at round-off.
- **GPU data:** all fields stay on the device (`target enter data`); the grid and material constants are `declare target` module variables. Only the 1-D interface buffers (and the preconditioner diagonal) move, via `target update`.

## Coupling choices (`case/precice-config.xml`)

- **Dirichlet–Neumann, oriented for an insulating wall.** The fluid reads Heat-Flux; the solid reads Temperature. For a fixed-point coupling iteration, the error amplification factor is roughly the ratio of the Dirichlet side's thermal conductance to the Neumann side's. Here k_s/d = 400 W/m²K against a fluid-side h ≈ 1700 W/m²K, so the low-conductance wall should take the Dirichlet data. That's the reverse of preCICE's flow-over-heated-plate tutorial, where the solid conducts well.
- **Serial-implicit coupling** with IQN-ILS acceleration on Heat-Flux. Both solvers checkpoint and roll back their state. Convergence takes about 4 iterations per window early in the transient and 1 near steady state (1.41 per window on average).
- **Mapping:** `nearest-projection`, i.e. linear interpolation along the interface edges that both participants define. The meshes are non-matching: 192 fluid wall faces against 97 solid vertices.
- **`m2n:sockets` on `lo`** (TCP/IP on the loopback interface). On a cluster with the two codes on different nodes, set `network="ib0"` (or whatever the fabric is called) and use a shared `exchange-directory`.

## Build

preCICE ≥ 3.0 must be findable by CMake (`find_package(precice 3 CONFIG)`), in the same way
as the solvers in `precice/tutorials`.

```bash
# Local testing with GCC (stdpar runs multithreaded on the host through TBB,
# OpenMP target regions execute on the host)
cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/precice
cmake --build build -j

# Grace Hopper with NVHPC
cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/precice \
      -DCMAKE_C_COMPILER=nvc -DCMAKE_CXX_COMPILER=nvc++ -DCMAKE_Fortran_COMPILER=nvfortran \
      -DLBM_BACKEND=STDPAR -DLBM_GPU_FLAGS="cc90,mem:managed" \
      -DSOLID_OFFLOAD=ON  -DSOLID_GPU_FLAGS="cc90"
#   -DLBM_BACKEND=CUDA   -> the same lambdas launched from a __global__ kernel (nvc++ -cuda)
#   -DLBM_GPU_FLAGS=cc90,mem:unified / -DSOLID_GPU_FLAGS=cc90,mem:unified
#                        -> GH system-allocated unified memory
```

Further options:
- `LBM_BACKEND=SERIAL` (debugging), `LBM_STDPAR_MULTICORE=ON` (nvc++ `-stdpar=multicore`).
- `SOLID_OFFLOAD=OFF` (`-mp=multicore`).
- `SOLID_GNU_OFFLOAD="-foffload=nvptx-none"` (a gfortran built with offloading support).
- The preCICE Fortran module (`precice/fortran-module`, tag `v3.4.0`) is pulled in by FetchContent. To build offline, pass `-DFETCHCONTENT_SOURCE_DIR_PRECICE_FORTRAN_MODULE=<dir>`.
- The solid project enables C only because the preCICE CMake package calls `find_dependency(Threads)`.

## Run

```bash
case/run-coupled.sh                 # both participants, logs in case/fluid|solid/*.log
#   on a shared host build: OMP_NUM_THREADS=1 OMP_WAIT_POLICY=passive keeps the solid's
#   idle OpenMP threads from spinning against the fluid's TBB workers
python3 case/plot_energy.py         # -> case/energy_balance.png, case/energy_summary.txt
python3 case/animate.py             # -> case/heat_exchanger.mp4 (needs ffmpeg)
```

`animate.py` uses every VTK snapshot. For a smooth video, set `output_interval 0.1` in
`params.txt`; at 10 fps, 12 s of simulated time gives 121 frames.

In a batch job, the two `run.sh` scripts can be started on separate nodes or GPUs.
`CUDA_VISIBLE_DEVICES` decides which GPU each one uses.

## Energy conservation checks

Both participants log a per-window energy budget, in W per metre of depth:

| file | quantities |
|---|---|
| `fluid/output/fluid_energy.csv` | per channel: enthalpy inflow (H_in − H_out), wall heat Q_wall, storage dE/dt, outlet bulk T |
| `solid/output/solid_energy.csv` | Q into the hot and cold fluid, wall storage dE/dt, PCG iterations |
| `*/output/*_interface.csv` | final T and q profiles on the fluid and solid meshes |

How each check is computed:
- **Fluid enthalpy flow:** the D2Q5 populations that actually stream across x = 0 and x = L, accumulated on the device over every LBM substep. That makes the channel budget exact, not a finite-difference estimate.
- **Wall storage:** the same half-cell weights as the discretization.
- **Interface:** the integral of q over the fluid mesh (after mapping) is compared with the integral over the solid mesh.
- **Whole system:** ΣH_net,in = Σ dE/dt over both channels and the wall.

Results for t = 0…20 s (`case/energy_summary.txt`), Q = heat through the wall:

| check | max \|residual\| / Q, serpentine (default) | straight channels |
|---|---|---|
| hot channel  dE/dt − ΔH − Q_wall | 8e-12 | 8e-12 |
| cold channel dE/dt − ΔH − Q_wall | 8e-12 | 1e-11 |
| wall         dE/dt + Q_hot + Q_cold | 8e-9 | 3e-9 |
| whole system | 3e-6 | 4e-5 |
| steady state: hot loss − cold gain | 1e-8 | 2e-12 |

The whole-system residual is set by the implicit-coupling tolerance (see below). In the
serpentine case the steady heat flow is 91.9 W/m (hot outlet 341.3 K, cold outlet 302.2 K).
It reaches steady state after about 5 s, at 1.61 coupling iterations per window.

The whole-system residual jumps once windows start converging in a single coupling
iteration (around t ≈ 8 s in the serpentine run). From then on, the fluid computes each
window with the previous window's flux, and the mismatch equals the flux change per window.
That decays as the system settles. `<min-iterations value="2"/>` or a tighter Heat-Flux limit
removes the jump, at the cost of about one extra fluid solve per window.

**Effect of the baffles**, both cases at the same flow rates (u = 2.5 / 1.75 mm/s), t = 12 s:

| | Q [W/m] | ε | U [W/m²K] | hot / cold outlet [K] |
|---|---|---|---|---|
| straight channels | 84.5 | 0.193 | 281 | 342.1 / 301.3 |
| alternating baffles | 91.9 | 0.210 | 311 | 341.3 / 302.2 |

The polymer wall alone has a resistance of d/k = 2.5e-3 m²K/W. Subtracting it, the baffles
cut the convective resistance (1/h_hot + 1/h_cold) by about 30 %, from 1.06e-3 to
0.72e-3 m²K/W, but the overall U rises only 11 %. At Re ≈ 3 the gain comes from the flow being
forced onto the coupled wall. The closed eddies behind the baffles work against it, which
shows as the heat-flux dips at the baffle roots. Pressure drop isn't evaluated yet.

The interface mismatch Q(fluid mesh) − Q(solid mesh) is the residual of the coupling iteration. The fluid's last iteration of a window used the previous (accelerated) flux iterate, so the mismatch is bounded by the 1e-4 Heat-Flux convergence limit. Tightening that limit tightens the system balance. Here the mapping itself happens to conserve energy, because the meshes are nested 2:1 and the midpoint rule on the fluid mesh then equals the trapezoidal rule on the solid mesh. In general, consistent nearest-projection is *not* conservative. For heat flux with arbitrary mesh ratios, write a nodal heat rate [W] with a `conservative` write mapping and divide by the local face length on the reading side.

Plausibility of the straight-channel case at steady state: Q = 95.8 W/m gives ε = 0.096 and NTU = 0.104, i.e.
U ≈ 289 W/m²K. The fully developed estimate is 1/U = 2/h + d/k with Nu = 5.385 (one side
heated, the other insulated), which gives 270 W/m²K. The 7 % excess is consistent with the
thermally developing entrance region (x* = L/(D_h Re Pr) ≈ 0.04–0.06).

## GPU data path for the coupling

preCICE (up to and including v3.4) only accepts **host-readable** buffers. The
`writeData`/`readData` spans are copied into preCICE's own host-side buffers. Mapping,
acceleration and the m2n communication all run on the host. Even the GPU mapping backends
(Ginkgo or Kokkos, v3.2+) take host data and copy it internally. The two codes handle this
as follows:

- **Code A (managed memory):** a kernel gathers the wall temperature into a small, contiguous
  managed buffer. After `device_sync()`, that pointer goes straight to preCICE. The flux is
  read into a second small managed buffer that the next kernel picks up. There's no explicit
  copy, and only those pages migrate, not the lattice.
- **Code B (OpenMP):** `!$omp target update from(q)` / `to(Tb)` on the 1-D interface
  buffers only. With `-gpu=mem:unified` on Grace Hopper these updates are
  practically free, and the same code still works on discrete GPUs.

## Limitations / next steps

- Single rank per participant. For MPI-parallel runs, both adapters need a partitioned
  interface (`setMeshVertices` per rank) and preCICE built with MPI.
- The heat flux is read once per window (end-of-window value, consistent with the solid's
  backward Euler). With `substeps="true"` and waveform interpolation, the fluid could read
  time-interpolated flux in every LBM substep.
- The CUDA backend (`LBM_BACKEND=CUDA`) and NVHPC flags are written but untested here: the
  test machine had no NVIDIA toolchain or GPU. The stdpar and serial backends and the
  Fortran OpenMP code were run (GCC 13 + TBB, gfortran 13) against preCICE v3.4.1.
- With the default resolution, temperatures overshoot the inlet range by up to 0.7 K
  (1 % of ΔT) at the first baffle tip. The cell Péclet number in the gaps is about 2.3; use
  `lbm_ny 48` or a TRT thermal collision to remove it.
- Gap velocities in the serpentine reach a lattice Mach number of about 0.09. Faster flows need a smaller
  `lbm_dt` (τ moves towards 0.5) or a finer lattice, or a more robust collision operator
  (TRT/MRT; TRT also makes the bounce-back wall location viscosity-independent).
- The thermal conductivity of the fluid enters only through τ_g. Temperature-dependent
  properties or buoyancy (Boussinesq forcing in D2Q9) would be local additions to the fused
  kernel.

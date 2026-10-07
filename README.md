# preCICE conjugate-heat-transfer prototype: LBM fluid ↔ implicit FD wall

A 2-D counterflow heat exchanger with baffled (serpentine) channels as **two preCICE
participants** coupled across two surfaces: the Fluid advances both channels as two
lattices, the Solid is the insulating wall between them.

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

Default case (`case/params.txt`): water channels H = 1 mm, L = 6 mm, 0.5 mm wall with
k = 0.2 W/(m K), Re ≈ 3.1 / 2.2, Pr = 5.4. Temperature is a passive scalar (forced
convection, constant properties, no buoyancy).
- **Baffles:** 8 per channel, 2 cells (62.5 µm) thick, 0.65 H tall, alternating between the
  outer and the coupled wall (serpentine flow with inverted-U loops over the plates on each
  channel's lower wall). The channels are staggered by half a pitch; solid fraction 5.4 %
  (the code refuses > 10 %).
- **Straight channels:** `baffle_count 0` (with `u_mean_hot 6e-3`, `u_mean_cold 4e-3` for
  the straight-channel run below).

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

**Code A: fluid (`fluid-lbm`)**
- **Masked LBM:** a per-node solid flag; every fluid→solid or fluid→wall link uses half-way
  bounce-back (adiabatic for D2Q5), solid nodes are skipped. Any geometry can be drawn into
  the mask (`make_baffles()` in `main.cpp`).
- **Baffle roots on the coupled wall** cover interface faces (8 of 192 per channel). Heat
  delivered under a root goes to the nearest wetted face, which also provides the root's
  wall temperature: a zero-height, perfectly conducting fin that keeps the coupling exactly
  conservative. Conduction inside the baffles would need a conducting-solid model.
- **Flow:** D2Q9 BGK with the incompressible He & Luo equilibrium (divergence-free steady
  velocity). Parabolic velocity inlet and ρ = 1 outlet via Guo's non-equilibrium
  extrapolation.
- **Heat:** D2Q5 BGK for θ = T − T_ref advected by the D2Q9 velocity. Coupled wall:
  prescribed-flux bounce-back `g_in = g_out + q·Δt/(ρc_p Δx)`; the wall temperature sent to
  preCICE is `T_w = T_0 + q Δx/(2k)`.
- **Grid:** the inlet/outlet boundary columns sit half a spacing outside [0, L], so all
  interface heat enters interior nodes.
- **Start-up:** the flow is developed for `lbm_flow_init_time` (2 s, 8000 steps), then the
  thermal populations are reset to the inlet temperature in equilibrium with the developed
  velocity. Otherwise the start-up pressure waves around the baffles act as spurious
  sources of θ (the hot stream reached 358 K against a 350 K inlet).
- **Time stepping:** 200 LBM steps (Δt = 0.25 ms) per 0.05 s coupling window.
- **Kernels:** two launches per channel and step, the fused pull stream+collide kernel
  (which also books the scalar flux through the inlet/outlet faces for the energy budget)
  and the inlet/outlet columns. The macroscopic fields are derived from the populations
  once per window for the budget and the output.

**Code B: wall (`solid-fdm`)**
- **Discretization:** vertex-centred finite differences in finite-volume form with half
  cells on every boundary: SPD matrix, interface flux from the boundary half-cell energy
  balance, hence discretely conservative.
- **Time stepping:** backward Euler, one step per window.
- **Linear solver:** matrix-free Jacobi-PCG (the diagonal is a 1-D array per step; p·Ap is
  formed inside the matrix-vector kernel, three target regions per iteration). Stops at a
  residual reduction of `cg_tol` relative to the initial residual (the flux imbalance
  driving the step), with an absolute floor at round-off.
- **GPU data:** all fields stay on the device (`target enter data`), constants are
  `declare target` module variables; only the 1-D interface buffers and the preconditioner
  diagonal move (`target update`).

## Coupling (`case/precice-config.xml`)

- **Dirichlet–Neumann oriented for an insulating wall:** the fluid reads Heat-Flux, the
  solid reads Temperature. The fixed-point error amplification is roughly the Dirichlet
  side's conductance over the Neumann side's; here k_s/d = 400 W/m²K against a fluid-side
  h ≈ 1700 W/m²K, so the wall takes the Dirichlet data (the reverse of preCICE's
  flow-over-heated-plate tutorial).
- **Serial-implicit coupling** with IQN-ILS on Heat-Flux; both solvers checkpoint and roll
  back. About 4 iterations per window early in the transient, 1 near steady state.
- **Mapping:** `nearest-projection` (linear interpolation along the interface edges) between
  non-matching meshes: 192 fluid faces, 97 solid vertices.
- **`m2n:sockets` on `lo`.** With the codes on different nodes, set `network="ib0"` (or the
  fabric's name) and a shared `exchange-directory`.

## Build

preCICE ≥ 3.0 must be findable by CMake (`find_package(precice 3 CONFIG)`).

```bash
# GCC testing build: stdpar multithreaded on the host via TBB, OpenMP target regions on the host
cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/precice && cmake --build build -j

# Grace Hopper with NVHPC
cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/precice \
      -DCMAKE_C_COMPILER=nvc -DCMAKE_CXX_COMPILER=nvc++ -DCMAKE_Fortran_COMPILER=nvfortran \
      -DLBM_BACKEND=STDPAR -DLBM_GPU_FLAGS="cc90,mem:managed" \
      -DSOLID_OFFLOAD=ON  -DSOLID_GPU_FLAGS="cc90"
```

Options: `LBM_BACKEND=CUDA` (same lambdas from a `__global__` kernel, nvc++ `-cuda` or nvcc)
or `SERIAL`; `LBM_STDPAR_MULTICORE=ON` (nvc++ `-stdpar=multicore`); `mem:unified` in the GPU
flags for GH system-allocated unified memory; `SOLID_OFFLOAD=OFF` (`-mp=multicore`);
`SOLID_GNU_OFFLOAD="-foffload=nvptx-none"` for an offloading gfortran. The preCICE Fortran
module (`precice/fortran-module` `v3.4.0`) comes via FetchContent; offline, pass
`-DFETCHCONTENT_SOURCE_DIR_PRECICE_FORTRAN_MODULE=<dir>`. The solid project enables C only
because the preCICE package calls `find_dependency(Threads)`.

## Run

```bash
case/run-coupled.sh                 # both participants, logs in case/fluid|solid/*.log
python3 case/plot_energy.py         # -> case/energy_balance.png, case/energy_summary.txt
python3 case/animate.py             # -> case/heat_exchanger.mp4 (needs ffmpeg)
```

On a shared host build, `OMP_NUM_THREADS=1 OMP_WAIT_POLICY=passive` keeps the solid's idle
OpenMP threads from spinning against the fluid's TBB workers. In a batch job the two
`run.sh` scripts can run on separate nodes or GPUs (`CUDA_VISIBLE_DEVICES`). `animate.py`
uses every VTK snapshot; `output_interval 0.1` gives a smooth video (121 frames for 12 s).

## Energy conservation checks

Both participants log a per-window budget in W per metre of depth:

| file | quantities |
|---|---|
| `fluid/output/fluid_energy.csv` | per channel: enthalpy inflow H_in − H_out, wall heat Q_wall, storage dE/dt, outlet bulk T |
| `solid/output/solid_energy.csv` | Q into the hot and cold fluid, wall storage dE/dt, PCG iterations |
| `*/output/*_interface.csv` | final T and q profiles on the fluid and solid meshes |

The fluid enthalpy flow is taken from the D2Q5 populations that actually cross x = 0 and
x = L, accumulated over every substep, so the channel budget is exact. The wall storage uses
the discretization's half-cell weights. The interface check compares the integral of q over
the fluid mesh (after mapping) with the integral over the solid mesh; the whole-system check
is ΣH_net,in = Σ dE/dt over both channels and the wall.

Results for t = 0…20 s (`case/energy_summary.txt`), Q = heat through the wall:

| check | max \|residual\| / Q, serpentine (default) | straight channels |
|---|---|---|
| hot channel  dE/dt − ΔH − Q_wall | 8e-12 | 8e-12 |
| cold channel dE/dt − ΔH − Q_wall | 8e-12 | 1e-11 |
| wall         dE/dt + Q_hot + Q_cold | 8e-9 | 3e-9 |
| whole system | 3e-6 | 4e-5 |
| steady state: hot loss − cold gain | 1e-8 | 2e-12 |

The serpentine case settles after about 5 s at 91.9 W/m (hot outlet 341.3 K, cold outlet
302.2 K), 1.61 coupling iterations per window. The whole-system residual equals the
interface mismatch Q(fluid mesh) − Q(solid mesh), i.e. the coupling-iteration residual: the
fluid's last iteration of a window used the previous flux iterate, so it is bounded by the
1e-4 Heat-Flux convergence limit. It jumps once windows converge in one iteration (t ≈ 8 s),
when the mismatch becomes the flux change per window; `<min-iterations value="2"/>` or a
tighter limit removes the jump for about one extra fluid solve per window. The mapping
itself conserves energy only because the meshes are nested 2:1 (midpoint rule on the fluid
mesh = trapezoidal rule on the solid mesh); consistent nearest-projection is not conservative
in general. For arbitrary mesh ratios, write a nodal heat rate [W] with a `conservative`
mapping and divide by the local face length on the reading side.

**Effect of the baffles**, same flow rates (u = 2.5 / 1.75 mm/s), t = 12 s:

| | Q [W/m] | ε | U [W/m²K] | hot / cold outlet [K] |
|---|---|---|---|---|
| straight channels | 84.5 | 0.193 | 281 | 342.1 / 301.3 |
| alternating baffles | 91.9 | 0.210 | 311 | 341.3 / 302.2 |

The wall alone contributes d/k = 2.5e-3 m²K/W. The baffles cut the convective resistance
1/h_hot + 1/h_cold by about 30 % (1.06e-3 → 0.72e-3 m²K/W), but U rises only 11 %. At Re ≈ 3
the gain comes from the flow being forced onto the coupled wall; the closed eddies behind the
baffles work against it (heat-flux dips at the roots). Pressure drop is not evaluated.

Plausibility, straight channels at steady state: Q = 95.8 W/m gives ε = 0.096, NTU = 0.104,
U ≈ 289 W/m²K. The fully developed estimate 1/U = 2/h + d/k with Nu = 5.385 (one side
heated, one insulated) gives 270 W/m²K; the 7 % excess matches the thermally developing
entrance region (x* = L/(D_h Re Pr) ≈ 0.04–0.06).

## GPU data path for the coupling

preCICE (up to v3.4) only accepts host-readable buffers: `writeData`/`readData` copy into
preCICE's host buffers, and mapping, acceleration and m2n run on the host (the Ginkgo/Kokkos
mapping backends also take host data). Code A gathers the wall temperature into a small
managed buffer whose pointer goes to preCICE after `device_sync()`, and reads the flux into
a second one that the next kernel picks up; only those pages migrate, not the lattice. Code
B does `!$omp target update from(q)` / `to(Tb)` on the 1-D buffers; with `-gpu=mem:unified`
on Grace Hopper these are practically free, and the same code works on discrete GPUs.

## Limitations / next steps

- Single rank per participant; MPI runs need a partitioned interface in both adapters and
  preCICE built with MPI.
- The flux is read once per window (end-of-window value, consistent with backward Euler).
  With `substeps="true"` and waveform interpolation the fluid could read it every substep.
- The CUDA backend and the NVHPC flags are untested (no NVIDIA toolchain on the test
  machine). Tested: stdpar and serial backends, Fortran OpenMP on the host (GCC 13 + TBB,
  gfortran 13, preCICE v3.4.1).
- At the default resolution the temperature overshoots the inlet range by up to 0.7 K (1 %
  of ΔT) at the first baffle tip (cell Péclet ≈ 2.3 in the gaps); `lbm_ny 48` or a TRT
  thermal collision removes it.
- Gap velocities reach lattice Mach ≈ 0.09. Faster flows need a smaller `lbm_dt`, a finer
  lattice, or TRT/MRT collision (TRT also makes the bounce-back wall location
  viscosity-independent).
- Fluid conductivity enters only through τ_g; temperature-dependent properties or Boussinesq
  buoyancy would be local additions to the fused kernel.

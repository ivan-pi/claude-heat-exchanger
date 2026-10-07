# Device-direct exchange of the coupling data: what it would take

Question: could the interface data (wall temperature, heat flux) go straight from one
participant's GPU into the other participant's buffer, and is GPU-aware MPI enough for
that, or does preCICE itself have to change?

Short answer: GPU-aware MPI is necessary for the transport but not sufficient. The
application never sends the coupling data itself; preCICE does, from its own host-side
buffers, after copying, mapping, storing and accelerating the data on the host. A
device-direct path therefore has to be built inside preCICE. Mirrored host/device copies
on the application side only make today's host API work, which is what this prototype does.

## Where the data goes today (preCICE 3.x)

```
Fluid (process A)                                      Solid (process B)
kernel -> managed buffer                               device field
  |  device_sync()                                        |  target update from(q)
  v                                                       v
writeData(span)  ---- memcpy into preCICE host storage (Eigen) ----
  mapping (host; Ginkgo/Kokkos RBF backends copy to the GPU and back)
  waveform storage, convergence measures, IQN acceleration (host)
  m2n: sockets (Boost.Asio) or MPI (MPI_Isend/Irecv of host arrays)
                       ---- network ---->
                                               preCICE host storage (Eigen)
                                               readData(span): memcpy into the user's buffer
                                               target update to(Tb) / next kernel touches the page
```

Two facts follow from this picture:

- The two participants are separate executables with separate address spaces (connected by
  TCP sockets or by MPI ports / a shared MPI world). There is no common buffer. The only
  cross-process channel is preCICE's `m2n` layer, and an application cannot replace it with
  its own MPI calls, because preCICE owns the data between `writeData` and `readData`.
- `writeData` and `readData` take host-readable spans. A device pointer only works if it is
  managed or system-allocated unified memory, in which case the page migrates to the host
  on access. preCICE's own storage is ordinary host memory allocated inside its library,
  so it is not managed even when the application is compiled with `-gpu=mem:managed`.

## Why GPU-aware MPI alone does not do it

GPU-aware MPI lets `MPI_Send`/`MPI_Recv` take device pointers and use GPUDirect RDMA
between nodes. That helps only where MPI is called with a device buffer. In the current
chain the MPI call sits inside preCICE and receives preCICE's host buffer, so building
preCICE against a GPU-aware MPI changes nothing. Keeping device and host copies of the
application's interface buffers does not change this either: preCICE still copies from
the host copy into its own storage and transports from there.

## What preCICE would need

1. **A memory-space-aware API.** `writeData`/`readData` (or a new variant) would accept a
   device span plus a memory-space tag, or hand out preCICE-owned device buffers for the
   solver to fill, together with a stream or event on which the data becomes valid.
2. **Device-resident storage.** The per-mesh data vectors and the waveform samples kept
   per time window are Eigen host vectors; they would have to live on the device (Kokkos
   views or similar) or at least in pinned host memory for staging.
3. **Mapping on the device.** Nearest-neighbour and nearest-projection mappings run on the
   host. The Ginkgo and Kokkos RBF backends (v3.2+) run on the GPU but copy host data in
   and results out; they would need to consume device data directly.
4. **Acceleration and convergence measures.** IQN-ILS, the residual norms and the waveform
   interpolation operate on host Eigen vectors. For interface-sized data they could stay
   on the host behind a device-to-host copy, but then the device-direct path is only
   complete for explicit or pure-relaxation coupling; for the implicit scheme used here the
   accelerating participant brings the data to the host anyway.
5. **A device-capable communication backend.** The MPI backends (`m2n:mpi`,
   `m2n:mpi-single`) could pass device pointers once preCICE is built with a GPU-aware
   MPI; this is where GPU-aware MPI becomes the enabler. The default `m2n:sockets` backend
   has no device path at all. On a single node, CUDA IPC handles or NVSHMEM would be
   alternatives, but they would equally have to be implemented inside the `com` layer.
6. **Synchronization semantics.** preCICE would have to know when the producing kernel has
   finished (stream/event arguments to `advance`) and must not touch the buffers while
   the solver's kernels run.

None of this exists in preCICE up to v3.4, so the answer to the second question is yes:
preCICE would need to be modified (a device data path is on its development roadmap, the
GPU mapping backends being the first step), and GPU-aware MPI would be one component of it.

## What is worth doing now

For this case the interface is 192 values per mesh, so the exchange is latency-bound, not
bandwidth-bound: the cost per window is the device synchronization before `writeData` and
the round trip through the coupling scheme, not the bytes. Within the current API:

- Keep the device-side gather into small contiguous buffers (done), so that only those
  pages migrate and the lattice never touches the host.
- On Grace Hopper with `-gpu=mem:unified`, the host-side copy in `writeData` is a read over
  NVLink-C2C and costs microseconds; a device-direct path would save little more than the
  explicit `device_sync()`.
- Prefer `m2n:mpi` over `m2n:sockets` when both participants run under one MPI launcher;
  it has lower latency and is the backend a future device path would build on.
- A device-direct transport becomes worthwhile for 3-D interfaces with 1e5 to 1e6 vertices
  per rank, where the host staging and the per-rank copies dominate; at that point the
  partitioned (multi-rank) interface in both adapters is needed first.

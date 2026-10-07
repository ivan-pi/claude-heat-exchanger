// channel.hpp -- one channel of the counterflow heat exchanger (masked LBM).
//
// Geometry: nx x ny lattice between two plane walls (y = 0 and y = ny), plus arbitrary
//          internal solid nodes given by a mask (baffles, ribs, ...). Every link from a
//          fluid node to a solid node or to a wall uses half-way bounce-back. Columns 0
//          and nx-1 are the inlet/outlet boundary columns; the nc = nx-2 interior
//          columns 1..nc are the coupled length, and "face" f = 0..nc-1 is the piece of
//          the coupled wall under column f+1 (the interface buffers are indexed by face).
// Flow:    D2Q9 BGK, incompressible equilibrium of He & Luo (1997) so that the steady
//          velocity field is divergence free, velocity inlet (parabolic) and
//          constant-density outlet, both via non-equilibrium extrapolation (Guo et al. 2002).
// Heat:    D2Q5 BGK passive scalar (advection-diffusion of T) using the D2Q9 velocity.
//          Evolved variable theta = T - T_ref (reduces compressibility/round-off errors).
//          Outer wall and masked solids: adiabatic (bounce-back). Coupled wall: prescribed heat flux
//          (bounce-back plus source), inlet: Dirichlet T_in, outlet: zero gradient.
//
// Kernels: one fused pull-stream + collide kernel per step (it also books the scalar flux
//          through the inlet/outlet faces for the energy budget), followed by the
//          inlet/outlet-column kernel. The macroscopic fields rho, u, T are not stored
//          every step; update_fields() derives them from the populations on demand.
// Layout:  structure of arrays, f[k*N + j*nx + i], i (streamwise) fastest.
// Memory:  lbm::device_vector -> host/device accessible (managed or unified memory).
#pragma once

#include "parallel_for.hpp"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

namespace lbm {

// ---------------------------------------------------------------------------------------
// D2Q9 velocity set; directions 0..4 are the D2Q5 set of the thermal lattice.
// The tables are function-local constexpr arrays rather than namespace-scope ones: nvcc
// does not allow run-time indexing of a namespace-scope constexpr array from device code.
// ---------------------------------------------------------------------------------------
namespace d2q9 {

LBM_HD constexpr int cx(int k)
{
  constexpr int t[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
  return t[k];
}
LBM_HD constexpr int cy(int k)
{
  constexpr int t[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
  return t[k];
}
LBM_HD constexpr int opp(int k)
{
  constexpr int t[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};
  return t[k];
}

// Incompressible equilibrium of He & Luo, rho_0 = 1.
LBM_HD inline double feq(int k, double r, double u, double v)
{
  const double w  = k == 0 ? 4.0 / 9.0 : (k < 5 ? 1.0 / 9.0 : 1.0 / 36.0);
  const double cu = cx(k) * u + cy(k) * v;
  return w * (r + 3.0 * cu + 4.5 * cu * cu - 1.5 * (u * u + v * v));
}

// D2Q5 equilibrium for theta = T - T_ref with w0 = 1/3, w1..4 = 1/6, i.e. c_s^2 = 1/3.
LBM_HD inline double geq(int k, double th, double u, double v)
{
  const double w = k == 0 ? 1.0 / 3.0 : 1.0 / 6.0;
  return w * th * (1.0 + 3.0 * (cx(k) * u + cy(k) * v));
}

// BGK relaxation time <-> lattice diffusivity (viscosity or thermal), c_s^2 = 1/3.
inline double relaxation_time(double diffusivity) { return 3.0 * diffusivity + 0.5; }
LBM_HD inline double diffusivity(double tau) { return (tau - 0.5) / 3.0; }

} // namespace d2q9

// Parabolic (Poiseuille) profile with mean U; half-way walls at y = 0 and y = 1.
LBM_HD inline double inlet_profile(int j, int ny, double U)
{
  const double y = (j + 0.5) / ny;
  return 6.0 * U * y * (1.0 - y);
}

struct ChannelSetup {
  int    nx             = 0;    // lattice nodes along the channel (incl. inlet/outlet columns)
  int    ny             = 0;    // lattice nodes across the channel
  double u_mean         = 0.0;  // mean inlet velocity [lattice units], sign = flow direction
  double T_in           = 0.0;  // inlet (and initial) temperature [K]
  double T_ref          = 0.0;  // reference temperature, D2Q5 evolves T - T_ref [K]
  bool   coupled_bottom = true; // true: coupled wall is the bottom wall (y = 0)
  double tau_f          = 1.0;  // flow relaxation time
  double tau_g          = 1.0;  // thermal relaxation time
};

// ---------------------------------------------------------------------------------------
// Device view of one channel for the step kernels: raw pointers and scalars only, so it
// can be captured by value.
// ---------------------------------------------------------------------------------------
struct LatticeView {
  int                 nx, ny, N;
  double              omf, omg;      // 1 / tau_f, 1 / tau_g
  double              U, T_in, Tref; // mean inlet velocity, inlet and reference temperature
  int                 jwall;         // row index of the coupled wall seen from the fluid: -1 (bottom) or ny (top)
  int                 i_in, i_out;   // inlet and outlet boundary columns
  const double       *fs, *gs;       // populations at the start of the step
  double             *fd, *gd;       // populations at the end of the step
  const double       *jf;            // lattice heat flux into the fluid through face f (column f+1) of the coupled wall
  double             *acc;           // scalar flux accumulated through x = 0 (rows 0..ny-1) and x = L (ny..2ny-1)
  const std::uint8_t *msk;           // 1 = solid node
};

// Source node of the population streaming along direction k into fluid node (i, j), or
// -1 when the link comes from a wall or a solid node (half-way bounce-back). Only for
// interior columns, so i - cx(k) stays inside the lattice.
LBM_HD inline int pull_source(const LatticeView &L, int i, int j, int k)
{
  const int sj = j - d2q9::cy(k);
  if (sj < 0 || sj >= L.ny) {
    return -1;
  }
  const int nb = sj * L.nx + i - d2q9::cx(k);
  return L.msk[nb] ? -1 : nb;
}

// Fused pull-streaming + BGK collision for node idx. Interior columns only; the boundary
// columns are set afterwards by boundary_node(). Also books the D2Q5 populations that are
// about to cross x = 0 (between columns 0|1) and x = L (columns nx-2|nx-1).
LBM_HD inline void stream_collide(const LatticeView &L, int idx)
{
  const int nx = L.nx, N = L.N;
  const int i = idx % nx, j = idx / nx;

  if (i == 0) {
    L.acc[j] += L.gs[1 * N + idx] - L.gs[3 * N + idx + 1];
    return;
  }
  if (i == nx - 2) {
    L.acc[L.ny + j] += L.gs[1 * N + idx] - L.gs[3 * N + idx + 1];
  }
  if (i == nx - 1 || L.msk[idx]) {
    return; // outlet/inlet column, or solid node (inactive, never read by fluid neighbours)
  }

  double fl[9], gl[5];
  double r = 0.0, u = 0.0, v = 0.0, th = 0.0;
  LBM_UNROLL
  for (int k = 0; k < 9; ++k) {
    const int nb = pull_source(L, i, j, k);
    fl[k]        = nb < 0 ? L.fs[d2q9::opp(k) * N + idx] : L.fs[k * N + nb];
    r += fl[k];
    u += d2q9::cx(k) * fl[k];
    v += d2q9::cy(k) * fl[k];
  }
  LBM_UNROLL
  for (int k = 0; k < 5; ++k) {
    const int nb = pull_source(L, i, j, k);
    if (nb < 0) { // adiabatic bounce-back; the coupled wall injects the prescribed flux
      gl[k] = L.gs[d2q9::opp(k) * N + idx] + (j - d2q9::cy(k) == L.jwall ? L.jf[i - 1] : 0.0);
    } else {
      gl[k] = L.gs[k * N + nb];
    }
    th += gl[k];
  }
  LBM_UNROLL
  for (int k = 0; k < 9; ++k) {
    L.fd[k * N + idx] = fl[k] - L.omf * (fl[k] - d2q9::feq(k, r, u, v));
  }
  LBM_UNROLL
  for (int k = 0; k < 5; ++k) {
    L.gd[k * N + idx] = gl[k] - L.omg * (gl[k] - d2q9::geq(k, th, u, v));
  }
}

// Inlet/outlet column node t (t < ny: inlet row t, else outlet row t - ny): non-equilibrium
// extrapolation from the neighbouring interior column (post-collision). BGK collision
// preserves the moments, so they are taken from the post-collision populations.
LBM_HD inline void boundary_node(const LatticeView &L, int t)
{
  const int  nx = L.nx, ny = L.ny, N = L.N;
  const int  j     = t % ny;
  const bool inlet = t < ny;
  const int  ib    = inlet ? L.i_in : L.i_out;
  const int  in    = ib == 0 ? 1 : nx - 2;
  const int  idb   = j * nx + ib;
  const int  idn   = j * nx + in;

  double fn[9], gn[5];
  double rn = 0.0, un = 0.0, vn = 0.0, thn = 0.0;
  LBM_UNROLL
  for (int k = 0; k < 9; ++k) {
    fn[k] = L.fd[k * N + idn];
    rn += fn[k];
    un += d2q9::cx(k) * fn[k];
    vn += d2q9::cy(k) * fn[k];
  }
  LBM_UNROLL
  for (int k = 0; k < 5; ++k) {
    gn[k] = L.gd[k * N + idn];
    thn += gn[k];
  }
  const double rb  = inlet ? rn : 1.0;
  const double ub  = inlet ? inlet_profile(j, ny, L.U) : un;
  const double vb  = inlet ? 0.0 : vn;
  const double thb = inlet ? L.T_in - L.Tref : thn;
  LBM_UNROLL
  for (int k = 0; k < 9; ++k) {
    L.fd[k * N + idb] = d2q9::feq(k, rb, ub, vb) + fn[k] - d2q9::feq(k, rn, un, vn);
  }
  LBM_UNROLL
  for (int k = 0; k < 5; ++k) {
    L.gd[k * N + idb] = d2q9::geq(k, thb, ub, vb) + gn[k] - d2q9::geq(k, thn, un, vn);
  }
}

// ---------------------------------------------------------------------------------------
class Channel {
public:
  explicit Channel(const ChannelSetup &s)
      : s_(s), N_(s.nx * s.ny), nc_(s.nx - 2),
        f_{device_vector<double>(9 * N_), device_vector<double>(9 * N_)},
        g_{device_vector<double>(5 * N_), device_vector<double>(5 * N_)},
        rho_(N_), ux_(N_), uy_(N_), T_(N_), f_ckpt_(9 * N_), g_ckpt_(5 * N_),
        jflux_(nc_, 0.0), qbuf_(nc_, 0.0), Twall_(nc_, 0.0), face_acc_(2 * s.ny, 0.0), mask_(N_, 0),
        iface_src_dev_(nc_), iface_src_(nc_), q_received_(nc_, 0.0)
  {
    std::iota(iface_src_.begin(), iface_src_.end(), 0);
    std::copy(iface_src_.begin(), iface_src_.end(), iface_src_dev_.begin());
  }

  int nx() const { return s_.nx; }
  int ny() const { return s_.ny; }
  int interface_size() const { return nc_; } // coupled faces 0..nc-1 under columns 1..nc
  int coupled_row() const { return s_.coupled_bottom ? 0 : s_.ny - 1; }
  int inlet_column() const { return s_.u_mean >= 0.0 ? 0 : s_.nx - 1; }
  int outlet_column() const { return s_.u_mean >= 0.0 ? s_.nx - 1 : 0; }
  const ChannelSetup &setup() const { return s_; }

  // Solid mask (1 = solid), host vector of size nx*ny, index j*nx + i. Must be set before
  // initialize(). The coupled-wall row and the inlet/outlet columns must stay fluid.
  //
  // Solid nodes on the coupled-wall row (roots of baffles attached to the coupled wall)
  // cover interface faces. For each covered face the nearest wetted face is recorded:
  // heat delivered by the solid under a root is passed to that face, and the root face
  // reports that face's wall temperature (zero-height, perfectly conducting fin). This
  // keeps the coupling exactly conservative.
  void set_mask(const std::vector<std::uint8_t> &m)
  {
    std::copy(m.begin(), m.end(), mask_.begin());
    const std::uint8_t *wall = mask_.data() + coupled_row() * s_.nx + 1; // wall[f]: node above face f
    for (int f = 0; f < nc_; ++f) {
      iface_src_[f] = f;
      for (int d = 1; wall[f] && d < nc_; ++d) {
        if (f - d >= 0 && !wall[f - d]) {
          iface_src_[f] = f - d;
          break;
        }
        if (f + d < nc_ && !wall[f + d]) {
          iface_src_[f] = f + d;
          break;
        }
      }
    }
    std::copy(iface_src_.begin(), iface_src_.end(), iface_src_dev_.begin());
  }
  int covered_interface_faces() const
  {
    int n = 0;
    for (int f = 0; f < nc_; ++f) {
      n += iface_src_[f] != f;
    }
    return n;
  }
  const std::uint8_t *mask() const { return mask_.data(); }
  double solid_fraction() const
  {
    return double(std::count(mask_.begin(), mask_.end(), std::uint8_t(1))) / N_;
  }

  // Interface buffers (nc entries, host-visible managed/unified memory), indexed by face.
  double       *heat_flux() { return qbuf_.data(); }                       // preCICE read target [W/m^2]; redistributed by apply_heat_flux()
  const double *heat_flux() const { return qbuf_.data(); }                 // as applied
  const double *heat_flux_received() const { return q_received_.data(); } // as mapped by preCICE
  double        wall_heat_sum() const { return q_sum_; }                   // sum of the applied flux over the faces
  const double *wall_temperature() const { return Twall_.data(); }        // preCICE write source [K]
  // Scalar flux through the faces x = 0 (between columns 0|1, entries 0..ny-1) and
  // x = L (columns nx-2|nx-1, entries ny..2ny-1), summed over all steps since the last
  // reset_face_flux(): exactly what streaming transferred, in lattice units.
  const double *face_flux_accumulated() const { return face_acc_.data(); }

  // Macroscopic fields, valid after update_fields() (call device_sync() before host reads).
  const double *T() const { return T_.data(); }
  const double *ux() const { return ux_.data(); }
  const double *uy() const { return uy_.data(); }

  // ------------------------------------------------------------------------------------
  // Flow populations at rest density with the inlet profile; thermal field uniform at T_in.
  void initialize()
  {
    const int    nx = s_.nx, ny = s_.ny, N = N_;
    const double U  = s_.u_mean;
    double      *f  = f_[cur_].data();
    const std::uint8_t *msk = mask_.data();
    parallel_for(N, LBM_LAMBDA(int idx) {
      const double u = msk[idx] ? 0.0 : inlet_profile(idx / nx, ny, U);
      LBM_UNROLL
      for (int k = 0; k < 9; ++k) {
        f[k * N + idx] = d2q9::feq(k, 1.0, u, 0.0);
      }
    });
    reset_thermal();
    reset_face_flux();
  }

  // Develop the flow field before the coupled run: nsteps LBM steps with zero wall flux,
  // then re-initialise the thermal populations to the uniform initial temperature in
  // equilibrium with the developed velocity. Without this, the start-up pressure waves
  // (the initial profile is not divergence free around the baffles) act as spurious
  // sources/sinks of theta in the conservative D2Q5 equation (temperatures outside the
  // inlet range).
  void develop_flow(long nsteps)
  {
    for (long s = 0; s < nsteps; ++s) {
      step();
    }
    reset_thermal();
    reset_face_flux();
  }

  // ------------------------------------------------------------------------------------
  // Convert the physical wall heat flux in heat_flux() [W/m^2, positive into the fluid]
  // into the lattice scalar flux j = q / (rho c_p) * dt/dx used by the wall boundary rule.
  // Keeps a copy of the flux as received (mapped) from preCICE, then moves the flux of
  // faces covered by baffle roots to the nearest wetted face (sum preserved). Host only:
  // the lattice flux migrates to the device on the first step that reads it.
  void apply_heat_flux(double scale)
  {
    std::copy(qbuf_.begin(), qbuf_.end(), q_received_.begin());
    for (int f = 0; f < nc_; ++f) {
      if (iface_src_[f] != f) {
        qbuf_[iface_src_[f]] += qbuf_[f];
        qbuf_[f] = 0.0;
      }
    }
    q_sum_ = 0.0;
    for (int f = 0; f < nc_; ++f) {
      jflux_[f] = scale * qbuf_[f];
      q_sum_ += qbuf_[f];
    }
  }

  void reset_face_flux()
  {
    double *acc = face_acc_.data();
    parallel_for(2 * s_.ny, LBM_LAMBDA(int j) { acc[j] = 0.0; });
  }

  // ------------------------------------------------------------------------------------
  // One LBM time step: fused pull-stream + collide, then the inlet/outlet columns.
  void step()
  {
    const LatticeView L = view();
    parallel_for(N_, LBM_LAMBDA(int idx) { stream_collide(L, idx); });
    parallel_for(2 * s_.ny, LBM_LAMBDA(int t) { boundary_node(L, t); });
    cur_ = 1 - cur_;
  }

  // Macroscopic fields from the current populations (solid nodes keep their fixed values).
  void update_fields()
  {
    const int     N    = N_;
    const double  Tref = s_.T_ref;
    const double *f = f_[cur_].data(), *g = g_[cur_].data();
    double *rho = rho_.data(), *ux = ux_.data(), *uy = uy_.data(), *T = T_.data();
    const std::uint8_t *msk = mask_.data();
    parallel_for(N, LBM_LAMBDA(int idx) {
      if (msk[idx]) {
        return;
      }
      double r = 0.0, u = 0.0, v = 0.0, th = 0.0;
      LBM_UNROLL
      for (int k = 0; k < 9; ++k) {
        r += f[k * N + idx];
        u += d2q9::cx(k) * f[k * N + idx];
        v += d2q9::cy(k) * f[k * N + idx];
      }
      LBM_UNROLL
      for (int k = 0; k < 5; ++k) {
        th += g[k * N + idx];
      }
      rho[idx] = r;
      ux[idx]  = u;
      uy[idx]  = v;
      T[idx]   = th + Tref;
    });
  }

  // ------------------------------------------------------------------------------------
  // Wall temperature on the coupled wall (half a lattice spacing outside the first fluid
  // node):  T_w = T_0 + q dx / (2 k)  ==  T_0 + j / (2 alpha_lattice).
  // Gathered into a small contiguous buffer so only this buffer migrates to the host.
  void gather_wall_temperature()
  {
    const int     nx = s_.nx, N = N_;
    const int     row0  = coupled_row() * nx + 1; // node above face 0
    const double  Tref  = s_.T_ref;
    const double  inv2a = 1.0 / (2.0 * d2q9::diffusivity(s_.tau_g));
    const double *g     = g_[cur_].data();
    const double *jf    = jflux_.data();
    const int    *src   = iface_src_dev_.data();
    double       *Tw    = Twall_.data();
    parallel_for(nc_, LBM_LAMBDA(int f) {
      const int s   = src[f]; // == f unless the face is covered by a baffle root
      const int idx = row0 + s;
      double    th  = 0.0;
      LBM_UNROLL
      for (int k = 0; k < 5; ++k) {
        th += g[k * N + idx];
      }
      Tw[f] = Tref + th + jf[s] * inv2a;
    });
  }

  // ------------------------------------------------------------------------------------
  // Implicit-coupling checkpoints (device-side copies).
  void save_checkpoint()
  {
    copy(f_[cur_].data(), f_ckpt_.data(), 9 * N_);
    copy(g_[cur_].data(), g_ckpt_.data(), 5 * N_);
  }
  void restore_checkpoint()
  {
    copy(f_ckpt_.data(), f_[cur_].data(), 9 * N_);
    copy(g_ckpt_.data(), g_[cur_].data(), 5 * N_);
  }

private:
  LatticeView view()
  {
    LatticeView L;
    L.nx    = s_.nx;
    L.ny    = s_.ny;
    L.N     = N_;
    L.omf   = 1.0 / s_.tau_f;
    L.omg   = 1.0 / s_.tau_g;
    L.U     = s_.u_mean;
    L.T_in  = s_.T_in;
    L.Tref  = s_.T_ref;
    L.jwall = s_.coupled_bottom ? -1 : s_.ny;
    L.i_in  = inlet_column();
    L.i_out = outlet_column();
    L.fs    = f_[cur_].data();
    L.gs    = g_[cur_].data();
    L.fd    = f_[1 - cur_].data();
    L.gd    = g_[1 - cur_].data();
    L.jf    = jflux_.data();
    L.acc   = face_acc_.data();
    L.msk   = mask_.data();
    return L;
  }

  // Thermal populations in equilibrium at T_in with the current flow field; also refreshes
  // the macroscopic fields (solid nodes: at rest, T_ref).
  void reset_thermal()
  {
    const int     N    = N_;
    const double  Tref = s_.T_ref, th0 = s_.T_in - s_.T_ref;
    const double *f    = f_[cur_].data();
    double       *g    = g_[cur_].data();
    double *rho = rho_.data(), *ux = ux_.data(), *uy = uy_.data(), *T = T_.data();
    const std::uint8_t *msk = mask_.data();
    parallel_for(N, LBM_LAMBDA(int idx) {
      const bool solid = msk[idx] != 0;
      double     r = 0.0, u = 0.0, v = 0.0;
      LBM_UNROLL
      for (int k = 0; k < 9; ++k) {
        r += f[k * N + idx];
        u += d2q9::cx(k) * f[k * N + idx];
        v += d2q9::cy(k) * f[k * N + idx];
      }
      const double th = solid ? 0.0 : th0;
      LBM_UNROLL
      for (int k = 0; k < 5; ++k) {
        g[k * N + idx] = d2q9::geq(k, th, u, v);
      }
      rho[idx] = solid ? 1.0 : r;
      ux[idx]  = u;
      uy[idx]  = v;
      T[idx]   = Tref + th;
    });
  }

  static void copy(const double *src, double *dst, int n)
  {
    parallel_for(n, LBM_LAMBDA(int i) { dst[i] = src[i]; });
  }

  ChannelSetup s_;
  int          N_, nc_;
  int          cur_   = 0;
  double       q_sum_ = 0.0;

  device_vector<double> f_[2], g_[2];
  device_vector<double> rho_, ux_, uy_, T_;
  device_vector<double> f_ckpt_, g_ckpt_;
  device_vector<double> jflux_;    // lattice heat flux into the fluid per coupled face
  device_vector<double> qbuf_;     // physical heat flux read from preCICE [W/m^2], redistributed
  device_vector<double> Twall_;    // wall temperature written to preCICE [K]
  device_vector<double> face_acc_; // accumulated face fluxes for the energy budget
  device_vector<std::uint8_t> mask_;          // 1 = solid node
  device_vector<int>          iface_src_dev_; // device copy of iface_src_ for the gather kernel
  std::vector<int>            iface_src_;     // wetted face that serves interface face f
  std::vector<double>         q_received_;    // host copy of the flux as read from preCICE
};

} // namespace lbm

// channel.hpp -- one channel of the counterflow heat exchanger (masked LBM).
//
// Geometry: nx x ny lattice between two plane walls (y = 0 and y = ny), plus arbitrary
//          internal solid nodes given by a mask (baffles, ribs, ...). Every link from a
//          fluid node to a solid node or to a wall uses half-way bounce-back.
// Flow:    D2Q9 BGK, incompressible equilibrium of He & Luo (1997) so that the steady
//          velocity field is divergence free, velocity inlet (parabolic) and
//          constant-density outlet, both via non-equilibrium extrapolation (Guo et al. 2002).
// Heat:    D2Q5 BGK passive scalar (advection-diffusion of T) using the D2Q9 velocity.
//          Evolved variable theta = T - T_ref (reduces compressibility/round-off errors).
//          Outer wall and masked solids: adiabatic (bounce-back). Coupled wall: prescribed heat flux
//          (bounce-back plus source), inlet: Dirichlet T_in, outlet: zero gradient.
//
// Layout:  structure of arrays, f[k*N + j*nx + i], i (streamwise) fastest.
// Memory:  lbm::device_vector -> host/device accessible (managed or unified memory).
#pragma once

#include "parallel_for.hpp"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace lbm {

struct ChannelSetup {
  int    nx               = 0;     // lattice nodes along the channel
  int    ny               = 0;     // lattice nodes across the channel
  double u_mean           = 0.0;   // mean inlet velocity [lattice units], sign = flow direction
  double T_in             = 0.0;   // inlet temperature [K]
  double T_init           = 0.0;   // initial temperature [K]
  double T_ref            = 0.0;   // reference temperature, D2Q5 evolves T - T_ref [K]
  bool   coupled_bottom   = true;  // true: coupled wall is the bottom wall (y = 0)
  double tau_f            = 1.0;   // flow relaxation time
  double tau_g            = 1.0;   // thermal relaxation time
};

class Channel {
public:
  explicit Channel(const ChannelSetup &s)
      : s_(s), N_(s.nx * s.ny),
        f_{device_vector<double>(9 * N_), device_vector<double>(9 * N_)},
        g_{device_vector<double>(5 * N_), device_vector<double>(5 * N_)},
        rho_(N_), ux_(N_), uy_(N_), T_(N_),
        jflux_(s.nx, 0.0), qbuf_(s.nx, 0.0), Twall_(s.nx, 0.0),
        f_ckpt_(9 * N_), g_ckpt_(5 * N_), face_acc_(2 * s.ny, 0.0), mask_(N_, 0),
        iface_src_(s.nx, 0), q_received_(s.nx, 0.0)
  {
    for (int i = 0; i < s.nx; ++i) {
      iface_src_[i] = i;
    }
  }

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
    for (int idx = 0; idx < N_; ++idx) {
      mask_[idx] = m[idx];
    }
    const int jc = s_.coupled_bottom ? 0 : s_.ny - 1;
    for (int i = 0; i < s_.nx; ++i) {
      iface_src_[i] = i;
      if (!mask_[jc * s_.nx + i]) {
        continue;
      }
      for (int d = 1; d < s_.nx; ++d) {
        if (i - d >= 0 && !mask_[jc * s_.nx + i - d]) {
          iface_src_[i] = i - d;
          break;
        }
        if (i + d < s_.nx && !mask_[jc * s_.nx + i + d]) {
          iface_src_[i] = i + d;
          break;
        }
      }
    }
  }
  int covered_interface_faces() const
  {
    int n = 0;
    for (int i = 0; i < s_.nx; ++i) {
      n += iface_src_[i] != i;
    }
    return n;
  }
  const std::uint8_t *mask() const { return mask_.data(); }
  double solid_fraction() const
  {
    long n = 0;
    for (int idx = 0; idx < N_; ++idx) {
      n += mask_[idx];
    }
    return double(n) / N_;
  }

  int nx() const { return s_.nx; }
  int ny() const { return s_.ny; }
  const ChannelSetup &setup() const { return s_; }

  // Host-visible interface buffers handed to preCICE (managed/unified memory).
  double *heat_flux_buffer() { return qbuf_.data(); }        // read target  [W/m^2]
  double *wall_temperature_buffer() { return Twall_.data(); } // write source [K]

  // Field access for diagnostics/output (call device_sync() first).
  const double *T() const { return T_.data(); }
  const double *ux() const { return ux_.data(); }
  const double *uy() const { return uy_.data(); }
  const double *rho() const { return rho_.data(); }
  const double *heat_flux() const { return qbuf_.data(); }                 // as applied
  const double *heat_flux_received() const { return q_received_.data(); } // as mapped
  // Scalar flux through the faces x = 0 (between columns 0|1, entries 0..ny-1) and
  // x = L (columns nx-2|nx-1, entries ny..2ny-1), summed over all steps since the last
  // reset_face_flux(): exactly what streaming transferred, in lattice units.
  const double *face_flux_accumulated() const { return face_acc_.data(); }
  void reset_face_flux()
  {
    double *acc = face_acc_.data();
    parallel_for(2 * s_.ny, LBM_LAMBDA(int j) { acc[j] = 0.0; });
  }
  const double *wall_temperature() const { return Twall_.data(); }

  // ------------------------------------------------------------------------------------
  void initialize()
  {
    const int    nx = s_.nx, ny = s_.ny, N = N_;
    const double U = s_.u_mean, T0 = s_.T_init, Tref = s_.T_ref;
    double *f = f_[cur_].data(), *g = g_[cur_].data();
    double *rho = rho_.data(), *ux = ux_.data(), *uy = uy_.data(), *T = T_.data();
    const std::uint8_t *msk = mask_.data();

    parallel_for(N, LBM_LAMBDA(int idx) {
      const int    j     = idx / nx;
      const bool   solid = msk[idx] != 0;
      const double u     = solid ? 0.0 : inlet_profile(j, ny, U);
      const double t0    = solid ? Tref : T0;
      for (int k = 0; k < 9; ++k) {
        f[k * N + idx] = feq9(k, 1.0, u, 0.0);
      }
      for (int k = 0; k < 5; ++k) {
        g[k * N + idx] = geq5(k, t0 - Tref, u, 0.0);
      }
      rho[idx] = 1.0;
      ux[idx]  = u;
      uy[idx]  = 0.0;
      T[idx]   = t0;
    });
    device_sync();
  }

  // ------------------------------------------------------------------------------------
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
    const int    N = N_;
    const double T0 = s_.T_init, Tref = s_.T_ref;
    double       *g = g_[cur_].data(), *T = T_.data();
    const double *ux = ux_.data(), *uy = uy_.data();
    const std::uint8_t *msk = mask_.data();
    parallel_for(N, LBM_LAMBDA(int idx) {
      const double t0 = msk[idx] ? Tref : T0;
      for (int k = 0; k < 5; ++k) {
        g[k * N + idx] = geq5(k, t0 - Tref, ux[idx], uy[idx]);
      }
      T[idx] = t0;
    });
    reset_face_flux();
    device_sync();
  }

  // ------------------------------------------------------------------------------------
  // Convert the physical wall heat flux in qbuf_ [W/m^2, positive into the fluid] into
  // the lattice scalar flux j = q / (rho c_p) * dt/dx used by the wall boundary rule.
  //
  // Host part first: keep a copy of the flux as received (mapped) from preCICE, then move
  // the flux of faces covered by baffle roots to the nearest wetted face (sum preserved).
  void apply_heat_flux(double scale)
  {
    for (int i = 0; i < s_.nx; ++i) {
      q_received_[i] = qbuf_[i];
    }
    for (int i = 0; i < s_.nx; ++i) {
      if (iface_src_[i] != i) {
        qbuf_[iface_src_[i]] += qbuf_[i];
        qbuf_[i] = 0.0;
      }
    }
    const double *q = qbuf_.data();
    double       *j = jflux_.data();
    parallel_for(s_.nx, LBM_LAMBDA(int i) { j[i] = scale * q[i]; });
  }

  // ------------------------------------------------------------------------------------
  // One LBM time step: fused pull-stream + collide, then inlet/outlet columns.
  void step()
  {
    const int nx = s_.nx, ny = s_.ny, N = N_;
    const double omf = 1.0 / s_.tau_f, omg = 1.0 / s_.tau_g;
    const bool   bottom = s_.coupled_bottom;
    const double Tref   = s_.T_ref;

    const double *fs = f_[cur_].data();
    double       *fd = f_[1 - cur_].data();
    const double *gs = g_[cur_].data();
    double       *gd = g_[1 - cur_].data();
    const double *jf = jflux_.data();
    double *rho = rho_.data(), *ux = ux_.data(), *uy = uy_.data(), *T = T_.data();
    double *acc = face_acc_.data();
    const std::uint8_t *msk = mask_.data();

    // energy-budget bookkeeping: D2Q5 populations about to cross x = 0 and x = L
    parallel_for(ny, LBM_LAMBDA(int j) {
      const int a = j * nx, b = j * nx + nx - 2;
      acc[j] += gs[1 * N + a] - gs[3 * N + a + 1];
      acc[ny + j] += gs[1 * N + b] - gs[3 * N + b + 1];
    });

    parallel_for(N, LBM_LAMBDA(int idx) {
      const int cx[9]  = {0, 1, 0, -1, 0, 1, -1, -1, 1};
      const int cy[9]  = {0, 0, 1, 0, -1, 1, 1, -1, -1};
      const int op9[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};
      const int op5[5] = {0, 3, 4, 1, 2};

      const int i = idx % nx;
      const int j = idx / nx;

      if (msk[idx]) { // solid node: inactive, never read by fluid neighbours
        rho[idx] = 1.0;
        ux[idx]  = 0.0;
        uy[idx]  = 0.0;
        T[idx]   = Tref;
        return;
      }

      // ---- flow: pull streaming, half-way bounce-back at walls and solid nodes ------
      double fl[9];
      for (int k = 0; k < 9; ++k) {
        const int sj = j - cy[k];
        if (sj < 0 || sj >= ny) {
          fl[k] = fs[op9[k] * N + idx];
        } else {
          int si = i - cx[k];
          si     = si < 0 ? 0 : (si >= nx ? nx - 1 : si); // inlet/outlet columns are overwritten later
          const int nb = sj * nx + si;
          fl[k]        = msk[nb] ? fs[op9[k] * N + idx] : fs[k * N + nb];
        }
      }
      double r = 0.0, mx = 0.0, my = 0.0;
      for (int k = 0; k < 9; ++k) {
        r += fl[k];
        mx += cx[k] * fl[k];
        my += cy[k] * fl[k];
      }
      const double u = mx, v = my; // incompressible model: rho_0 = 1
      for (int k = 0; k < 9; ++k) {
        fd[k * N + idx] = fl[k] - omf * (fl[k] - feq9(k, r, u, v));
      }

      // ---- heat: D2Q5, bounce-back walls; coupled wall injects the flux j ----------
      double gl[5];
      double t = 0.0;
      for (int k = 0; k < 5; ++k) {
        const int sj = j - cy[k];
        if (sj < 0) {
          gl[k] = gs[op5[k] * N + idx] + (bottom ? jf[i] : 0.0);
        } else if (sj >= ny) {
          gl[k] = gs[op5[k] * N + idx] + (bottom ? 0.0 : jf[i]);
        } else {
          int si = i - cx[k];
          si     = si < 0 ? 0 : (si >= nx ? nx - 1 : si);
          const int nb = sj * nx + si;
          gl[k]        = msk[nb] ? gs[op5[k] * N + idx] : gs[k * N + nb]; // adiabatic solid
        }
        t += gl[k];
      }
      for (int k = 0; k < 5; ++k) {
        gd[k * N + idx] = gl[k] - omg * (gl[k] - geq5(k, t, u, v));
      }

      rho[idx] = r;
      ux[idx]  = u;
      uy[idx]  = v;
      T[idx]   = t + Tref;
    });

    // ---- inlet / outlet columns: non-equilibrium extrapolation (post-collision) -------
    const double U       = s_.u_mean;
    const double T_in    = s_.T_in;
    const int    i_inlet = U >= 0.0 ? 0 : nx - 1;
    const int    i_out   = U >= 0.0 ? nx - 1 : 0;

    parallel_for(2 * ny, LBM_LAMBDA(int t) {
      const int  j      = t % ny;
      const bool inlet  = t < ny;
      const int  ib     = inlet ? i_inlet : i_out;
      const int  in     = ib == 0 ? 1 : nx - 2;
      const int  idb    = j * nx + ib;
      const int  idn    = j * nx + in;
      const double rn = rho[idn], un = ux[idn], vn = uy[idn], Tn = T[idn] - Tref;

      double rb, ub, vb, Tb;
      if (inlet) {
        rb = rn;
        ub = inlet_profile(j, ny, U);
        vb = 0.0;
        Tb = T_in - Tref;
      } else {
        rb = 1.0;
        ub = un;
        vb = vn;
        Tb = Tn;
      }
      for (int k = 0; k < 9; ++k) {
        fd[k * N + idb] = feq9(k, rb, ub, vb) + (fd[k * N + idn] - feq9(k, rn, un, vn));
      }
      for (int k = 0; k < 5; ++k) {
        gd[k * N + idb] = geq5(k, Tb, ub, vb) + (gd[k * N + idn] - geq5(k, Tn, un, vn));
      }
      rho[idb] = rb;
      ux[idb]  = ub;
      uy[idb]  = vb;
      T[idb]   = Tb + Tref;
    });

    cur_ = 1 - cur_;
  }

  // ------------------------------------------------------------------------------------
  // Wall temperature on the coupled wall (half a lattice spacing outside the first fluid
  // node):  T_w = T_0 + q dx / (2 k)  ==  T_0 + j / (2 alpha_lattice).
  // Gathered into a small contiguous buffer so only this buffer migrates to the host.
  void gather_wall_temperature()
  {
    const int     nx    = s_.nx;
    const int     jrow  = s_.coupled_bottom ? 0 : s_.ny - 1;
    const double  inv2a = 1.0 / (2.0 * (s_.tau_g - 0.5) / 3.0);
    const double *T     = T_.data();
    const double *jf    = jflux_.data();
    const int    *src   = iface_src_.data();
    double       *Tw    = Twall_.data();
    parallel_for(nx, LBM_LAMBDA(int i) {
      const int s = src[i]; // == i unless the face is covered by a baffle root
      Tw[i]       = T[jrow * nx + s] + jf[s] * inv2a;
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

  // ------------------------------------------------------------------------------------
  static LBM_HD double inlet_profile(int j, int ny, double U)
  {
    const double y = (j + 0.5) / ny; // half-way walls at y = 0 and y = 1
    return 6.0 * U * y * (1.0 - y);
  }

  static LBM_HD double feq9(int k, double r, double u, double v)
  {
    const int    cx[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
    const int    cy[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
    const double w     = k == 0 ? 4.0 / 9.0 : (k < 5 ? 1.0 / 9.0 : 1.0 / 36.0);
    const double cu    = cx[k] * u + cy[k] * v;
    return w * (r + 3.0 * cu + 4.5 * cu * cu - 1.5 * (u * u + v * v)); // He & Luo, rho_0 = 1
  }

  // D2Q5 equilibrium for theta = T - T_ref, with w0 = 1/3, w1..4 = 1/6  ->  c_s^2 = 1/3,  alpha = (tau_g - 1/2)/3
  static LBM_HD double geq5(int k, double T, double u, double v)
  {
    const int    cx[5] = {0, 1, 0, -1, 0};
    const int    cy[5] = {0, 0, 1, 0, -1};
    const double w     = k == 0 ? 1.0 / 3.0 : 1.0 / 6.0;
    return w * T * (1.0 + 3.0 * (cx[k] * u + cy[k] * v));
  }

private:
  static void copy(const double *src, double *dst, int n)
  {
    parallel_for(n, LBM_LAMBDA(int i) { dst[i] = src[i]; });
  }

  ChannelSetup s_;
  int          N_;
  int          cur_ = 0;

  device_vector<double> f_[2], g_[2];
  device_vector<double> rho_, ux_, uy_, T_;
  device_vector<double> jflux_; // lattice heat flux into the fluid at the coupled wall
  device_vector<double> qbuf_;  // physical heat flux read from preCICE [W/m^2]
  device_vector<double> Twall_; // wall temperature written to preCICE [K]
  device_vector<double> f_ckpt_, g_ckpt_;
  device_vector<double> face_acc_; // accumulated face fluxes for the energy budget
  device_vector<std::uint8_t> mask_; // 1 = solid node
  device_vector<int>          iface_src_;  // wetted face that serves interface face i
  std::vector<double>         q_received_; // host copy of the flux read from preCICE
};

} // namespace lbm

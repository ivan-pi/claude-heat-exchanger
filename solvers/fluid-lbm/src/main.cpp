// fluid-lbm: Code A of the coupled heat-exchanger prototype.
//
// Two counterflow channels (hot on top, cold at the bottom) separated by a solid wall
// that is simulated by the Fortran participant "Solid". Each channel has one coupled
// wall (the one facing the solid) and one adiabatic outer wall.
//
//   y = 2H+d  ---------------------------------------------  adiabatic
//             hot channel   -->  u_hot,  T_in_hot at x = 0
//   y = H+d   =============  Fluid-Hot-Mesh / Solid-Hot-Mesh  =============
//             solid wall (Code B)
//   y = H     =============  Fluid-Cold-Mesh / Solid-Cold-Mesh =============
//             cold channel  <--  u_cold, T_in_cold at x = L
//   y = 0     ---------------------------------------------  adiabatic
//
// Coupling (Dirichlet-Neumann): Fluid reads Heat-Flux (W/m^2, positive into the fluid),
// writes the wall Temperature (K). See README.md for why the insulating wall takes the
// Dirichlet side.
//
// Usage: fluid-lbm <precice-config.xml> <params.txt> [output-dir]

#include "channel.hpp"
#include "params.hpp"
#include "parallel_for.hpp"

#include <precice/precice.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Physical {
  double L, H, d;        // channel length, channel height, wall thickness [m]
  double rho, cp, k, nu; // fluid properties
  double u_hot, u_cold;  // mean velocities [m/s]
  double T_hot, T_cold;  // inlet temperatures [K]
  double dx, dt;         // lattice spacing and time step
  double alpha() const { return k / (rho * cp); }
};

// Per-channel energy budget per unit depth, evaluated on the device.
// Control volume: the coupled length 0 <= x <= L, i.e. lattice columns 1 .. nx-2; the
// inlet/outlet boundary columns 0 and nx-1 lie half a spacing outside. Hence
//     dE/dt = H_net_in + Q_wall
// with H the enthalpy flow (advective + diffusive, relative to T_ref) through the faces
// x = 0 and x = L, and Q_wall the heat received through the coupled wall.
struct Budget {
  double E        = 0; // stored thermal energy  rho c_p int (T - T_ref) dA   [J/m]
  double H_net_in = 0; // enthalpy flow in minus out through x = 0 and x = L  [W/m]
  double Q_wall   = 0; // heat received through the coupled wall              [W/m]
  double Tb_out   = 0; // bulk (mixing-cup) outlet temperature                [K]
};

// One channel with everything the coupling and the output need to know about it.
struct Side {
  const char      *tag;     // "hot" / "cold": log columns and file names
  std::string      mesh;    // preCICE mesh name
  double           y0;      // y of the channel's lower wall [m]
  double           y_iface; // y of the coupled wall [m]
  lbm::Channel     channel;
  std::vector<int> ids;     // preCICE vertex ids of the coupled faces
  Budget           last;    // budget at the end of the last completed window
};

// Field sums of one reduction pass: sum (T - T_ref) over the coupled columns, and
// sum u T, sum u over the last interior column before the outlet.
struct FieldSums {
  double T, uT, u;
  LBM_HD FieldSums operator+(const FieldSums &o) const { return {T + o.T, uT + o.uT, u + o.u}; }
};

// Budget over the window just completed; the channel's fields must be up to date.
Budget budget(const lbm::Channel &c, const Physical &p, long nsteps)
{
  const int     nx = c.nx(), ny = c.ny(), nc = c.interface_size();
  const double *T = c.T(), *u = c.ux();
  const double  Tref = c.setup().T_ref;
  const int     io   = c.outlet_column() == 0 ? 1 : nx - 2; // last interior column before the outlet

  const FieldSums s = lbm::parallel_reduce(nc * ny, LBM_LAMBDA(int t) -> FieldSums {
    const int  i = 1 + t % nc, j = t / nc, idx = j * nx + i;
    const bool out = i == io;
    return {T[idx] - Tref, out ? u[idx] * T[idx] : 0.0, out ? u[idx] : 0.0};
  });

  Budget b;
  b.E      = p.rho * p.cp * p.dx * p.dx * s.T;
  b.Tb_out = s.uT / s.u;
  b.Q_wall = p.dx * c.wall_heat_sum();
  // Enthalpy flow through x = 0 and x = L, averaged over the nsteps LBM steps of the
  // window, from the populations that actually streamed across the faces. Together with
  // the wall flux (constant within a window) this closes the budget to round-off.
  if (nsteps > 0) {
    const double *acc = c.face_flux_accumulated();
    const double  net = lbm::parallel_reduce(ny, LBM_LAMBDA(int j) { return acc[j] - acc[ny + j]; });
    b.H_net_in        = p.rho * p.cp * p.dx * p.dx / p.dt * net / nsteps;
  }
  return b;
}

// Baffle geometry: n_baffles thin plates, evenly spaced, `height` cells tall.
//   alternating = false : all plates on the OUTER (adiabatic) wall; the stream passes
//                         through the gap along the coupled wall, cavities in between.
//   alternating = true  : plates alternate between the outer and the coupled wall
//                         (first one on the outer wall) -> serpentine / zig-zag flow,
//                         inverted-U loops over the plates standing on the lower wall.
// `shift` (fraction of a pitch) staggers the two channels. The plates stay at least two
// columns away from the inlet/outlet columns. Replace this function (or load a mask from
// file) for any other geometry.
std::vector<std::uint8_t> make_baffles(int nx, int ny, bool coupled_bottom, int n_baffles, int height,
                                       int thickness, double shift, bool alternating)
{
  std::vector<std::uint8_t> m(static_cast<std::size_t>(nx) * ny, 0);
  const int nc = nx - 2; // coupled columns 1..nc
  for (int b = 0; b < n_baffles; ++b) {
    const double xc = (b + 1.0 + shift) * nc / (n_baffles + 1.0); // centre, in coupled-column units
    const int    i0 = 1 + static_cast<int>(std::lround(xc - 0.5 * thickness));
    for (int i = std::max(i0, 3); i < std::min(i0 + thickness, nx - 3); ++i) {
      const bool on_outer = !alternating || b % 2 == 0;
      const bool from_top = on_outer == coupled_bottom; // outer wall is on top if coupled at bottom
      for (int h = 0; h < height; ++h) {
        const int j = from_top ? ny - 1 - h : h;
        m[static_cast<std::size_t>(j) * nx + i] = 1;
      }
    }
  }
  return m;
}

void write_vtk(const std::string &file, const lbm::Channel &c, const Physical &p, double y0)
{
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const int    n   = c.nx() * c.ny();
  std::ofstream out(file);
  out << "# vtk DataFile Version 3.0\nfluid-lbm channel\nASCII\nDATASET STRUCTURED_POINTS\n";
  out << "DIMENSIONS " << c.nx() << ' ' << c.ny() << " 1\n";
  out << std::setprecision(10);
  out << "ORIGIN " << -0.5 * p.dx << ' ' << y0 + 0.5 * p.dx << " 0\n"; // column 0 = inlet/outlet node
  out << "SPACING " << p.dx << ' ' << p.dx << ' ' << p.dx << '\n';
  out << "POINT_DATA " << n << '\n';
  out << "SCALARS T double 1\nLOOKUP_TABLE default\n";
  for (int idx = 0; idx < n; ++idx) {
    out << (c.mask()[idx] ? nan : c.T()[idx]) << '\n'; // NaN marks solid nodes
  }
  out << "SCALARS solid int 1\nLOOKUP_TABLE default\n";
  for (int idx = 0; idx < n; ++idx) {
    out << int(c.mask()[idx]) << '\n';
  }
  const double us = p.dx / p.dt;
  out << "VECTORS velocity double\n";
  for (int idx = 0; idx < n; ++idx) {
    out << c.ux()[idx] * us << ' ' << c.uy()[idx] * us << " 0\n";
  }
}

std::string vtk_name(const std::string &outdir, const char *tag, int count)
{
  std::ostringstream name;
  name << outdir << "/fluid_" << tag << '_' << std::setw(4) << std::setfill('0') << count << ".vtk";
  return name.str();
}

} // namespace

int main(int argc, char **argv)
{
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <precice-config.xml> <params.txt> [output-dir]\n", argv[0]);
    return 1;
  }
  const std::string config = argv[1];
  const std::string outdir = argc > 3 ? argv[3] : "output";
  std::filesystem::create_directories(outdir);

  // ------------------------------------------------------------------ parameters ------
  lbm::Params prm(argv[2]);
  Physical    p{};
  p.L      = prm.real("channel_length");
  p.H      = prm.real("channel_height");
  p.d      = prm.real("wall_thickness");
  p.rho    = prm.real("fluid_density");
  p.cp     = prm.real("fluid_cp");
  p.k      = prm.real("fluid_conductivity");
  p.nu     = prm.real("fluid_viscosity");
  p.u_hot  = prm.real("u_mean_hot");
  p.u_cold = prm.real("u_mean_cold");
  p.T_hot  = prm.real("T_in_hot");
  p.T_cold = prm.real("T_in_cold");
  const int ny = prm.integer("lbm_ny");
  p.dt         = prm.real("lbm_dt");
  p.dx         = p.H / ny;
  // nc lattice columns cover the coupled length [0, L]; one extra boundary column on each
  // side carries the inlet/outlet condition (x = -dx/2 and x = L + dx/2).
  const int    nc              = static_cast<int>(std::lround(p.L / p.dx));
  const int    nx              = nc + 2;
  const double output_interval = prm.real("output_interval");

  const double tau_f = lbm::d2q9::relaxation_time(p.nu * p.dt / (p.dx * p.dx));
  const double tau_g = lbm::d2q9::relaxation_time(p.alpha() * p.dt / (p.dx * p.dx));
  const double uh_l  = p.u_hot * p.dt / p.dx;
  const double uc_l  = p.u_cold * p.dt / p.dx;

  std::printf("[fluid] backend           : %s\n", lbm::backend_name());
  std::printf("[fluid] lattice           : 2 x (%d x %d) incl. inlet/outlet columns, dx = %.4g m, dt = %.4g s\n", nx, ny, p.dx, p.dt);
  std::printf("[fluid] tau_f, tau_g      : %.4f, %.4f   (Pr = %.3g)\n", tau_f, tau_g, p.nu / p.alpha());
  std::printf("[fluid] u_max (lattice)   : %.4f, %.4f   (Re = %.3g, %.3g)\n", 1.5 * uh_l, 1.5 * uc_l,
              p.u_hot * p.H / p.nu, p.u_cold * p.H / p.nu);
  // baffles (masked solid nodes); baffle_count = 0 gives straight channels
  const int    n_baffles = prm.integer_or("baffle_count", 0);
  const bool   b_alt     = prm.integer_or("baffle_alternating", 1) != 0;
  const int    b_height  = static_cast<int>(std::lround(prm.real_or("baffle_height", 0.0) * ny));
  const int    b_thick   = std::max(1, static_cast<int>(std::lround(prm.real_or("baffle_thickness", p.dx) / p.dx)));
  const double contract  = n_baffles > 0 ? double(ny) / (ny - b_height) : 1.0; // gap velocity factor
  const double u_peak    = 1.5 * std::max(uh_l, uc_l) * contract;
  std::printf("[fluid] baffles           : %d per channel (%s), %d x %d cells (height x thickness)\n",
              n_baffles, b_alt ? "alternating" : "outer wall", b_height, b_thick);
  std::printf("[fluid] u_max in gap (est): %.4f (lattice)\n", u_peak);
  if (b_height >= ny - 2) {
    throw std::runtime_error("baffle_height leaves no gap along the coupled wall");
  }
  if (tau_f < 0.505 || tau_g < 0.505 || u_peak > 0.15) {
    std::fprintf(stderr, "[fluid] WARNING: lattice parameters outside the comfortable stability range\n");
  }

  lbm::ChannelSetup hs;
  hs.nx             = nx;
  hs.ny             = ny;
  hs.u_mean         = +uh_l; // hot stream flows in +x
  hs.T_in           = p.T_hot;
  hs.coupled_bottom = true;
  hs.tau_f          = tau_f;
  hs.tau_g          = tau_g;
  hs.T_ref          = 0.5 * (p.T_hot + p.T_cold);

  lbm::ChannelSetup cs = hs;
  cs.u_mean         = -uc_l; // cold stream flows in -x (counterflow)
  cs.T_in           = p.T_cold;
  cs.coupled_bottom = false;

  std::array<Side, 2> sides{{{"hot", "Fluid-Hot-Mesh", p.H + p.d, p.H + p.d, lbm::Channel(hs), {}, {}},
                             {"cold", "Fluid-Cold-Mesh", 0.0, p.H, lbm::Channel(cs), {}, {}}}};

  for (std::size_t s = 0; s < sides.size(); ++s) {
    lbm::Channel &c = sides[s].channel;
    // the two channels are staggered by half a pitch
    c.set_mask(make_baffles(nx, ny, c.setup().coupled_bottom, n_baffles, b_height, b_thick, 0.5 * s, b_alt));
    std::printf("[fluid] %-4s channel      : %d interface faces under baffle roots (flux passed to nearest wetted face),"
                " solid fraction %.2f %%\n",
                sides[s].tag, c.covered_interface_faces(), 100 * c.solid_fraction());
    if (c.solid_fraction() > 0.10) {
      throw std::runtime_error("solid fraction above 10 %: the masked lattice wastes too many nodes");
    }
    c.initialize();
  }
  {
    // flow-only start-up phase (thermal field reset afterwards), not part of the coupled time
    const double t_init = prm.real_or("lbm_flow_init_time", 2.0);
    const long   n_init = std::lround(t_init / p.dt);
    std::printf("[fluid] developing the flow: %ld LBM steps (%.2f s) before coupling\n", n_init, t_init);
    for (auto &s : sides) {
      s.channel.develop_flow(n_init);
    }
  }

  // ------------------------------------------------------------------ preCICE ---------
  precice::Participant participant("Fluid", config, 0, 1);

  for (auto &s : sides) {
    // interface vertices at the wall faces of the coupled columns 1..nc
    s.ids.resize(nc);
    std::vector<double> coords(2 * nc);
    for (int i = 0; i < nc; ++i) {
      coords[2 * i]     = (i + 0.5) * p.dx;
      coords[2 * i + 1] = s.y_iface;
    }
    participant.setMeshVertices(s.mesh, coords, s.ids);
    std::vector<int> edges(2 * (nc - 1));
    for (int i = 0; i < nc - 1; ++i) {
      edges[2 * i]     = s.ids[i];
      edges[2 * i + 1] = s.ids[i + 1];
    }
    participant.setMeshEdges(s.mesh, edges);
  }

  const auto n_faces = static_cast<std::size_t>(nc);

  auto write_wall_temperature = [&] {
    for (auto &s : sides) {
      s.channel.gather_wall_temperature();
    }
    lbm::device_sync(); // host is about to read managed memory
    for (auto &s : sides) {
      participant.writeData(s.mesh, "Temperature", s.ids, {s.channel.wall_temperature(), n_faces});
    }
  };

  if (participant.requiresInitialData()) {
    write_wall_temperature();
  }
  participant.initialize();

  // q [W/m^2] -> lattice scalar flux j = q/(rho c_p) * dt/dx
  const double flux_scale = p.dt / (p.rho * p.cp * p.dx);

  double time = 0.0, next_output = output_interval;
  int    window = 0, iterations = 0, output_count = 0;
  long   lbm_steps_total = 0;

  std::ofstream elog(outdir + "/fluid_energy.csv");
  elog << "time";
  for (const auto &s : sides) {
    elog << ',' << s.tag << "_H_net_in," << s.tag << "_Q_wall," << s.tag << "_dEdt," << s.tag << "_Tb_out";
  }
  elog << '\n' << std::setprecision(12);

  // Budgets of the window just completed (fields refreshed), one CSV line.
  auto log_energy = [&](long nsub) {
    elog << time;
    for (auto &s : sides) {
      s.channel.update_fields();
      const Budget b = budget(s.channel, p, nsub);
      elog << ',' << b.H_net_in << ',' << b.Q_wall << ',' << (b.E - s.last.E) / (nsub * p.dt) << ',' << b.Tb_out;
      s.last = b;
    }
    elog << '\n';
  };

  // Console summary of the last budgets and VTK snapshots of the current fields.
  auto diagnostics_and_output = [&] {
    std::printf("[fluid] t = %7.3f s  window %5d", time, window);
    for (const auto &s : sides) {
      std::printf(" | %s: Tb_out = %7.3f K, H_in-H_out = %9.4f W/m, Q_wall = %9.4f W/m", s.tag, s.last.Tb_out,
                  s.last.H_net_in, s.last.Q_wall);
    }
    std::printf("\n");
    lbm::device_sync();
    for (const auto &s : sides) {
      write_vtk(vtk_name(outdir, s.tag, output_count), s.channel, p, s.y0);
    }
    ++output_count;
  };

  for (auto &s : sides) {
    s.last = budget(s.channel, p, 0); // fields are valid after develop_flow()
  }
  diagnostics_and_output();

  while (participant.isCouplingOngoing()) {
    if (participant.requiresWritingCheckpoint()) {
      for (auto &s : sides) {
        s.channel.save_checkpoint();
      }
    }

    // The fluid subcycles with its own lattice time step inside one preCICE step.
    const double dt_window = participant.getMaxTimeStepSize();
    const long   nsub      = std::lround(dt_window / p.dt);
    if (nsub < 1 || std::abs(nsub * p.dt - dt_window) > 1e-9 * dt_window) {
      throw std::runtime_error("time-window-size must be an integer multiple of lbm_dt");
    }

    // Read the (end-of-window) wall heat flux directly into the managed buffers; the
    // lattice flux migrates to the device on the first step that touches it.
    for (auto &s : sides) {
      participant.readData(s.mesh, "Heat-Flux", s.ids, dt_window, {s.channel.heat_flux(), n_faces});
      s.channel.apply_heat_flux(flux_scale);
      s.channel.reset_face_flux();
    }
    for (long i = 0; i < nsub; ++i) {
      for (auto &s : sides) {
        s.channel.step();
      }
    }
    lbm_steps_total += nsub;
    write_wall_temperature();

    participant.advance(nsub * p.dt);
    ++iterations;

    if (participant.requiresReadingCheckpoint()) {
      for (auto &s : sides) {
        s.channel.restore_checkpoint();
      }
    } else {
      time += nsub * p.dt;
      ++window;
      log_energy(nsub);
      if (time >= next_output - 1e-12 || !participant.isCouplingOngoing()) {
        diagnostics_and_output();
        next_output += output_interval;
      }
    }
  }

  // Final interface profiles (wall temperature and heat flux, both interfaces).
  lbm::device_sync();
  {
    std::ofstream out(outdir + "/fluid_interface.csv");
    // q_*: flux as mapped from the solid; q_*_applied: after moving the flux of faces
    // covered by baffle roots to the nearest wetted face (same integral)
    out << "x";
    for (const auto &s : sides) {
      out << ",Tw_" << s.tag << ",q_" << s.tag;
    }
    for (const auto &s : sides) {
      out << ",q_" << s.tag << "_applied";
    }
    out << '\n' << std::setprecision(10);
    for (int f = 0; f < nc; ++f) {
      out << (f + 0.5) * p.dx;
      for (const auto &s : sides) {
        out << ',' << s.channel.wall_temperature()[f] << ',' << s.channel.heat_flux_received()[f];
      }
      for (const auto &s : sides) {
        out << ',' << s.channel.heat_flux()[f];
      }
      out << '\n';
    }
  }

  std::printf("[fluid] done: %d windows, %d coupling iterations (%.2f per window), %ld LBM steps per channel (incl. repeated iterations)\n",
              window, iterations, window > 0 ? double(iterations) / window : 0.0, lbm_steps_total);
  participant.finalize();
  return 0;
}

!> solid-fdm: Code B of the coupled heat-exchanger prototype.
!>
!> Conduction in the insulating wall between the two channels, coupled through preCICE
!> on two interface meshes:
!>   Solid-Hot-Mesh  at y = H + d   (top of the wall,    hot channel side)
!>   Solid-Cold-Mesh at y = H       (bottom of the wall, cold channel side)
!> Reads Temperature (Dirichlet), writes Heat-Flux (W/m^2, positive from wall into fluid).
!>
!> Usage: solid-fdm <precice-config.xml> <params.txt> [output-dir]
program solid_main
  use, intrinsic :: iso_fortran_env, only: dp => real64, output_unit
  use precice
  use case_params
  use heat_solver
  implicit none

  character(len=*), parameter :: participant = 'Solid'
  character(len=*), parameter :: mesh_hot = 'Solid-Hot-Mesh', mesh_cold = 'Solid-Cold-Mesh'
  character(len=*), parameter :: d_temp = 'Temperature', d_flux = 'Heat-Flux'

  character(len=512) :: config, params, outdir, fname
  integer  :: nx, ny, nv, i, unit, maxit, ongoing, req, cg_its, cg_its_window, window, iters, nout, elog
  integer, allocatable :: ids_hot(:), ids_cold(:), edges(:)
  real(dp), allocatable :: coords(:)
  real(dp) :: L, H, d, ks, rhoc, T0, dt_solid, tol, out_dt
  logical  :: converged
  real(dp) :: dt, dt_max, time, next_out, relres, E_old, E_new, dEdt, Qh, Qc

  call get_command_argument(1, config)
  call get_command_argument(2, params)
  call get_command_argument(3, outdir)
  if (len_trim(config) == 0 .or. len_trim(params) == 0) then
    write(output_unit, '(a)') 'usage: solid-fdm <precice-config.xml> <params.txt> [output-dir]'
    error stop 1
  end if
  if (len_trim(outdir) == 0) outdir = 'output'
  call execute_command_line('mkdir -p '//trim(outdir))

  ! ------------------------------------------------------------- parameters ---------
  L        = param_real(params, 'channel_length')
  H        = param_real(params, 'channel_height')
  d        = param_real(params, 'wall_thickness')
  ks       = param_real(params, 'solid_conductivity')
  rhoc     = param_real(params, 'solid_density') * param_real(params, 'solid_cp')
  T0       = param_real(params, 'solid_T_init')
  nx       = param_int(params, 'solid_nx')
  ny       = param_int(params, 'solid_ny')
  dt_solid = param_real(params, 'solid_dt')
  tol      = param_real(params, 'cg_tol')
  maxit    = param_int(params, 'cg_maxit')
  out_dt   = param_real(params, 'output_interval')

  call solver_init(nx, ny, L, d, H, ks, rhoc, T0)
  nv = nx + 1
  write(output_unit, '(a,i0,a,i0,a,es10.3,a,es10.3,a)') '[solid] grid ', nx, ' x ', ny, &
       ' intervals, dx = ', L / nx, ' m, dy = ', d / ny, ' m'

  ! ------------------------------------------------------------- preCICE ------------
  call precicef_create(participant, trim(config), 0, 1, len(participant), len_trim(config))

  allocate(coords(2 * nv), ids_hot(nv), ids_cold(nv), edges(2 * nx))
  do i = 0, nx
    coords(2 * i + 1) = i * (L / nx)
    coords(2 * i + 2) = H + d
  end do
  call precicef_set_mesh_vertices(mesh_hot, nv, coords, ids_hot, len(mesh_hot))
  do i = 0, nx
    coords(2 * i + 2) = H
  end do
  call precicef_set_mesh_vertices(mesh_cold, nv, coords, ids_cold, len(mesh_cold))

  do i = 1, nx
    edges(2 * i - 1) = ids_hot(i);  edges(2 * i) = ids_hot(i + 1)
  end do
  call precicef_set_mesh_edges(mesh_hot, nx, edges, len(mesh_hot))
  do i = 1, nx
    edges(2 * i - 1) = ids_cold(i);  edges(2 * i) = ids_cold(i + 1)
  end do
  call precicef_set_mesh_edges(mesh_cold, nx, edges, len(mesh_cold))

  call precicef_requires_initial_data(req)
  if (req == 1) then
    call precicef_write_data(mesh_hot, d_flux, nv, ids_hot, q_hot, len(mesh_hot), len(d_flux))
    call precicef_write_data(mesh_cold, d_flux, nv, ids_cold, q_cold, len(mesh_cold), len(d_flux))
  end if

  call precicef_initialize()

  ! energy log: one line per completed time window
  open(newunit=elog, file=trim(outdir)//'/solid_energy.csv', status='replace', action='write')
  write(elog, '(a)') 'time,Q_into_hot_fluid,Q_into_cold_fluid,dEdt_solid,cg_iterations'

  time = 0;  window = 0;  iters = 0;  nout = 0;  cg_its_window = 0
  next_out = out_dt
  E_old = solver_energy()
  call output()

  call precicef_is_coupling_ongoing(ongoing)
  do while (ongoing /= 0)
    call precicef_requires_writing_checkpoint(req)
    if (req == 1) then
      call solver_save_checkpoint()
      cg_its_window = 0
    end if

    call precicef_get_max_time_step_size(dt_max)
    dt = min(dt_solid, dt_max)

    ! Dirichlet data at the end of the step (backward Euler)
    call precicef_read_data(mesh_hot, d_temp, nv, ids_hot, dt, Tb_hot, len(mesh_hot), len(d_temp))
    call precicef_read_data(mesh_cold, d_temp, nv, ids_cold, dt, Tb_cold, len(mesh_cold), len(d_temp))
    call solver_set_interface_temperature()

    call solver_step(dt, tol, maxit, cg_its, relres, converged)
    cg_its_window = cg_its_window + cg_its
    if (.not. converged) then
      write(output_unit, '(a,i0,a,es10.3)') '[solid] WARNING: PCG not converged after ', cg_its, &
           ' iterations, rel. residual ', relres
    end if

    call solver_interface_flux(dt)
    call precicef_write_data(mesh_hot, d_flux, nv, ids_hot, q_hot, len(mesh_hot), len(d_flux))
    call precicef_write_data(mesh_cold, d_flux, nv, ids_cold, q_cold, len(mesh_cold), len(d_flux))

    call precicef_advance(dt)
    iters = iters + 1

    call precicef_requires_reading_checkpoint(req)
    if (req == 1) then
      call solver_restore_checkpoint()
    else
      time = time + dt
      window = window + 1
      E_new = solver_energy()
      dEdt  = (E_new - E_old) / dt
      E_old = E_new
      Qh = boundary_heat(q_hot)
      Qc = boundary_heat(q_cold)
      write(elog, '(4(es20.12,","),i0)') time, Qh, Qc, dEdt, cg_its_window
      call precicef_is_coupling_ongoing(ongoing)
      if (time >= next_out - 1.0e-12_dp .or. ongoing == 0) then
        ! discrete conservation check: dE/dt = heat in = -(Qh + Qc)
        write(output_unit, '(a,f8.3,a,i5,a,f10.4,a,f10.4,a,es10.3,a,i0)') '[solid] t = ', time, &
             ' s  window ', window, ' | Q->hot = ', Qh, ' W/m, Q->cold = ', Qc, &
             ' W/m, dE/dt+Qh+Qc = ', dEdt + Qh + Qc, ' W/m | PCG its ', cg_its
        call output()
        next_out = next_out + out_dt
      end if
      cycle
    end if
    call precicef_is_coupling_ongoing(ongoing)
  end do

  close(elog)

  ! final interface profiles on the solid mesh
  open(newunit=unit, file=trim(outdir)//'/solid_interface.csv', status='replace', action='write')
  write(unit, '(a)') 'x,T_hot,q_hot,T_cold,q_cold'
  do i = 0, nx
    write(unit, '(4(es20.12,","),es20.12)') i * (L / nx), Tb_hot(i), q_hot(i), Tb_cold(i), q_cold(i)
  end do
  close(unit)

  write(output_unit, '(a,i0,a,i0,a,f6.2,a)') '[solid] done: ', window, ' windows, ', iters, &
       ' coupling iterations (', real(iters, dp) / max(window, 1), ' per window)'
  call precicef_finalize()
  call solver_finalize()

contains

  subroutine output()
    write(fname, '(a,"/solid_",i4.4,".vtk")') trim(outdir), nout
    call solver_write_vtk(trim(fname))
    nout = nout + 1
  end subroutine output

  !> Heat flow through an interface per unit depth [W/m] (trapezoidal rule).
  real(dp) function boundary_heat(q)
    real(dp), intent(in) :: q(0:)
    boundary_heat = (sum(q) - 0.5_dp * (q(0) + q(nx))) * (L / nx)
  end function boundary_heat

end program solid_main

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
  character(len=15), parameter :: mesh(HOT:COLD) = ['Solid-Hot-Mesh ', 'Solid-Cold-Mesh']
  character(len=*), parameter :: d_temp = 'Temperature', d_flux = 'Heat-Flux'

  character(len=512) :: config, params, outdir
  integer  :: nx, ny, nv, i, s, unit, maxit, ongoing, req, cg_its, cg_its_window, window, iters, nout, elog
  integer, allocatable :: ids(:, :)
  real(dp) :: L, H, d, ks, rhoc, T0, dt_solid, tol, out_dt, dxs, y_iface(HOT:COLD)
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
  call params_load(params)
  L        = param_real('channel_length')
  H        = param_real('channel_height')
  d        = param_real('wall_thickness')
  ks       = param_real('solid_conductivity')
  rhoc     = param_real('solid_density') * param_real('solid_cp')
  T0       = param_real('solid_T_init')
  nx       = param_int('solid_nx')
  ny       = param_int('solid_ny')
  dt_solid = param_real('solid_dt')
  tol      = param_real('cg_tol')
  maxit    = param_int('cg_maxit')
  out_dt   = param_real('output_interval')

  call solver_init(nx, ny, L, d, H, ks, rhoc, T0)
  dxs = solver_dx()
  nv  = nx + 1
  write(output_unit, '(a,i0,a,i0,a,es10.3,a,es10.3,a)') '[solid] grid ', nx, ' x ', ny, &
       ' intervals, dx = ', dxs, ' m, dy = ', d / ny, ' m'

  ! ------------------------------------------------------------- preCICE ------------
  call precicef_create(participant, trim(config), 0, 1, len(participant), len_trim(config))

  y_iface = [H + d, H]
  allocate(ids(nv, HOT:COLD))
  do s = HOT, COLD
    call define_mesh(s)
  end do

  call precicef_requires_initial_data(req)
  if (req == 1) call write_flux()

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
    do s = HOT, COLD
      call precicef_read_data(trim(mesh(s)), d_temp, nv, ids(:, s), dt, Tb(:, s), len_trim(mesh(s)), len(d_temp))
    end do

    call solver_step(dt, tol, maxit, cg_its, relres, converged)
    cg_its_window = cg_its_window + cg_its
    if (.not. converged) then
      write(output_unit, '(a,i0,a,es10.3)') '[solid] WARNING: PCG not converged after ', cg_its, &
           ' iterations, rel. residual ', relres
    end if

    call solver_interface_flux(dt)
    call write_flux()

    call precicef_advance(dt)
    iters = iters + 1
    call precicef_is_coupling_ongoing(ongoing)

    call precicef_requires_reading_checkpoint(req)
    if (req == 1) then
      call solver_restore_checkpoint()
    else
      time = time + dt
      window = window + 1
      E_new = solver_energy()
      dEdt  = (E_new - E_old) / dt
      E_old = E_new
      Qh = boundary_heat(q(:, HOT))
      Qc = boundary_heat(q(:, COLD))
      write(elog, '(4(es20.12,","),i0)') time, Qh, Qc, dEdt, cg_its_window
      if (time >= next_out - 1.0e-12_dp .or. ongoing == 0) then
        ! discrete conservation check: dE/dt = heat in = -(Qh + Qc)
        write(output_unit, '(a,f8.3,a,i5,a,f10.4,a,f10.4,a,es10.3,a,i0)') '[solid] t = ', time, &
             ' s  window ', window, ' | Q->hot = ', Qh, ' W/m, Q->cold = ', Qc, &
             ' W/m, dE/dt+Qh+Qc = ', dEdt + Qh + Qc, ' W/m | PCG its ', cg_its
        call output()
        next_out = next_out + out_dt
      end if
    end if
  end do

  close(elog)

  ! final interface profiles on the solid mesh
  open(newunit=unit, file=trim(outdir)//'/solid_interface.csv', status='replace', action='write')
  write(unit, '(a)') 'x,T_hot,q_hot,T_cold,q_cold'
  do i = 0, nx
    write(unit, '(4(es20.12,","),es20.12)') i * dxs, Tb(i, HOT), q(i, HOT), Tb(i, COLD), q(i, COLD)
  end do
  close(unit)

  write(output_unit, '(a,i0,a,i0,a,f6.2,a)') '[solid] done: ', window, ' windows, ', iters, &
       ' coupling iterations (', real(iters, dp) / max(window, 1), ' per window)'
  call precicef_finalize()
  call solver_finalize()

contains

  !> Vertices at the nx+1 grid nodes of interface s, connected by nx edges.
  subroutine define_mesh(s)
    integer, intent(in) :: s
    real(dp) :: coords(2 * nv)
    integer  :: edges(2 * nx)
    coords(1::2) = [(i * dxs, i = 0, nx)]
    coords(2::2) = y_iface(s)
    call precicef_set_mesh_vertices(trim(mesh(s)), nv, coords, ids(:, s), len_trim(mesh(s)))
    edges(1::2) = ids(1:nx, s)
    edges(2::2) = ids(2:nv, s)
    call precicef_set_mesh_edges(trim(mesh(s)), nx, edges, len_trim(mesh(s)))
  end subroutine define_mesh

  subroutine write_flux()
    integer :: m
    do m = HOT, COLD
      call precicef_write_data(trim(mesh(m)), d_flux, nv, ids(:, m), q(:, m), len_trim(mesh(m)), len(d_flux))
    end do
  end subroutine write_flux

  subroutine output()
    character(len=512) :: fname
    write(fname, '(a,"/solid_",i4.4,".vtk")') trim(outdir), nout
    call solver_write_vtk(trim(fname))
    nout = nout + 1
  end subroutine output

  !> Heat flow through an interface per unit depth [W/m] (trapezoidal rule).
  real(dp) function boundary_heat(qs)
    real(dp), intent(in) :: qs(0:)
    boundary_heat = (sum(qs) - 0.5_dp * (qs(0) + qs(nx))) * dxs
  end function boundary_heat

end program solid_main

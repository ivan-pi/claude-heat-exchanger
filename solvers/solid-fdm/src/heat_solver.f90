!> Transient 2-D heat conduction in the insulating wall (Code B).
!>
!>   rho c dT/dt = div(k grad T)   on [0, L] x [y0, y0 + d]
!>
!> Vertex-centred finite differences written in finite-volume form (half cells on the
!> boundaries) so that the system matrix is symmetric positive definite and the interface
!> heat flux can be recovered conservatively from the boundary half-cell balance.
!>
!>   j = 0   (y = y0)     : interface with the cold channel, Dirichlet (from preCICE)
!>   j = ny  (y = y0 + d) : interface with the hot channel,  Dirichlet (from preCICE)
!>   i = 0, nx            : adiabatic
!>
!> Time integration: backward Euler. Linear solver: Jacobi-preconditioned CG, matrix free.
!> All kernels run as OpenMP target regions; the fields stay resident on the device and
!> only the 1-D interface buffers are moved with "target update".
module heat_solver
  use, intrinsic :: iso_fortran_env, only: dp => real64
  implicit none
  private

  public :: solver_init, solver_finalize, solver_set_interface_temperature, solver_step, &
            solver_interface_flux, solver_save_checkpoint, solver_restore_checkpoint, &
            solver_write_vtk, solver_energy, solver_nx, solver_dx

  integer  :: nx = 0, ny = 0
  real(dp) :: dx = 0, dy = 0, kcond = 0, rhoc = 0, y0 = 0

  ! fields (0:nx, 0:ny); r, z, p, Ap have zero boundary rows by construction
  real(dp), allocatable :: T(:,:), Told(:,:), T_ckpt(:,:), r(:,:), z(:,:), p(:,:), Ap(:,:)
  ! interface buffers (0:nx)
  real(dp), allocatable, public :: Tb_hot(:), Tb_cold(:), q_hot(:), q_cold(:)

contains

  integer function solver_nx()
    solver_nx = nx
  end function solver_nx

  real(dp) function solver_dx()
    solver_dx = dx
  end function solver_dx

  ! ---------------------------------------------------------------------------------
  subroutine solver_init(nx_, ny_, length, thickness, y_bottom, k_, rhoc_, T_init)
    integer,  intent(in) :: nx_, ny_
    real(dp), intent(in) :: length, thickness, y_bottom, k_, rhoc_, T_init

    nx = nx_;  ny = ny_
    dx = length / nx;  dy = thickness / ny
    kcond = k_;  rhoc = rhoc_;  y0 = y_bottom

    allocate(T(0:nx, 0:ny), Told(0:nx, 0:ny), T_ckpt(0:nx, 0:ny))
    allocate(r(0:nx, 0:ny), z(0:nx, 0:ny), p(0:nx, 0:ny), Ap(0:nx, 0:ny))
    allocate(Tb_hot(0:nx), Tb_cold(0:nx), q_hot(0:nx), q_cold(0:nx))

    T = T_init;  Told = T_init;  T_ckpt = T_init
    r = 0;  z = 0;  p = 0;  Ap = 0
    Tb_hot = T_init;  Tb_cold = T_init;  q_hot = 0;  q_cold = 0

    !$omp target enter data map(to: T, Told, T_ckpt, r, z, p, Ap, Tb_hot, Tb_cold, q_hot, q_cold)
  end subroutine solver_init

  subroutine solver_finalize()
    !$omp target exit data map(delete: T, Told, T_ckpt, r, z, p, Ap, Tb_hot, Tb_cold, q_hot, q_cold)
    deallocate(T, Told, T_ckpt, r, z, p, Ap, Tb_hot, Tb_cold, q_hot, q_cold)
  end subroutine solver_finalize

  ! ---------------------------------------------------------------------------------
  !> Push the interface temperatures (host buffers filled by preCICE) to the device.
  subroutine solver_set_interface_temperature()
    !$omp target update to(Tb_hot, Tb_cold)
  end subroutine solver_set_interface_temperature

  ! ---------------------------------------------------------------------------------
  !> y = A x on interior rows 1..ny-1. The neighbours in rows 0 and ny are read from x:
  !> zero for CG directions, the Dirichlet values when x = T.
  subroutine apply_A(x, y, dt)
    real(dp), intent(in)    :: x(0:, 0:)
    real(dp), intent(inout) :: y(0:, 0:)
    real(dp), intent(in)    :: dt
    integer  :: i, j, nx_, ny_
    real(dp) :: wx, cap, ax, ay, s, dx_, dy_, k_, rc_

    nx_ = nx; ny_ = ny; dx_ = dx; dy_ = dy; k_ = kcond; rc_ = rhoc

    !$omp target teams loop collapse(2) private(wx, cap, ax, ay, s)
    do j = 1, ny_ - 1
      do i = 0, nx_
        wx = dx_
        if (i == 0 .or. i == nx_) wx = 0.5_dp * dx_
        cap = rc_ * wx * dy_ / dt
        ax  = k_ * dy_ / dx_
        ay  = k_ * wx / dy_
        s = (cap + 2.0_dp * ay) * x(i, j) - ay * (x(i, j - 1) + x(i, j + 1))
        if (i > 0)   s = s + ax * (x(i, j) - x(i - 1, j))
        if (i < nx_) s = s + ax * (x(i, j) - x(i + 1, j))
        y(i, j) = s
      end do
    end do
  end subroutine apply_A

  ! ---------------------------------------------------------------------------------
  !> One backward-Euler step of size dt, starting PCG from the old temperature field.
  !> Stopping test: ||r|| <= tol*||r_0|| (r_0 = flux imbalance that drives the step) or
  !> ||r|| <= 1e-14*||b|| (round-off floor, reached immediately at steady state).
  !> Note: a test relative to ||b|| alone would be meaningless here, since b ~ rho c T/dt
  !> is dominated by the stored energy and not by the heat flow.
  subroutine solver_step(dt, tol, maxit, iters, relres, converged)
    real(dp), intent(in)  :: dt, tol
    integer,  intent(in)  :: maxit
    integer,  intent(out) :: iters
    real(dp), intent(out) :: relres     !< ||r|| / ||r_0||
    logical,  intent(out) :: converged
    integer  :: i, j, it, nx_, ny_
    real(dp) :: wx, cap, diag, dx_, dy_, k_, rc_
    real(dp) :: rz, rz_new, pAp, alpha, beta, rr, bb, r0, stop_at

    nx_ = nx; ny_ = ny; dx_ = dx; dy_ = dy; k_ = kcond; rc_ = rhoc

    ! Told <- T, then impose the new Dirichlet data on both interfaces
    !$omp target teams loop collapse(2)
    do j = 0, ny_
      do i = 0, nx_
        Told(i, j) = T(i, j)
      end do
    end do
    !$omp target teams loop
    do i = 0, nx_
      T(i, 0)   = Tb_cold(i)
      T(i, ny_) = Tb_hot(i)
    end do

    ! r = b - A T with b = cap*Told (Dirichlet contributions enter through A T)
    call apply_A(T, Ap, dt)
    rz = 0;  bb = 0;  rr = 0
    !$omp target teams loop collapse(2) private(wx, cap, diag) reduction(+:rz, bb, rr)
    do j = 1, ny_ - 1
      do i = 0, nx_
        wx = dx_
        if (i == 0 .or. i == nx_) wx = 0.5_dp * dx_
        cap  = rc_ * wx * dy_ / dt
        diag = cap + 2.0_dp * k_ * wx / dy_
        if (i > 0)   diag = diag + k_ * dy_ / dx_
        if (i < nx_) diag = diag + k_ * dy_ / dx_
        r(i, j) = cap * Told(i, j) - Ap(i, j)
        z(i, j) = r(i, j) / diag
        p(i, j) = z(i, j)
        rz = rz + r(i, j) * z(i, j)
        rr = rr + r(i, j)**2
        bb = bb + (cap * Told(i, j))**2
      end do
    end do
    bb = sqrt(bb)
    r0 = sqrt(rr)
    stop_at = max(tol * r0, 1.0e-14_dp * bb)

    iters = 0
    converged = sqrt(rr) <= stop_at
    do it = 1, maxit
      if (converged) exit
      iters = it

      call apply_A(p, Ap, dt)
      pAp = 0
      !$omp target teams loop collapse(2) reduction(+:pAp)
      do j = 1, ny_ - 1
        do i = 0, nx_
          pAp = pAp + p(i, j) * Ap(i, j)
        end do
      end do
      alpha = rz / pAp

      rz_new = 0;  rr = 0
      !$omp target teams loop collapse(2) private(wx, cap, diag) reduction(+:rz_new, rr)
      do j = 1, ny_ - 1
        do i = 0, nx_
          wx = dx_
          if (i == 0 .or. i == nx_) wx = 0.5_dp * dx_
          cap  = rc_ * wx * dy_ / dt
          diag = cap + 2.0_dp * k_ * wx / dy_
          if (i > 0)   diag = diag + k_ * dy_ / dx_
          if (i < nx_) diag = diag + k_ * dy_ / dx_
          T(i, j) = T(i, j) + alpha * p(i, j)
          r(i, j) = r(i, j) - alpha * Ap(i, j)
          z(i, j) = r(i, j) / diag
          rz_new = rz_new + r(i, j) * z(i, j)
          rr = rr + r(i, j)**2
        end do
      end do
      converged = sqrt(rr) <= stop_at

      beta = rz_new / rz
      rz = rz_new
      !$omp target teams loop collapse(2)
      do j = 1, ny_ - 1
        do i = 0, nx_
          p(i, j) = z(i, j) + beta * p(i, j)
        end do
      end do
    end do
    relres = 0
    if (r0 > 0) relres = sqrt(rr) / r0
  end subroutine solver_step

  ! ---------------------------------------------------------------------------------
  !> Heat flux from the solid into the fluid [W/m^2] at both interfaces, recovered from
  !> the energy balance of the boundary half cells (consistent with the discrete scheme):
  !>   q = [ -rho c (w dy/2) dT/dt + k w/dy (T_inner - T_b) + k (dy/2)/dx d2T/dx2 ] / w
  !> Positive values mean heat flows out of the wall into the fluid.
  subroutine solver_interface_flux(dt)
    real(dp), intent(in) :: dt
    integer  :: i, nx_, ny_
    real(dp) :: wx, chalf, ay, axh, s, dx_, dy_, k_, rc_

    nx_ = nx; ny_ = ny; dx_ = dx; dy_ = dy; k_ = kcond; rc_ = rhoc

    !$omp target teams loop private(wx, chalf, ay, axh, s)
    do i = 0, nx_
      wx = dx_
      if (i == 0 .or. i == nx_) wx = 0.5_dp * dx_
      chalf = rc_ * wx * 0.5_dp * dy_ / dt
      ay    = k_ * wx / dy_
      axh   = k_ * 0.5_dp * dy_ / dx_

      ! hot interface, j = ny
      s = -chalf * (T(i, ny_) - Told(i, ny_)) + ay * (T(i, ny_ - 1) - T(i, ny_))
      if (i > 0)   s = s + axh * (T(i - 1, ny_) - T(i, ny_))
      if (i < nx_) s = s + axh * (T(i + 1, ny_) - T(i, ny_))
      q_hot(i) = s / wx

      ! cold interface, j = 0
      s = -chalf * (T(i, 0) - Told(i, 0)) + ay * (T(i, 1) - T(i, 0))
      if (i > 0)   s = s + axh * (T(i - 1, 0) - T(i, 0))
      if (i < nx_) s = s + axh * (T(i + 1, 0) - T(i, 0))
      q_cold(i) = s / wx
    end do

    !$omp target update from(q_hot, q_cold)
  end subroutine solver_interface_flux

  ! ---------------------------------------------------------------------------------
  !> Thermal energy stored in the wall per unit depth [J/m], same half-cell weights as
  !> the discretisation, so dE/dt + Q_hot + Q_cold vanishes up to the PCG tolerance.
  real(dp) function solver_energy() result(E)
    integer  :: i, j, nx_, ny_
    real(dp) :: w
    nx_ = nx; ny_ = ny
    E = 0
    !$omp target teams loop collapse(2) private(w) reduction(+:E)
    do j = 0, ny_
      do i = 0, nx_
        w = 1.0_dp
        if (i == 0 .or. i == nx_) w = 0.5_dp * w
        if (j == 0 .or. j == ny_) w = 0.5_dp * w
        E = E + w * T(i, j)
      end do
    end do
    E = E * rhoc * dx * dy
  end function solver_energy

  ! ---------------------------------------------------------------------------------
  subroutine solver_save_checkpoint()
    integer :: i, j, nx_, ny_
    nx_ = nx; ny_ = ny
    !$omp target teams loop collapse(2)
    do j = 0, ny_
      do i = 0, nx_
        T_ckpt(i, j) = T(i, j)
      end do
    end do
  end subroutine solver_save_checkpoint

  subroutine solver_restore_checkpoint()
    integer :: i, j, nx_, ny_
    nx_ = nx; ny_ = ny
    !$omp target teams loop collapse(2)
    do j = 0, ny_
      do i = 0, nx_
        T(i, j) = T_ckpt(i, j)
      end do
    end do
  end subroutine solver_restore_checkpoint

  ! ---------------------------------------------------------------------------------
  subroutine solver_write_vtk(filename)
    character(len=*), intent(in) :: filename
    integer :: unit, i, j

    !$omp target update from(T)
    open(newunit=unit, file=filename, status='replace', action='write')
    write(unit, '(a)') '# vtk DataFile Version 3.0'
    write(unit, '(a)') 'solid-fdm wall'
    write(unit, '(a)') 'ASCII'
    write(unit, '(a)') 'DATASET STRUCTURED_POINTS'
    write(unit, '(a,3(1x,i0))') 'DIMENSIONS', nx + 1, ny + 1, 1
    write(unit, '(a,3(1x,es17.9))') 'ORIGIN', 0.0_dp, y0, 0.0_dp
    write(unit, '(a,3(1x,es17.9))') 'SPACING', dx, dy, dx
    write(unit, '(a,1x,i0)') 'POINT_DATA', (nx + 1) * (ny + 1)
    write(unit, '(a)') 'SCALARS T double 1'
    write(unit, '(a)') 'LOOKUP_TABLE default'
    do j = 0, ny
      do i = 0, nx
        write(unit, '(es17.9)') T(i, j)
      end do
    end do
    close(unit)
  end subroutine solver_write_vtk

end module heat_solver

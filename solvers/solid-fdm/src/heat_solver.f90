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

  public :: solver_init, solver_finalize, solver_step, solver_interface_flux, &
            solver_save_checkpoint, solver_restore_checkpoint, solver_write_vtk, &
            solver_energy, solver_dx

  !> Interface index: hot side is the top row j = ny, cold side the bottom row j = 0.
  integer, parameter, public :: HOT = 1, COLD = 2

  ! grid and material constants, resident on the device after solver_init
  integer  :: nx = 0, ny = 0
  real(dp) :: dx = 0, dy = 0, kcond = 0, rhoc = 0
  !$omp declare target(nx, ny, dx, dy, kcond, rhoc)
  real(dp) :: y0 = 0

  ! fields (0:nx, 0:ny); r, z, p, Ap have zero boundary rows by construction
  real(dp), allocatable :: T(:,:), Told(:,:), T_ckpt(:,:), r(:,:), z(:,:), p(:,:), Ap(:,:)
  real(dp), allocatable :: wx(:)      ! column width: dx inside, dx/2 at the adiabatic ends (0:nx)
  real(dp), allocatable :: invdiag(:) ! Jacobi preconditioner 1/A_ii per column, for the current dt (0:nx)
  ! interface buffers (0:nx, HOT:COLD): Dirichlet temperature in, heat flux out
  real(dp), allocatable, public :: Tb(:,:), q(:,:)

contains

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
    !$omp target update to(nx, ny, dx, dy, kcond, rhoc)

    allocate(T(0:nx, 0:ny), Told(0:nx, 0:ny), T_ckpt(0:nx, 0:ny))
    allocate(r(0:nx, 0:ny), z(0:nx, 0:ny), p(0:nx, 0:ny), Ap(0:nx, 0:ny))
    allocate(wx(0:nx), invdiag(0:nx), Tb(0:nx, HOT:COLD), q(0:nx, HOT:COLD))

    T = T_init;  Told = T_init;  T_ckpt = T_init
    r = 0;  z = 0;  p = 0;  Ap = 0
    wx = dx;  wx(0) = 0.5_dp * dx;  wx(nx) = 0.5_dp * dx
    invdiag = 0
    Tb = T_init;  q = 0

    !$omp target enter data map(to: T, Told, T_ckpt, r, z, p, Ap, wx, invdiag, Tb, q)
  end subroutine solver_init

  subroutine solver_finalize()
    !$omp target exit data map(delete: T, Told, T_ckpt, r, z, p, Ap, wx, invdiag, Tb, q)
    deallocate(T, Told, T_ckpt, r, z, p, Ap, wx, invdiag, Tb, q)
  end subroutine solver_finalize

  ! ---------------------------------------------------------------------------------
  !> Device-side copy dst <- src of a full field.
  subroutine device_copy(src, dst)
    real(dp), intent(in),  contiguous :: src(0:, 0:)
    real(dp), intent(out), contiguous :: dst(0:, 0:)
    integer :: i, j
    !$omp target teams loop collapse(2)
    do j = 0, ny
      do i = 0, nx
        dst(i, j) = src(i, j)
      end do
    end do
  end subroutine device_copy

  ! ---------------------------------------------------------------------------------
  !> y = A x on the interior rows 1..ny-1, and xAy = <x, y> over them. The neighbours in
  !> rows 0 and ny are read from x: zero for CG directions, the Dirichlet values when x = T.
  subroutine apply_A(x, y, dt, xAy)
    real(dp), intent(in),    contiguous :: x(0:, 0:)
    real(dp), intent(inout), contiguous :: y(0:, 0:)
    real(dp), intent(in)  :: dt
    real(dp), intent(out) :: xAy
    integer  :: i, j
    real(dp) :: ax, cdt, kdy, cap, ay, s

    ax  = kcond * dy / dx   ! conductance of an x-link
    cdt = rhoc * dy / dt    ! capacity per unit column width
    kdy = kcond / dy        ! y-conductance per unit column width
    xAy = 0
    !$omp target teams loop collapse(2) private(cap, ay, s) reduction(+:xAy)
    do j = 1, ny - 1
      do i = 0, nx
        cap = cdt * wx(i)
        ay  = kdy * wx(i)
        s = (cap + 2.0_dp * ay) * x(i, j) - ay * (x(i, j - 1) + x(i, j + 1))
        if (i > 0)  s = s + ax * (x(i, j) - x(i - 1, j))
        if (i < nx) s = s + ax * (x(i, j) - x(i + 1, j))
        y(i, j) = s
        xAy = xAy + x(i, j) * s
      end do
    end do
  end subroutine apply_A

  ! ---------------------------------------------------------------------------------
  !> One backward-Euler step of size dt, starting PCG from the old temperature field.
  !> The Dirichlet data Tb (host buffers filled by preCICE) is pushed to the device here.
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
    integer  :: i, j
    real(dp) :: ax, cdt, cap
    real(dp) :: rz, rz_new, pAp, alpha, beta, rr, bb, r0, stop_at

    ax  = kcond * dy / dx
    cdt = rhoc * dy / dt
    ! Jacobi preconditioner for this dt: diagonal of A, which depends on the column only
    do i = 0, nx
      invdiag(i) = cdt * wx(i) + 2.0_dp * kcond * wx(i) / dy
      if (i > 0)  invdiag(i) = invdiag(i) + ax
      if (i < nx) invdiag(i) = invdiag(i) + ax
      invdiag(i) = 1.0_dp / invdiag(i)
    end do
    !$omp target update to(invdiag, Tb)

    ! Told <- T, then impose the new Dirichlet data on both interfaces
    !$omp target teams loop collapse(2)
    do j = 0, ny
      do i = 0, nx
        Told(i, j) = T(i, j)
        if (j == 0)  T(i, j) = Tb(i, COLD)
        if (j == ny) T(i, j) = Tb(i, HOT)
      end do
    end do

    ! r = b - A T with b = cap*Told (Dirichlet contributions enter through A T)
    call apply_A(T, Ap, dt, pAp)
    rz = 0;  bb = 0;  rr = 0
    !$omp target teams loop collapse(2) private(cap) reduction(+:rz, bb, rr)
    do j = 1, ny - 1
      do i = 0, nx
        cap = cdt * wx(i)
        r(i, j) = cap * Told(i, j) - Ap(i, j)
        z(i, j) = r(i, j) * invdiag(i)
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
    do while (.not. converged .and. iters < maxit)
      iters = iters + 1

      call apply_A(p, Ap, dt, pAp)
      alpha = rz / pAp

      rz_new = 0;  rr = 0
      !$omp target teams loop collapse(2) reduction(+:rz_new, rr)
      do j = 1, ny - 1
        do i = 0, nx
          T(i, j) = T(i, j) + alpha * p(i, j)
          r(i, j) = r(i, j) - alpha * Ap(i, j)
          z(i, j) = r(i, j) * invdiag(i)
          rz_new = rz_new + r(i, j) * z(i, j)
          rr = rr + r(i, j)**2
        end do
      end do
      converged = sqrt(rr) <= stop_at
      if (converged) exit

      beta = rz_new / rz
      rz = rz_new
      !$omp target teams loop collapse(2)
      do j = 1, ny - 1
        do i = 0, nx
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
  !> Positive values mean heat flows out of the wall into the fluid. The result is
  !> brought to the host (q is the preCICE write source).
  subroutine solver_interface_flux(dt)
    real(dp), intent(in) :: dt
    integer  :: i, s, jb, jin
    real(dp) :: chalf, kdy, axh, f

    chalf = 0.5_dp * rhoc * dy / dt   ! half-cell capacity per unit column width
    kdy   = kcond / dy
    axh   = 0.5_dp * kcond * dy / dx  ! conductance of a half-cell x-link
    !$omp target teams loop collapse(2) private(jb, jin, f)
    do s = HOT, COLD
      do i = 0, nx
        jb  = merge(ny, 0, s == HOT)      ! boundary row
        jin = merge(ny - 1, 1, s == HOT)  ! its inner neighbour
        f = -chalf * (T(i, jb) - Told(i, jb)) + kdy * (T(i, jin) - T(i, jb))
        if (i > 0)  f = f + axh / wx(i) * (T(i - 1, jb) - T(i, jb))
        if (i < nx) f = f + axh / wx(i) * (T(i + 1, jb) - T(i, jb))
        q(i, s) = f
      end do
    end do
    !$omp target update from(q)
  end subroutine solver_interface_flux

  ! ---------------------------------------------------------------------------------
  !> Thermal energy stored in the wall per unit depth [J/m], same half-cell weights as
  !> the discretisation, so dE/dt + Q_hot + Q_cold vanishes up to the PCG tolerance.
  real(dp) function solver_energy() result(E)
    integer  :: i, j
    real(dp) :: w
    E = 0
    !$omp target teams loop collapse(2) private(w) reduction(+:E)
    do j = 0, ny
      do i = 0, nx
        w = wx(i)
        if (j == 0 .or. j == ny) w = 0.5_dp * w
        E = E + w * T(i, j)
      end do
    end do
    E = E * rhoc * dy
  end function solver_energy

  ! ---------------------------------------------------------------------------------
  subroutine solver_save_checkpoint()
    call device_copy(T, T_ckpt)
  end subroutine solver_save_checkpoint

  subroutine solver_restore_checkpoint()
    call device_copy(T_ckpt, T)
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

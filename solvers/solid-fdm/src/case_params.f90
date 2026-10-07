!> Reader for the shared case file (case/params.txt): "key value" per line, '#' comments.
!> The file is read once with params_load(); later entries override earlier ones, as in
!> the C++ reader of the fluid participant. Keys without a default are mandatory.
module case_params
  use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
  implicit none
  private
  public :: params_load, param_real, param_int

  integer, parameter :: wlen = 128
  character(len=wlen), allocatable :: keys(:), vals(:)
  integer :: nkeys = 0

contains

  subroutine params_load(filename)
    character(len=*), intent(in) :: filename
    character(len=512) :: line
    character(len=wlen) :: k, v
    integer :: unit, ios, hash, i

    open(newunit=unit, file=filename, status='old', action='read', iostat=ios)
    if (ios /= 0) then
      write(error_unit, '(a)') 'case_params: cannot open '//trim(filename)
      error stop 1
    end if
    allocate(keys(64), vals(64))
    do
      read(unit, '(a)', iostat=ios) line
      if (ios /= 0) exit
      hash = index(line, '#')
      if (hash > 0) line(hash:) = ' '
      read(line, *, iostat=ios) k, v
      if (ios /= 0) cycle
      i = find(k)
      if (i == 0) then
        if (nkeys == size(keys)) then
          keys = [keys, keys]  ! grow
          vals = [vals, vals]
        end if
        nkeys = nkeys + 1
        i = nkeys
        keys(i) = k
      end if
      vals(i) = v
    end do
    close(unit)
  end subroutine params_load

  !> Index of key, 0 if absent.
  integer function find(key)
    character(len=*), intent(in) :: key
    do find = 1, nkeys
      if (keys(find) == key) return
    end do
    find = 0
  end function find

  !> Index of key; absent keys are fatal unless the caller has a default.
  integer function require(key, has_default)
    character(len=*), intent(in) :: key
    logical, intent(in) :: has_default
    require = find(key)
    if (require == 0 .and. .not. has_default) then
      write(error_unit, '(a)') 'case_params: missing parameter '//key
      error stop 1
    end if
  end function require

  real(dp) function param_real(key, default)
    character(len=*), intent(in) :: key
    real(dp), intent(in), optional :: default
    integer :: i
    i = require(key, present(default))
    if (i == 0) then
      param_real = default
    else
      read(vals(i), *) param_real
    end if
  end function param_real

  integer function param_int(key, default)
    character(len=*), intent(in) :: key
    integer, intent(in), optional :: default
    integer :: i
    i = require(key, present(default))
    if (i == 0) then
      param_int = default
    else
      read(vals(i), *) param_int
    end if
  end function param_int

end module case_params

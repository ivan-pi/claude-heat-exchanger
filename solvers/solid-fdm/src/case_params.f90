!> Reader for the shared case file (case/params.txt): "key value" per line, '#' comments.
module case_params
  use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
  implicit none
  private
  public :: param_real, param_int

contains

  function lookup(filename, key) result(value)
    character(len=*), intent(in) :: filename, key
    character(len=:), allocatable :: value
    character(len=512) :: line
    character(len=128) :: k, v
    integer :: unit, ios, hash

    open(newunit=unit, file=filename, status='old', action='read', iostat=ios)
    if (ios /= 0) then
      write(error_unit, '(a)') 'case_params: cannot open '//trim(filename)
      error stop 1
    end if
    do
      read(unit, '(a)', iostat=ios) line
      if (ios /= 0) exit
      hash = index(line, '#')
      if (hash > 0) line(hash:) = ' '
      read(line, *, iostat=ios) k, v
      if (ios /= 0) cycle
      if (trim(k) == key) then
        value = trim(v)
        close(unit)
        return
      end if
    end do
    close(unit)
    write(error_unit, '(a)') 'case_params: missing parameter '//key
    error stop 1
  end function lookup

  real(dp) function param_real(filename, key)
    character(len=*), intent(in) :: filename, key
    character(len=:), allocatable :: v
    v = lookup(filename, key)
    read(v, *) param_real
  end function param_real

  integer function param_int(filename, key)
    character(len=*), intent(in) :: filename, key
    character(len=:), allocatable :: v
    v = lookup(filename, key)
    read(v, *) param_int
  end function param_int

end module case_params

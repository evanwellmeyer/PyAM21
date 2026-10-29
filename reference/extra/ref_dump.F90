! Reference dump hook for the PyAM21 port.
!
! Off unless the environment variable REF_DUMP_DIR is set. When on, every
! ref_dump_begin(tag) ... ref_dump_end() block writes one stream file
!     $REF_DUMP_DIR/<tag>_s<step>_c<count>.bin
! where step is set by ref_dump_set_step (the atmospheric step number, 1-based)
! and count numbers repeated calls with the same tag within a step (1-based).
! Each array record: name (char*32), type (int4: 1 real8, 2 int4), ndim (int4),
! shape (ndim int4), data in Fortran order.  Logicals are written as int4 0/1.
! Optional REF_DUMP_STEPS="a,b,c" restricts the steps dumped.
module ref_dump_mod
  implicit none
  private
  public :: ref_dump_active, ref_dump_set_step, ref_dump_begin, ref_dump_end, ref_dump, &
            ref_dump_step

  logical, save :: initialized = .false.
  logical, save :: env_on = .false.
  logical, save :: in_block = .false.
  character(len=1024), save :: dumpdir = ''
  character(len=4096), save :: steplist = ''
  integer, save :: cur_step = 0
  integer, save :: iunit = -1
  integer, parameter :: maxtags = 256
  character(len=64), save :: tags(maxtags)
  integer, save :: tagcount(maxtags) = 0
  integer, save :: tagstep(maxtags) = -1
  integer, save :: ntags = 0

  interface ref_dump
    module procedure dump_r0, dump_r1, dump_r2, dump_r3, dump_r4, &
                     dump_i0, dump_i1, dump_i2, dump_i3, dump_l0, dump_l1, dump_l2, dump_l3
  end interface

contains

  subroutine init
    integer :: st, ln
    if (initialized) return
    initialized = .true.
    call get_environment_variable('REF_DUMP_DIR', dumpdir, ln, st)
    env_on = (st == 0 .and. ln > 0)
    call get_environment_variable('REF_DUMP_STEPS', steplist, ln, st)
    if (st /= 0) steplist = ''
  end subroutine init

  logical function step_selected()
    character(len=16) :: s
    if (len_trim(steplist) == 0) then
      step_selected = .true.
      return
    endif
    write (s, '(i0)') cur_step
    step_selected = index(','//trim(steplist)//',', ','//trim(s)//',') > 0
  end function step_selected

  logical function ref_dump_active()
    call init
    ref_dump_active = env_on .and. step_selected()
  end function ref_dump_active

  subroutine ref_dump_set_step(n)
    integer, intent(in) :: n
    cur_step = n
  end subroutine ref_dump_set_step

  integer function ref_dump_step()
    ref_dump_step = cur_step
  end function ref_dump_step

  subroutine ref_dump_begin(tag)
    character(len=*), intent(in) :: tag
    character(len=1300) :: fname
    integer :: i, it
    in_block = .false.
    if (.not. ref_dump_active()) return
    it = 0
    do i = 1, ntags
      if (trim(tags(i)) == trim(tag)) then
        it = i
        exit
      endif
    enddo
    if (it == 0) then
      ntags = ntags + 1
      it = ntags
      tags(it) = tag
    endif
    if (tagstep(it) /= cur_step) then
      tagstep(it) = cur_step
      tagcount(it) = 0
    endif
    tagcount(it) = tagcount(it) + 1
    write (fname, '(a,"/",a,"_s",i0,"_c",i0,".bin")') trim(dumpdir), trim(tag), cur_step, tagcount(it)
    open (newunit=iunit, file=trim(fname), access='stream', form='unformatted', status='replace')
    in_block = .true.
  end subroutine ref_dump_begin

  subroutine ref_dump_end()
    if (.not. in_block) return
    close (iunit)
    in_block = .false.
  end subroutine ref_dump_end

  subroutine header(name, typ, shp)
    character(len=*), intent(in) :: name
    integer, intent(in) :: typ
    integer, intent(in) :: shp(:)
    character(len=32) :: nm
    nm = name
    write (iunit) nm, int(typ, 4), int(size(shp), 4), int(shp, 4)
  end subroutine header

  subroutine dump_r0(name, a)
    character(len=*), intent(in) :: name
    real(8), intent(in) :: a
    integer :: shp(0)
    if (.not. in_block) return
    call header(name, 1, shp)
    write (iunit) a
  end subroutine dump_r0

  subroutine dump_r1(name, a)
    character(len=*), intent(in) :: name
    real(8), intent(in) :: a(:)
    if (.not. in_block) return
    call header(name, 1, shape(a))
    write (iunit) a
  end subroutine dump_r1

  subroutine dump_r2(name, a)
    character(len=*), intent(in) :: name
    real(8), intent(in) :: a(:,:)
    if (.not. in_block) return
    call header(name, 1, shape(a))
    write (iunit) a
  end subroutine dump_r2

  subroutine dump_r3(name, a)
    character(len=*), intent(in) :: name
    real(8), intent(in) :: a(:,:,:)
    if (.not. in_block) return
    call header(name, 1, shape(a))
    write (iunit) a
  end subroutine dump_r3

  subroutine dump_r4(name, a)
    character(len=*), intent(in) :: name
    real(8), intent(in) :: a(:,:,:,:)
    if (.not. in_block) return
    call header(name, 1, shape(a))
    write (iunit) a
  end subroutine dump_r4

  subroutine dump_i0(name, a)
    character(len=*), intent(in) :: name
    integer, intent(in) :: a
    integer :: shp(0)
    if (.not. in_block) return
    call header(name, 2, shp)
    write (iunit) int(a, 4)
  end subroutine dump_i0

  subroutine dump_i1(name, a)
    character(len=*), intent(in) :: name
    integer, intent(in) :: a(:)
    if (.not. in_block) return
    call header(name, 2, shape(a))
    write (iunit) int(a, 4)
  end subroutine dump_i1

  subroutine dump_i2(name, a)
    character(len=*), intent(in) :: name
    integer, intent(in) :: a(:,:)
    if (.not. in_block) return
    call header(name, 2, shape(a))
    write (iunit) int(a, 4)
  end subroutine dump_i2

  subroutine dump_i3(name, a)
    character(len=*), intent(in) :: name
    integer, intent(in) :: a(:,:,:)
    if (.not. in_block) return
    call header(name, 2, shape(a))
    write (iunit) int(a, 4)
  end subroutine dump_i3

  subroutine dump_l0(name, a)
    character(len=*), intent(in) :: name
    logical, intent(in) :: a
    integer :: shp(0)
    if (.not. in_block) return
    call header(name, 2, shp)
    write (iunit) merge(1_4, 0_4, a)
  end subroutine dump_l0

  subroutine dump_l1(name, a)
    character(len=*), intent(in) :: name
    logical, intent(in) :: a(:)
    if (.not. in_block) return
    call header(name, 2, shape(a))
    write (iunit) merge(1_4, 0_4, a)
  end subroutine dump_l1

  subroutine dump_l2(name, a)
    character(len=*), intent(in) :: name
    logical, intent(in) :: a(:,:)
    if (.not. in_block) return
    call header(name, 2, shape(a))
    write (iunit) merge(1_4, 0_4, a)
  end subroutine dump_l2

  subroutine dump_l3(name, a)
    character(len=*), intent(in) :: name
    logical, intent(in) :: a(:,:,:)
    if (.not. in_block) return
    call header(name, 2, shape(a))
    write (iunit) merge(1_4, 0_4, a)
  end subroutine dump_l3

end module ref_dump_mod

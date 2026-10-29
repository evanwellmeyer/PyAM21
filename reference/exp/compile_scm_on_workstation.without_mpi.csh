#!/bin/csh -f
unalias *
set echo
#--------------------------------------------------------------------------------------------------------
set platform  = gfdl_ws_64.intel.without_mpi            # A unique identifier for your platform
set template  = $cwd/../bin/mkmf.template.$platform     # path to template for your platform
set mkmf      = $cwd/../bin/mkmf                        # path to executable mkmf
set sourcedir = $cwd/../src                             # path to directory containing model source code
#--------------------------------------------------------------------------------------------------------
set execdir = $cwd/exec.$platform  # where code is compiled and executable is created
set executable = $execdir/scm.x

source $MODULESHOME/init/csh
module use -a /home/fms/local/modulefiles
module use /home/Seth.Underwood/publicmodules
module purge
module rm netcdf hdf5
module load ifort/11.1.073
module load icc/11.1.073
module load hdf5/1.8.6
module load netcdf/4.1.2
module load mpich2/1.5b1
module list

set pathnames = $cwd/path_names  # path to file containing list of source paths
#--------------------------------------------------------------------------------------------------------
# setup directory structure
if ( ! -d $execdir ) mkdir -p $execdir
cd $execdir
#--------------------------------------------------------------------------------------------------------

# execute mkmf to create makefile
set cppDefs = "-Duse_netCDF -Duse_LARGEFILE -DSPMD -DLAND_BND_TRACERS -DSCM -DINTERNAL_FILE_NML"
$mkmf -a $sourcedir -t $template -p $executable:t -c "$cppDefs" $pathnames $sourcedir/shared/include $sourcedir/shared/mpp/include /usr/local/include
if ( $status != 0 ) then
   unset echo
   echo "ERROR: mkmf failed for scm" 
   exit 1
endif

# --- execute make ---
make $executable:t
if ( $status != 0 ) then
   unset echo
   echo "ERROR: make failed for scm" 
   exit 1
endif

unset echo
echo "NOTE: make successful for scm"

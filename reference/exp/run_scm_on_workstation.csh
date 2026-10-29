#!/bin/csh -f
#Minimal runscript
set echo 
#--------------------------------------------------------------------------------------------------------
set experiment = astex
# The available experiments are:
# arm
# astex
# gcss_atex
# gcss_astex
# gcss_arm
# gcss_bomex
# gcss_dycoms2_rf01_a
# gcss_dycoms2_rf01_b
# gcss_dycoms2_rf02
# gcss_rico_short
# mpace_a
# mpace_b
#--------------------------------------------------------------------------------------------------------
# define variables
set platform  = gfdl_ws_64.intel                  # A unique identifier for your platform
set npes = 1                                      # Number of processors
set num_executions = 1                            # Number of times the model is run. Each run restarts from previous run.
set time_stamp = $cwd/../bin/time_stamp.csh       # Path to timestamp.csh
set model_executable = $cwd/exec.$platform/scm.x  # Path to model executable
set workdir = /local2/home/work/SCM/$experiment   # Where model is run and model output is produced
set inputdir = $cwd/../input/$experiment
#--------------------------------------------------------------------------------------------------------
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
limit stacksize unlimited

set namelist   = $inputdir/input.nml    # path to namelist file (contains all namelists)
set diagtable  = $inputdir/diag_table   # path to diagnositics table (specifies fields and files for diagnostic output)
set fieldtable = $inputdir/field_table  # path to field table (specifies tracers)
set datatable  = $inputdir/data_table   # path to data table (specifies data files used by data_override)
#--------------------------------------------------------------------------------------------------------
# Input files used by all experiments
set inputfiles = ( \
  /archive/fms/module_data/khartoum/albedo.data.nc \
  /archive/cm3/ipcc_ar5/input/common/navy_topography.data.nc \
  /archive/fms/module_data/khartoum/navy_pctwater.data.nc \
  /archive/fms/module_data/ipcc_ar4/lad_data.cpio \
  /archive/fms/module_data/perth/sea_esf.cpio )

# mpace_a and mpace_b use hurrell_sst.cpio, all others use sst_data.cpio
 if ( $experiment == mpace_a || $experiment == mpace_b ) then
   set inputfiles = ( $inputfiles /archive/fms/module_data/perth/hurrell_sst.cpio )
 else
   set inputfiles = ( $inputfiles /archive/fms/module_data/khartoum/sst_data.cpio )
 endif
#------------------------------------------------------------------------------------------------------------------
# The input data above also exists as a zipped tarball here: ftp://ftp.gfdl.noaa.gov/perm/pjp/SCM/input_data.tar.gz
#------------------------------------------------------------------------------------------------------------------

# Add all files in $inputdir 
 set inputfiles = ( $inputfiles $inputdir/* )

# setup directory structure
if ( -d $workdir ) then
  /bin/rm -rf $workdir/*
else
  mkdir -p $workdir
endif
cd $workdir
mkdir INPUT RESTART
#--------------------------------------------------------------------------------------------------------
# get input data and executable
cp $namelist   input.nml
cp $diagtable  diag_table
cp $fieldtable field_table
cp $datatable data_table
cp $model_executable .

cd $workdir/INPUT
foreach file ( $inputfiles )
  if ( $file:e == 'cpio' ) then
    cpio -iv < $file
  else if ( $file:e == 'tar' ) then
    tar -xvf $file
  else
    cp $file .
  endif
end 
cd $workdir

set irun = 1
while ( $irun <= $num_executions )
#--------------------------------------------------------------------------------------------------------
# run the model
/net2/nnz/opt/mpich2-1.3_ifort11_x64/bin/mpiexec.hydra -np $npes ./$model_executable:t # (workstation)
if ($status != 0) then
  echo "Error in execution of $cwd/$model_executable:t"
  exit 1
endif
#--------------------------------------------------------------------------------------------------------
# Prepare to run the model again
cd $workdir
/bin/rm INPUT/*.res   INPUT/*.res.nc   INPUT/*.res.nc.0???   INPUT/*.res.tile?.nc   INPUT/*.res.tile?.nc.0???
mv    RESTART/*.res RESTART/*.res.nc RESTART/*.res.nc.0??? RESTART/*.res.tile?.nc RESTART/*.res.tile?.nc.0??? INPUT
#--------------------------------------------------------------------------------------------------------
@ irun ++
end
exit 0

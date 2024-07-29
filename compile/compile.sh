#!/bin/bash
#PBS -P zg98
#PBS -q normal
#PBS -l walltime=08:00:00
#PBS -l ncpus=1
#PBS -l mem=16gb
#PBS -N build
#PBS -l wd
#PBS -l jobfs=4gb

# source $PBS_O_WORKDIR/modules.sh
source $HOME/scripts/load-configs-zg98.sh
source $HOME/scripts/load-intel.sh

julia --project=$PBS_O_WORKDIR $PBS_O_WORKDIR/compile/compile.jl

#!/bin/bash
#PBS -P zg98
#PBS -q normal
#PBS -l walltime=48:00:00
#PBS -l ncpus=144
#PBS -l mem=192gb
#PBS -N Upwind_instab
#PBS -l wd

source $HOME/scripts/load-configs-zg98.sh
source $HOME/scripts/load-intel.sh

mpiexec -n 144 julia --project=$PBS_O_WORKDIR -e'
  using PoissonIntegrator;
  main_tsw_entropy(;nprocs=(12,12),testcase=instability,ps=[1],ns=[192],CFLs=[0.2],tF=[100.0],
                    const_jac=false,
                    out_loc="tsw_entropy_instability_upwinding_n192",out_freq=20,
                    options=options_cg_gmres,
                    upwinding=true,
                    nls_tols = (;atol=1e-13,rtol=1e-11,maxiter=10))
'
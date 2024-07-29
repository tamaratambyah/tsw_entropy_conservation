#!/bin/bash
#PBS -P zg98
#PBS -q normal
#PBS -l walltime=48:00:00
#PBS -l ncpus=192
#PBS -l mem=192gb
#PBS -N merging_btilde_n192
#PBS -l wd

source $HOME/scripts/load-configs-zg98.sh
source $HOME/scripts/load-intel.sh

mpiexec -n 192 julia --project=$PBS_O_WORKDIR -e'
  using PoissonIntegrator;
  main_tsw_entropy(;nprocs=(16,12),testcase=merging,ps=[1],ns=[192],CFLs=[0.4],tF=[10.0],
                    const_jac=true,
                    out_loc="tsw_entropy_merging_btilde_n192",out_freq=20,
                    options=options_cg_gmres,
                    upwinding=nothing,conserving=nothing,
                    nls_tols = (;atol=1e-12,rtol=1e-11,maxiter=20))
' 

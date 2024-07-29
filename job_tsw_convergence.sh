#!/bin/bash
#PBS -P zg98
#PBS -q normal
#PBS -l walltime=48:00:00
#PBS -l ncpus=36
#PBS -l mem=64gb
#PBS -N n16p1_convergence
#PBS -l wd

source $HOME/scripts/load-configs-zg98.sh
source $HOME/scripts/load-intel.sh

mpiexec -n 36 julia --project=$PBS_O_WORKDIR -e'
  using PoissonIntegrator;
  main_tsw_entropy(;nprocs=(6,6),testcase=convergence,ps=[1],ns=[16],CFLs=[0.2],tF=[26.56],
                    const_jac=true,
                    out_loc="tsw_entropy_convergence_p1_n16",
                    options=options_cg_gmres,
                    nls_tols = (;atol=1e-12,rtol=1e-11,maxiter=50))
'
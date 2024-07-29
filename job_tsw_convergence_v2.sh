#!/bin/bash
#PBS -P zg98
#PBS -q normal
#PBS -l walltime=48:00:00
#PBS -l ncpus=96
#PBS -l mem=128gb
#PBS -N n128p2_convergence
#PBS -l wd

source $HOME/scripts/load-configs-zg98.sh
source $HOME/scripts/load-intel.sh

mpiexec -n 96 julia --project=$PBS_O_WORKDIR -e'
  using PoissonIntegrator;
  main_tsw_entropy(;nprocs=(12,8),testcase=convergencerestarted,ps=[2],ns=[128],CFLs=[0.2],tF=[26.56],
                    const_jac=true,
                    out_loc="tsw_entropy_convergence_p2_n128",out_freq=50,
                    options=options_cg_gmres,
                    nls_tols = (;atol=1e-12,rtol=1e-11,maxiter=50))
'
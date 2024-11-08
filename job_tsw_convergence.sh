#!/bin/bash
#PBS -P zg98
#PBS -q normal
#PBS -l walltime=1:00:00
#PBS -l ncpus=48
#PBS -l mem=192gb
#PBS -N n16p1_convergence
#PBS -l wd

source $PBS_O_WORKDIR/modules.sh 

mpiexec -n 48 julia --project=$PBS_O_WORKDIR -e'
  using PoissonIntegrator;
  main_tsw_entropy(;nprocs=(6,8),testcase=convergence,ps=[1],ns=[16,32],CFLs=[0.2],tF=[0.1],
                    const_jac=true,
                    out_loc="tsw_entropy_convergence",
                    options=options_cg_gmres,
                    nls_tols = (;atol=1e-12,rtol=1e-11,maxiter=50))
'
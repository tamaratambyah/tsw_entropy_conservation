#!/bin/bash
#PBS -P zg98
#PBS -q normal
#PBS -l walltime=1:00:00
#PBS -l ncpus=48
#PBS -l mem=192gb
#PBS -N upwinding
#PBS -l wd

source $PBS_O_WORKDIR/modules.sh 

mpiexec -n 48 julia --project=$PBS_O_WORKDIR -e'
  using PoissonIntegrator;
  main_tsw_entropy(;nprocs=(6,8),testcase=instability,ps=[1],ns=[64],CFLs=[0.2],tF=[100.0],
                    const_jac=false,
                    out_loc="tsw_entropy_instability_upwinding_n64",out_freq=20,
                    options=options_cg_gmres,
                    upwinding_parms = (;upwinding=true,const_jac=false,ε=1e-4,soft=false),
                    nls_tols = (;atol=1e-13,rtol=1e-11,maxiter=200))
'

# tsw_entropy_conservation

Repo containing zenodo code for publication regarding entropy conservation for thermal shallow water equations

Author: Tamara A. Tambyah, November 2024, Monash University, Melbourne, Australia


### Set up
Clone the repo:
```
git clone git@github.com:tamaratambyah/tsw_entropy_conservation.git
```

Navigate inside the repo folder
```
cd tsw_entropy_conservation
```

Run with mpi:
```
mpiexec -n 4 julia --project=. -e'using PoissonIntegrator; main_tsw_entropy(nprocs=(2,2))'
```
* change the inputs to ``main_tsw_entropy`` as required (see job_*.sh examples)
* check pointing not included 

The job_*.sh scripts are compatible with Gadi@NCI Australian supercomputer
* change PROJECT in modules.sh to configure Julia
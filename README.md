# Paper1_zenodo
Repo containing zenodo code for publication regarding entropy conservation for thermal shallow water equations


### Set up
Clone the repo:
```
git clone git@github.com:tamaratambyah/Paper1_zenodo.git
```

Navigate to the repo folder and make the data directory:
```
cd Paper1_zenodo
mkdir data
```

Run with mpi:
```
mpiexec -n 4 julia --project=. -e'using PoissonIntegrator; main_tsw_entropy(nprocs=(2,2))'
```
* change the inputs to ``main_tsw_entropy`` as required (see job_*.sh examples)
* there are sometimes issues with the initial save/load of meta data. If this occurs, run again. 
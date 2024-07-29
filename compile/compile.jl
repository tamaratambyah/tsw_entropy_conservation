using PackageCompiler

create_sysimage([:PoissonIntegrator],
  sysimage_path=joinpath(@__DIR__,"..","PoissonIntegrator.so"),
  precompile_execution_file=joinpath(@__DIR__,"warmup.jl"))

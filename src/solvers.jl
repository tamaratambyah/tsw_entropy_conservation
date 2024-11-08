function get_solvers(nls_tols)
  # ode solvers
  P = JacobiLinearSolver()
  ls = CGSolver(P;atol=1e-16,rtol=1.e-16,verbose=false)#
  ls.log.depth = 2
  gmres = GMRESSolver(10,Pr=P,atol=1e-14,rtol=1.e-14,maxiter=250,verbose=false)
  gmres.log.depth = 1
  nls = GridapSolvers.NonlinearSolvers.NewtonSolver(gmres;maxiter=nls_tols.maxiter,atol=nls_tols.atol,rtol=nls_tols.rtol,verbose=true)

  ode_solvers = (;nls=nls,ls=ls)


  # initial condition solvers
  P = JacobiLinearSolver()
  _cg = CGSolver(P;atol=1e-20,rtol=1.e-50,verbose=1) #CGSolver(P;atol=1e-16,rtol=1.e-16,verbose=1)
  _gmres = GMRESSolver(10,atol=1e-20,rtol=1.e-20,verbose=1,maxiter=500) # FGMRESSolver(10,P,rtol=1.e-14,verbose=1) #GMRESSolver(40;Pr=P,Pl=P,rtol=1.e-14,verbose=true)
  _nls_gmres = GridapSolvers.NonlinearSolvers.NewtonSolver(_gmres;maxiter=500,atol=1e-16,rtol=1.e-20,verbose=true)
  _nls_cg = GridapSolvers.NonlinearSolvers.NewtonSolver(_cg;maxiter=500,atol=1e-16,rtol=1.e-16,verbose=true)

  IC_solvers = (;_nls_cg=_nls_cg,_nls_gmres=_nls_gmres,_cg=_cg)

  solvers = (;ode_solvers=ode_solvers,IC_solvers=IC_solvers)
  return solvers
end


function get_solvers(nls_tols,ranks)
  # ode solvers
  ls = PETScLinearSolver(petsc_ls_from_options_c)
  nls_ls = PETScLinearSolver(petsc_ls_from_options_g)
  nls = GridapSolvers.NonlinearSolvers.NewtonSolver(nls_ls;maxiter=nls_tols.maxiter,atol=nls_tols.atol,rtol=nls_tols.rtol,verbose=i_am_main(ranks))

  ode_solvers = (;nls=nls,ls=ls)


  # initial condition solvers
  _cg = PETScLinearSolver(petsc_ls_from_options_c)
  _gmres = PETScLinearSolver(petsc_gmres_jacobi)
  _nls_gmres = GridapSolvers.NonlinearSolvers.NewtonSolver(_gmres;maxiter=nls_tols.maxiter,atol=1e-14,rtol=1.e-14,verbose=i_am_main(ranks))
  _nls_cg = GridapSolvers.NonlinearSolvers.NewtonSolver(_cg;maxiter=nls_tols.maxiter,atol=1e-14,rtol=1.e-14,verbose=i_am_main(ranks))

  IC_solvers = (;_nls_cg=_nls_cg,_nls_gmres=_nls_gmres,_cg=_cg)

  solvers = (;ode_solvers=ode_solvers,IC_solvers=IC_solvers)
  return solvers
end


# linear solver 1 - from options: prefix g
function petsc_ls_from_options_g(ksp)

  @check_error_code GridapPETSc.PETSC.KSPSetOptionsPrefix(ksp[],"g_")
  @check_error_code GridapPETSc.PETSC.KSPSetFromOptions(ksp[])

end

# linear solver 2 - from options: prefix c
function petsc_ls_from_options_c(ksp)

  @check_error_code GridapPETSc.PETSC.KSPSetOptionsPrefix(ksp[],"c_")
  @check_error_code GridapPETSc.PETSC.KSPSetFromOptions(ksp[])

end


# linear solver - gmres, precondiioned with jacobi
function petsc_gmres_jacobi(ksp)
  rtol = PetscScalar(1.e-14)
  atol = GridapPETSc.PETSC.PETSC_DEFAULT
  dtol = GridapPETSc.PETSC.PETSC_DEFAULT
  maxits = GridapPETSc.PETSC.PETSC_DEFAULT

  # GMRES solver

  @check_error_code GridapPETSc.PETSC.KSPSetOptionsPrefix(ksp[],"gj_")
  @check_error_code GridapPETSc.PETSC.KSPSetFromOptions(ksp[])
  @check_error_code GridapPETSc.PETSC.KSPSetTolerances(ksp[], rtol, atol, dtol, maxits)

  pc       = Ref{GridapPETSc.PETSC.PC}()
  @check_error_code GridapPETSc.PETSC.KSPGetPC(ksp[],pc)
  @check_error_code GridapPETSc.PETSC.PCSetType(pc[],GridapPETSc.PETSC.PCJACOBI)

end


function Gridap.FESpaces.solve!(u,solver::LinearFESolver,feop::FEOperator,cache::Nothing)
  println("here")
  x = get_free_dof_values(u)
  op = Gridap.FESpaces.get_algebraic_operator(feop)
  cache = solve!(x,solver.ls,op)
  trial = Gridap.FESpaces.get_trial(feop)
  u_new = FEFunction(trial,x)
  (u_new, cache)
end

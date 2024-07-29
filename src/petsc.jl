import GridapPETSc: PETScNonlinearSolver
import GridapPETSc: PETScNonlinearSolverCache
import Gridap.Algebra: NonlinearOperator
import GridapPETSc: PETSC

function Gridap.Algebra.solve!(x::T,
  nls::PETScNonlinearSolver,
  op::NonlinearOperator,
  cache::PETScNonlinearSolverCache{<:T}) where T <: AbstractVector

  # println("my petsc solve")
  # @assert false #cache.op === op ### This is not relevant for Transient problems

  #if (cache.comm != MPI.COMM_SELF)
  #  GridapPETSc.gridap_petsc_gc() # Do garbage collection of PETSc objects
  #end

  copy!(cache.x_sys_layout,x)
  x_petsc = convert(GridapPETSc.PETScVector,cache.x_sys_layout)

  snes_ref = Ref{PETSC.SNES}()
  @check_error_code PETSC.SNESCreate(cache.comm,snes_ref)

  cache = GridapPETSc.PETScNonlinearSolverCache(cache.comm, snes_ref, op, x, cache.x_sys_layout, cache.res_sys_layout,
                           cache.jac_mat_A, cache.jac_mat_P,
                           x_petsc, cache.res_petsc,
                           cache.jac_petsc_mat_A, cache.jac_petsc_mat_P)

  # set petsc residual function
  ctx  = pointer_from_objref(cache)
  fptr = @cfunction(GridapPETSc.snes_residual, PetscInt, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}))
  PETSC.SNESSetFunction(cache.snes[],cache.res_petsc.vec[],fptr,ctx)

  # set petsc jacobian function
  fptr = @cfunction(GridapPETSc.snes_jacobian, PetscInt, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid},Ptr{Cvoid}))
  PETSC.SNESSetJacobian(cache.snes[],cache.jac_petsc_mat_A.mat[],cache.jac_petsc_mat_A.mat[],fptr,ctx)

  nls.setup(cache.snes)

  @check_error_code PETSC.SNESSolve(cache.snes[],C_NULL,cache.x_petsc.vec[])
  copy!(x,cache.x_petsc)
  cache
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

# non-linear solver - with mumps
function petsc_newton_mumps_setup(snes)
  ksp      = Ref{GridapPETSc.PETSC.KSP}()
  pc       = Ref{GridapPETSc.PETSC.PC}()

  rtol = PetscScalar(1.e-14)
  atol = GridapPETSc.PETSC.PETSC_DEFAULT
  dtol = GridapPETSc.PETSC.PETSC_DEFAULT
  maxits = GridapPETSc.PETSC.PETSC_DEFAULT


  @check_error_code GridapPETSc.PETSC.SNESSetFromOptions(snes[])
  @check_error_code GridapPETSc.PETSC.SNESGetKSP(snes[],ksp)
  #@check_error_code GridapPETSc.PETSC.KSPSetType(ksp[],GridapPETSc.PETSC.KSPGMRES)
  @check_error_code GridapPETSc.PETSC.KSPSetType(ksp[],GridapPETSc.PETSC.KSPPREONLY)
  @check_error_code GridapPETSc.PETSC.KSPGetPC(ksp[],pc)
  @check_error_code GridapPETSc.PETSC.PCSetType(pc[],GridapPETSc.PETSC.PCLU)
  @check_error_code GridapPETSc.PETSC.PCFactorSetMatSolverType(pc[],GridapPETSc.PETSC.MATSOLVERMUMPS)
  @check_error_code GridapPETSc.PETSC.PCFactorSetUpMatSolverType(pc[])
  mumpsmat = Ref{GridapPETSc.PETSC.Mat}()
  @check_error_code GridapPETSc.PETSC.PCFactorGetMatrix(pc[],mumpsmat)
  @check_error_code GridapPETSc.PETSC.MatMumpsSetIcntl(mumpsmat[],  4, 1)
  @check_error_code GridapPETSc.PETSC.MatMumpsSetIcntl(mumpsmat[], 28, 2)
  @check_error_code GridapPETSc.PETSC.MatMumpsSetIcntl(mumpsmat[], 29, 2)

  # @check_error_code GridapPETSc.PETSC.S`NESSetType`(snes[],GridapPETSc.PETSC.SNESNEWTONLS)
  # @check_error_code GridapPETSc.PETSC.SNESGetKSP(snes[],ksp)

  # @check_error_code GridapPETSc.PETSC.KSPSetType(ksp[],GridapPETSc.PETSC.KSPGMRES)
  @check_error_code GridapPETSc.PETSC.KSPSetTolerances(ksp[], rtol, atol, dtol, maxits)

  # @check_error_code GridapPETSc.PETSC.KSPView(ksp[],C_NULL)



end


# non-linear solver - with gmres, preconditioned with jacobi
function petsc_newton_gmres_setup(snes)
  ksp      = Ref{GridapPETSc.PETSC.KSP}()
  pc       = Ref{GridapPETSc.PETSC.PC}()

  rtol = PetscScalar(1.e-14)
  atol = GridapPETSc.PETSC.PETSC_DEFAULT
  dtol = GridapPETSc.PETSC.PETSC_DEFAULT
  maxits = GridapPETSc.PETSC.PETSC_DEFAULT


  @check_error_code GridapPETSc.PETSC.SNESSetFromOptions(snes[])
  @check_error_code GridapPETSc.PETSC.SNESGetKSP(snes[],ksp)
  @check_error_code GridapPETSc.PETSC.KSPSetType(ksp[],GridapPETSc.PETSC.KSPGMRES)
  @check_error_code GridapPETSc.PETSC.KSPGetPC(ksp[],pc)
  @check_error_code GridapPETSc.PETSC.PCSetType(pc[],GridapPETSc.PETSC.PCJACOBI)
  @check_error_code GridapPETSc.PETSC.KSPSetTolerances(ksp[], rtol, atol, dtol, maxits)

  # @check_error_code GridapPETSc.PETSC.KSPView(ksp[],C_NULL)



end

# linear solver - mumps
function petsc_mumps_setup(ksp)
  pc       = Ref{GridapPETSc.PETSC.PC}()
  mumpsmat = Ref{GridapPETSc.PETSC.Mat}()
  @check_error_code GridapPETSc.PETSC.KSPSetType(ksp[],GridapPETSc.PETSC.KSPPREONLY)
  @check_error_code GridapPETSc.PETSC.KSPGetPC(ksp[],pc)
  @check_error_code GridapPETSc.PETSC.PCSetType(pc[],GridapPETSc.PETSC.PCLU)
  @check_error_code GridapPETSc.PETSC.PCFactorSetMatSolverType(pc[],GridapPETSc.PETSC.MATSOLVERMUMPS)
  @check_error_code GridapPETSc.PETSC.PCFactorSetUpMatSolverType(pc[])
  @check_error_code GridapPETSc.PETSC.PCFactorGetMatrix(pc[],mumpsmat)
  @check_error_code GridapPETSc.PETSC.MatMumpsSetIcntl(mumpsmat[],  4, 1)
  #@check_error_code GridapPETSc.PETSC.MatMumpsSetIcntl(mumpsmat[],  7, 0)
  @check_error_code GridapPETSc.PETSC.MatMumpsSetIcntl(mumpsmat[], 28, 2)
  @check_error_code GridapPETSc.PETSC.MatMumpsSetIcntl(mumpsmat[], 29, 2)
  #@check_error_code GridapPETSc.PETSC.MatMumpsSetCntl(mumpsmat[], 3, 1.0e-6)
  #@check_error_code GridapPETSc.PETSC.KSPView(ksp[],C_NULL)
end

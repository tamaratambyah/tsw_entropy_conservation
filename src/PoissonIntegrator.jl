module PoissonIntegrator

using DrWatson
using Gridap, GridapDistributed, GridapPETSc
using PartitionedArrays
using LinearAlgebra, NLsolve
using Gridap.Visualization
using GridapSolvers
using JLD2

import Gridap.Visualization: Grid

import Gridap.ODEs: Assembler
import Gridap.ODEs: GenericTransientFESolution, TransientFEOperator
import Gridap.ODEs: ODESolver, ODEOperator
import Gridap.ODEs: StageOperator, LinearStageOperator, NonlinearOperator

import Gridap.ODEs: AbstractQuasilinearODE,  AbstractLinearODE, AbstractSemilinearODE
import Gridap.ODEs: get_weights, get_nodes, get_matrix
import Gridap.ODEs: LinearODE, QuasilinearODE, SemilinearODE, NonlinearODE
import Gridap.ODEs: ODEOperatorType
import Gridap.ODEs: allocate_tfeopcache

import Gridap.ODEs: get_order, get_trial, get_test, allocate_space, get_jacs
import Gridap.ODEs: _make_uh_from_us

import Gridap.ODEs: TransientFESolution, ODESolution

import Gridap.Algebra: NonlinearSolver
import Gridap.Algebra: LinearSolver, LinearSolverCache
import Gridap.Algebra: symbolic_setup, numerical_setup, numerical_setup!
import Gridap.Algebra: NLSolver, NLSolversCache

import Gridap.FESpaces: collect_cell_vector, collect_cell_matrix
import Gridap.FESpaces: allocate_matrix, assemble_vector_add!, assemble_matrix_add!, allocate_vector
import Gridap.FESpaces: assemble_matrix!, assemble_vector!
import Gridap.FESpaces: FEOperatorFromWeakForm, FEOperator
import Gridap.FESpaces: EvaluationFunction

import Gridap.Helpers: @unreachable
import Gridap.Helpers: @abstractmethod

import Gridap.CellData: DomainContribution, num_domains


include("TSW/TSWODEOperators.jl")
export TSWODEOperator



include("TSW/TSWPoissonIntegrator.jl")
export TSWPoissonIntegrator

include("TSW/TSWDiagNLOperators.jl")
include("TSW/TSWProgNLOperator.jl")

include("TSW/TSWPoissonFEOperator.jl")
export TSWOperator
export TransientTSWOperator

include("TSW/TSWODEOpsFromTFEOps.jl")

include("TSW/TSWTransientFESolutions.jl")
include("TSW/TSWODESolutions.jl")
export solve


include("helpers.jl")
export gradPerp
export vecPerp
export my_sign, upwinding_sign

include("solvers.jl")
export get_solvers

include("mytypes.jl")
export instability, convergence, vortex
export Instability, Convergence, Vortex

include("tsw_funcs.jl")
export get_FEspaces, compute_casimirs
export res_diagw, jac_diagw
export Tfunc, bbarfunc

include("initial_conditions.jl")
export get_testcase


include("Drivers/tsw_entropy.jl")
include("Drivers/convergence_test.jl")
export main_tsw_entropy


const options_cg_gmres = """
-g_ksp_type gmres
-g_ksp_rtol 1.0e-14
-g_ksp_converged_reason
-c_ksp_type cg
-c_ksp_rtol 1.0e-16
-c_ksp_converged_reason
-gj_ksp_type gmres
-gj_ksp_rtol 1.0e-14
-gj_ksp_converged_reason
-ksp_monitor
"""

export options_cg_gmres

end # module PoissonIntegrator

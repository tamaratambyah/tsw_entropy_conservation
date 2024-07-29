using PoissonIntegrator

main_tsw_entropy(;nprocs=(1,1),testcase=convergence,const_jac=true)



# main_tsw_entropy(;nprocs=(1,1),testcase=instability,upwinding=nothing)
# main_tsw_entropy(;nprocs=(1,1),testcase=restarted,tF=[0.02],upwinding=nothing)
main_tsw_entropy(;nprocs=(1,1),testcase=instability,upwinding=true,const_jac=true)
main_tsw_entropy(;nprocs=(1,1),testcase=restarted,tF=[0.02],upwinding=true,const_jac=true)

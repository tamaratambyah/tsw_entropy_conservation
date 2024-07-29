
function run_tsw_entropy(case::Dict,ranks,nprocs,options,out_dir,testcase::tswconvergence,upwinding)

  @unpack n, p, tF, CFL, out_loc, out_freq, const_jac, nls_tols = case

  u, h, B, H0, g, f, L, simName, convert2seconds, domain = get_testcase(testcase)

  save_tsw_test(n,p,u,h,B,out_dir,ranks,nprocs,options,domain)

  i_am_main(ranks) && println("convergence test -- ", simName)

  dx = L/n
  uu = sqrt(g*H0)
  C = CFL/(uu*p^2)
  dt = C*dx
  tFd = tF*convert2seconds
  i_am_main(ranks) && println("tf = ", tFd)
  i_am_main(ranks) && println("CFL = ",CFL,"; dt = ", dt, "; dx = ", dx, "; n = ", n)

  results = convergence_test(ranks,nprocs,options,n,dt,tFd,p,g,H0,f,L,domain,
                            u,h,B,convert2seconds,out_dir,out_freq,const_jac,nls_tols)

  simparams = @strdict dx dt C simName

  merge(case,simparams,results)


end

function run_tsw_entropy(case::Dict,ranks,nprocs,options,out_dir,testcase::tswconvergencerestarted,upwinding)

  @unpack n, p, tF, CFL, out_loc, out_freq, const_jac, nls_tols = case

  i_am_main(ranks) && println("restarting convergence test")

  results = restarted_convergence_test(ranks,nprocs,options,out_dir,out_freq,const_jac::Bool,nls_tols)

  merge(case,results)


end

function convergence_test(ranks,nprocs,options,n,dt,tF,p,g,H0,f,L,domain,u,h,B,convert2seconds,
                out_dir,out_freq,const_jac::Bool,nls_tols)


  i_am_main(ranks) && println("tsw convergence test")
  dir_meta = datadir(out_dir*"/meta")

  el2_u = 0.0
  el2_h = 0.0
  el2_B = 0.0
  T = 0.0
  el2_uold = 0.0
  el2_hold = 0.0
  el2_Bold = 0.0
  Told = 0.0

  degree = 5*(p+1)

  partition = (n, n)
  model = CartesianDiscreteModel(ranks, nprocs,domain, partition, isperiodic=(true,true))

  Ω = Triangulation(model)
  dΩ = Measure(Ω, degree)
  Λ = SkeletonTriangulation(model)
  dΛ = Measure(Λ,degree)
  n_Λ = get_normal_vector(Λ)

  # u space
  V = TestFESpace(model,
                    ReferenceFE(raviart_thomas,Float64,p), #p-1
                    conformity=:Hdiv
                    )
  U = TransientTrialFESpace(V)

  # h space, B space, T space
  W = TestFESpace(model,
                  ReferenceFE(lagrangian ,Float64,p),
                  conformity=:L2)
  R = TransientTrialFESpace(W)

  # F space
  V_F = TestFESpace(model,
                    ReferenceFE(raviart_thomas,Float64,p),
                    conformity=:Hdiv)
  U_F = TransientTrialFESpace(V_F)

  # Φ space
  W_Φ = TestFESpace(model,
                  ReferenceFE(lagrangian ,Float64,p),  #2*p
                  conformity=:L2)
  R_Φ = TransientTrialFESpace(W_Φ)

  # b space
  W_b = TestFESpace(model,
                  ReferenceFE(lagrangian ,Float64,p), #p+1
                  conformity=:L2)
  R_b = TransientTrialFESpace(W_b)


  # b^2 space
  W_b2 = TestFESpace(model,
                  ReferenceFE(lagrangian ,Float64,2*(p)), #2*(p+1)
                  conformity=:L2)
  R_b2 = TransientTrialFESpace(W_b2)

  # btilde space
  W_btilde = TestFESpace(model,
                  ReferenceFE(lagrangian ,Float64,(p)), #2*(p+1)
                  conformity=:L2)
  R_btilde = TransientTrialFESpace(W_btilde)

  # entropy space
  W_S = TestFESpace(model,
                ReferenceFE(lagrangian ,Float64,3*(p)), #2*(p+1)
                conformity=:L2)
  R_S = TransientTrialFESpace(W_S)

  # q space
  P_q = TestFESpace(model,
                  ReferenceFE(lagrangian,Float64,p+1),
                  conformity=:H1)
  H_q = TransientTrialFESpace(P_q)


  ###
  l2(w) = sqrt(sum( ∫( w⊙w )dΩ ))

  # b1 = \bar{b}     = 0.5 (b^n + b^k)
  # b2 = \hat{b}     = 1/3( (b^n)^2 + b^n b^k + (b^k)^2 )
  # b3 = \tilde{b}   -->  \tilde{b} \bar{b} = \hat{b}
  # bn               --> b^n h^n = B^n
  # bk = b_{n+1}^{k} --> b^k h^k = B^k



  ########################################
  # Prognostic # u,h,B
  ########################################
  X_prog = TransientMultiFieldFESpace([U,R,R]) # u, h, B
  Y_prog = MultiFieldFESpace([V,W,W]) # u, h, B

  res_prog((t,dt),(u0,h0,B0),(u,h,B),(T),(b1),(b2),(b3),(F,Φ,q,ω),(v,w,r), (b)) = (
          ∫( u⋅v - u0⋅v )dΩ
        + ∫( dt*(q*(vecPerp∘(F)⋅v) ) )dΩ
        - ∫( dt*(∇⋅v)*Φ  )dΩ
        + ∫( dt*0.5*(∇(T)⋅v)*b1  )dΩ - ∫( dt*0.5*(∇⋅v)*(b3*T) )dΩ - ∫( dt*0.5*T*(∇(b1)⋅v) )dΩ
          + ∫( h*w - h0*w )dΩ
        + ∫( dt*(∇⋅F)*w  )dΩ
          + ∫( B*r - B0*r )dΩ
        - ∫( dt*0.5*(∇(r)⋅F)*b1 )dΩ + ∫( dt*0.5*(b3*r)*(∇⋅F) )dΩ  + ∫( dt*0.5*r*(F⋅∇(b1))  )dΩ
          - ∫( dt*0.5*(mean(v*b1)⋅jump(T*n_Λ ) ) )dΛ + ∫( dt*0.5*(mean(v*T)⋅jump(b1*n_Λ ) ) )dΛ # u central
          + ∫( dt*0.5*(mean(F*b1)⋅jump(r*n_Λ ) ) )dΛ - ∫( dt*0.5*(mean(F*r)⋅jump(b1*n_Λ ))  )dΛ # B central
        #  + ∫( dt*0.5*(my_sign∘( (F⋅n_Λ).plus )*( (F⋅n_Λ).plus ) )*jump(b1*n_Λ )⋅jump(r*n_Λ )   )dΛ # B upwinding
        #  - ∫( dt*0.5*(my_sign∘( (F⋅n_Λ).plus )*( (v⋅n_Λ).plus ) )*jump(b1*n_Λ )⋅jump(T*n_Λ )   )dΛ # u upwinding
  )


  c = 0.5
  jac_prog((t,dt),(u0,h0,B0),(u,h,B),(du,dh,dB),(v,w,r),b,(F,Φ,q,ω),b3) = (
          ∫( du⋅v  )dΩ
        + ∫( (c*dt)*(ω*(vecPerp∘(du)⋅v) )  )dΩ
        - ∫( ((c*dt)*(1/2))*dB*(∇⋅v) )dΩ
        - ∫( ((c*dt)*(1/2)*b*dh)*(∇⋅v )  )dΩ
        + ∫( dh*w   )dΩ
        + ∫( (c*dt)*h0*(∇⋅du)*w  )dΩ
        + ∫( dB*r )dΩ
        + ∫( ((c*dt)*b*h0)*(∇⋅du)*r )dΩ
  )

  #   jac_prog((t,dt),(u0,h0,B0),(u,h,B),(du,dh,dB),(v,w,r),b) = ( ## Not a function of diagnostic variables
  #           ∫( du⋅v  )dΩ
  #         + ∫( (0.5*dt)*(f*(vecPerp∘(du)⋅v) )  )dΩ
  #         - ∫( ((0.5*dt)*(1/2))*dB*(∇⋅v) )dΩ
  #         - ∫( ((0.5*dt)*(1/2)*g)*dh*(∇⋅v)  )dΩ
  #         + ∫( dh*w  )dΩ
  #         + ∫( (0.5*dt)*H0*(∇⋅du)*w  )dΩ
  #         + ∫( dB*r )dΩ
  #         + ∫( ((0.5*dt)*g*H0)*(∇⋅du)*r )dΩ
  # )


  ########################################
  # Diagnostics # F,Φ,q,ω
  # diagnosed at every newton iteration
  ########################################
  X_diag = TransientMultiFieldFESpace([U_F,R_Φ,H_q,H_q]) # F, Φ, q, ω
  Y_diag = MultiFieldFESpace([V_F,W_Φ,P_q,P_q]) # F, Φ, q, ω

  res_diag((t,dt),(u0,h0,B0),(u,h,B),(F,Φ,q,ω),(s,ψ,p,ϕ)) = (
          ∫( F⋅s - (1.0/3.0)*h0*(u0⋅s) - (1.0/6.0)*h0*(u⋅s)
                  - (1/6)*h*(u0⋅s) - (1/3)*h*(u⋅s) )dΩ
        + ∫( Φ*ψ - (1.0/6.0)*(u0⋅u0)*ψ - (1.0/6.0)*(u0⋅u)*ψ
                  - (1.0/6.0)*(u⋅u)*ψ
                  - 0.25*B0*ψ - 0.25*B*ψ )dΩ
        + ∫( 0.5*q*h*p + 0.5*q*h0*p
            + 0.5*(gradPerp∘(∇(p))⋅u) + 0.5*(gradPerp∘(∇(p))⋅u0)
            - f*p )dΩ
        + ∫( ω*ϕ
            + 0.5*(gradPerp∘(∇(ϕ))⋅u) + 0.5*(gradPerp∘(∇(ϕ))⋅u0)
            - f*ϕ )dΩ
  )

  jac_diag((t,dt),(u0,h0,B0),(u,h,B),(F,Φ,q,ω),(dF,dΦ,dq,dω),(s,ψ,p,ϕ)) = (
          ∫( dF⋅s )dΩ
        + ∫( dΦ*ψ  )dΩ
        + ∫( 0.5*dq*h*p + 0.5*dq*h0*p )dΩ
        + ∫( dω*ϕ )dΩ
  )

  ########################################
  # Diagnostics # b
  ########################################
  X_diagb = R_b # b
  Y_diagb = W_b # b

  res_diagb((t,dt),(u0,h0,B0),(u,h,B),b,l) = (
    ∫( (b*( h  ) )*l - ( B )*l   )dΩ
  )

  jac_diagb((t,dt),(u0,h0,B0),(u,h,B),b,db,l) = (
    ∫( (db*(h) )*l    )dΩ
  )

  ########################################
  # Diagnostics # T, bbar (b1)
  # diagnosed after each solve
  ########################################
  Tfunc((u0,h0,B0),(u,h,B)) = 0.25*(h + h0 )
  bbarfunc(b0,b) = 0.5*( b0 + b )
  Tspace = R
  bbarspace = R_b

  ########################################
  # Diagnostics # bhat (b2)
  # diagnosed after each bbar compute
  ########################################

  X_diagw = R_b2 # b2
  Y_diagw = W_b2 # b2

  res_diagw((t,dt),(u0,h0,B0),(u,h,B),b0,b,b2,l2) = (
          ∫( b2*l2 - (1.0/3.0)*( b0*b0 + b0*b + b*b )*l2   )dΩ
          # ∫( b2*l2 - (1.0/2.0)*( b0*b0 + b*b )*l2   )dΩ
  )

  jac_diagw((t,dt),(u0,h0,B0),(u,h,B),b2,db2,l2) = (
          ∫( db2*l2  )dΩ
  )


  ########################################
  # Diagnostics # btilde (b3)
  # diagnosed after each bhat compute
  ########################################

  X_diagz = R_btilde # b3
  Y_diagz = W_btilde # b3

  res_diagz((t,dt),(F,Φ,q,ω),b1,b2,b3,l3,b) = (
    ∫( ((b3*b1))*l3 - b2*l3   )dΩ
  )

  jac_diagz((t,dt),(F,Φ,q,ω),b1,b2,b3,db3,l3,b) = (
    ∫( (db3*b1)*l3   )dΩ
  )



  ########################################
  # Casimirs
  ########################################
  energy(u,h,B) =  sum( ∫(0.5*h*(u⋅u) + 0.5*h*B  )dΩ)
  Mass(h) =    sum( ∫( h )dΩ)
  vorticity(u,h,q) = sum( ∫( h*q-f )dΩ) #sum( ∫( dotPerp∘(∇(u)) )dΩ)
  vorticity2(u,h,q) = sum( ∫( h*q )dΩ)
  vorticityq(q) = sum( ∫( q )dΩ)
  densityB(h,b) =   sum( ∫( h*b )dΩ)
  entropy_internal(b1,b2,b3,F) = sum( ∫( (∇⋅F)*(-1/2*b2)  )dΩ
                                   + ∫( 0.5*(b3*b1)*(∇⋅F) )dΩ
                                   + ∫( 0.5*b1*(F⋅∇(b1))  )dΩ
                                  - ∫( 0.5*(∇(b1)⋅F)*b1 )dΩ
  )

  entropy_center(b1,F) = sum( ∫( 0.5*(mean(F*b1)⋅jump(b1*n_Λ ) ) )dΛ - ∫( 0.5*(mean(F*b1)⋅jump(b1*n_Λ ))  )dΛ   )

  entropy_upwind(b,F) = sum( ∫( (my_sign∘( (F⋅n_Λ).plus )*( (F⋅n_Λ).plus ) )*jump(b*n_Λ )⋅jump(b*n_Λ )   )dΛ )
  corio(f) =  sum( ∫( f )dΩ)

  ########################################
  # Sovlers
  ########################################
  # ode solvers
  ls = PETScLinearSolver(petsc_ls_from_options_c)
  nls_ls = PETScLinearSolver(petsc_ls_from_options_g)
  nls = GridapSolvers.NonlinearSolvers.NewtonSolver(nls_ls;maxiter=nls_tols.maxiter,atol=nls_tols.atol,rtol=nls_tols.rtol,verbose=i_am_main(ranks))

  # initial condition solvers
  _cg = PETScLinearSolver(petsc_ls_from_options_c)
  _gmres = PETScLinearSolver(petsc_gmres_jacobi)
  _nls_gmres = GridapSolvers.NonlinearSolvers.NewtonSolver(_gmres;maxiter=nls_tols.maxiter,atol=1e-14,rtol=1.e-14,verbose=i_am_main(ranks))
  _nls_cg = GridapSolvers.NonlinearSolvers.NewtonSolver(_cg;maxiter=nls_tols.maxiter,atol=1e-14,rtol=1.e-14,verbose=i_am_main(ranks))


  ########################################
  # Initial conditions
  ########################################
  a((u,h,B),(v,w,r)) = ∫( u⋅v + h*w + B*r)dΩ
  l((v,w,r)) = ∫( u(0.0)⋅v + h(0.0)*w + B(0.0)*r  )dΩ
  op = AffineFEOperator(a,l,X_prog(0.0),Y_prog(0.0))
  xh0 = solve(_cg,op)
  uh0, hh0, Bh0 = xh0


  # T
  Th0 = Tfunc(xh0,xh0)

  _res_diagb(b,l) = res_diagb((0.0,0.0),xh0,xh0,(b),(l) )
  _jac_diagb(b,db,l) = jac_diagb((0.0,0.0),xh0,xh0,(b),(db),(l))
  _op_diagb = FEOperator(_res_diagb,_jac_diagb,X_diagb(0.0),Y_diagb)
  bh0 = solve(_nls_cg,_op_diagb)

  b1h0 = interpolate_everywhere(bbarfunc(bh0,bh0), bbarspace(0.0))

  res_diag00((F,Φ,q,ω),(s,ψ,p,ϕ)) = (
        res_diag((0.0,0.0),xh0,xh0,(F,Φ,q,ω),(s,ψ,p,ϕ))
  )
  jac_diag00((F,Φ,q,ω),(dF,dΦ,dq,dω),(s,ψ,p,ϕ)) =  (
        jac_diag((0.0,0.0),xh0,xh0,(F,Φ,q,ω),(dF,dΦ,dq,dω),(s,ψ,p,ϕ))
  )
  op_diag00 = FEOperator(res_diag00,jac_diag00,X_diag(0.0),Y_diag)
  yh0 = solve(_nls_cg,op_diag00)
  Fh0,Φh0,qh0 = yh0

  _res_diagw(b2,l2) = res_diagw((0.0,0.0),xh0,xh0,bh0,bh0,b2,l2)
  _jac_diagw(b2,db2,l2) = jac_diagw((0.0,0.0),xh0,xh0,b2,db2,l2)
  op_diagw0 = FEOperator(_res_diagw,_jac_diagw,X_diagw(0.0),Y_diagw)
  wh0 = solve(_nls_cg,op_diagw0)
  b2h0 = wh0

  _res_diagz((b3),(l3)) = res_diagz((0.0,0.0),yh0,b1h0,b2h0,b3,l3,bh0)
  _jac_diagz((b3),(db3),(l3)) = jac_diagz((0.0,0.0),yh0,b1h0,b2h0,b3,db3,l3,bh0)
  op_diagz0 = FEOperator(_res_diagz,_jac_diagz,X_diagz(0.0),Y_diagz)
  zh0 = solve(_nls_gmres,op_diagz0)
  b3h0 = zh0


  i_am_main(ranks) && println("done")


  ########################################
  # Operators
  ########################################
  op_prog = FEOperator(res_prog,jac_prog,X_prog,Y_prog)
  op_diag = FEOperator(res_diag,jac_diag,X_diag,Y_diag)
  op_diagb = FEOperator(res_diagb,jac_diagb,X_diagb,Y_diagb)
  op_diagw = FEOperator(res_diagw,jac_diagw,X_diagw,Y_diagw)
  op_diagz = FEOperator(res_diagz,jac_diagz,X_diagz,Y_diagz)

  op_tsw = TSWOperator(op_prog,op_diag,op_diagb,op_diagw,op_diagz,Tfunc,Tspace,bbarfunc,bbarspace)
  opT = TransientTSWOperator(op_tsw, const_jac=const_jac)

  t0 = 0.0

  ode_solver = TSWPoissonIntegrator(nls,ls,dt)

  sol_t  = solve(ode_solver,opT,t0,tF,xh0, bh0, yh0, wh0, zh0)

  i_am_main(ranks) && println("initial iterate")
  it = iterate(sol_t)

  counter = 1

  while !isnothing(it)

    el2_uold = el2_u
    el2_hold = el2_h
    el2_Bold = el2_B
    Told = T


    data, state = it
    t, xh = data
    _tF, stateF, state0, uF, odecache,
            bF,yF,wF,zF ,
            diagnosticsbF,diagnosticsF,diagnosticswF,diagnosticszF,
            diagnosticsb0, diagnostics0,diagonsticsw0,diagnosticsz0 = state[2]


    td = t/convert2seconds

    # extract solution at time t
    xh = FEFunction(X_prog(t),stateF[1])
    uh,hh,Bh = xh

    T = t
    el2_u = l2( uh0 - uh )/l2(uh0)
    el2_h = l2( hh0 - hh)/l2(hh0)
    el2_B = l2( Bh0 - Bh)/l2(Bh0)

    i_am_main(ranks) && println("t = ", td, "; tF = ", tF)
    i_am_main(ranks) && println(el2_u, "; ", el2_h, "; ", el2_B)

    if el2_u > 100.0
      i_am_main(ranks) && println("breaking early")
      break
    end
    if el2_h > 100.0
      i_am_main(ranks) && println("breaking early")
      break
    end
    if el2_B > 100.0
      i_am_main(ranks) && println("breaking early")
      break
    end

    if mod(counter,out_freq) == 0

      output = @strdict el2_u el2_h el2_B T el2_uold el2_hold el2_Bold Told
      i_am_main(ranks) && safesave(datadir(out_dir, ("tsw_convergence.jld2")), output)

      # save dofs
      i_am_main(ranks) && println("saving dofs and meta data")
      output_dof = @strdict t n p tF dt g H0 f L domain counter convert2seconds
      i_am_main(ranks) && safesave(out_dir*"/dofs.jld2", output_dof)
      psave(dir_meta, (xh.metadata.free_values))

    end


    it = iterate(sol_t, state)
    counter = counter + 1

  end


  results = @strdict el2_u el2_h el2_B T el2_uold el2_hold el2_Bold Told
  return results
end

function restarted_convergence_test(ranks,nprocs,options,
  out_dir,out_freq,const_jac::Bool,nls_tols)


  i_am_main(ranks) && println("convergence test -- RESTARTED")

  df = wload(datadir(out_dir, "dofs.jld2"))
  df2 = wload(datadir(out_dir, "tsw_convergence.jld2"))

  @unpack t, n, p, tF, dt, g, H0, f, L, domain, counter, convert2seconds = df
  @unpack el2_u, el2_h, el2_B, T, el2_uold, el2_hold, el2_Bold, Told = df2


  dir_meta = datadir(out_dir*"/meta")
  meta_load = pload(dir_meta,ranks)
  my_metadata = GridapDistributed.DistributedFEFunctionData(meta_load)

  i_am_main(ranks) && println("loaded data")

  i_am_main(ranks) && println("starting from t = ",t)
  i_am_main(ranks) && println("running to tF = ", tF)

  degree = 5*(p+1)

  partition = (n, n)
  model = CartesianDiscreteModel(ranks, nprocs,domain, partition, isperiodic=(true,true))

  Ω = Triangulation(model)
  dΩ = Measure(Ω, degree)
  Λ = SkeletonTriangulation(model)
  dΛ = Measure(Λ,degree)
  n_Λ = get_normal_vector(Λ)

  # u space
  V = TestFESpace(model,
        ReferenceFE(raviart_thomas,Float64,p), #p-1
        conformity=:Hdiv
        )
  U = TransientTrialFESpace(V)

  # h space, B space, T space
  W = TestFESpace(model,
      ReferenceFE(lagrangian ,Float64,p),
      conformity=:L2)
  R = TransientTrialFESpace(W)

  # F space
  V_F = TestFESpace(model,
        ReferenceFE(raviart_thomas,Float64,p),
        conformity=:Hdiv)
  U_F = TransientTrialFESpace(V_F)

  # Φ space
  W_Φ = TestFESpace(model,
      ReferenceFE(lagrangian ,Float64,p),  #2*p
      conformity=:L2)
  R_Φ = TransientTrialFESpace(W_Φ)

  # b space
  W_b = TestFESpace(model,
      ReferenceFE(lagrangian ,Float64,p), #p+1
      conformity=:L2)
  R_b = TransientTrialFESpace(W_b)


  # b^2 space
  W_b2 = TestFESpace(model,
      ReferenceFE(lagrangian ,Float64,2*(p)), #2*(p+1)
      conformity=:L2)
  R_b2 = TransientTrialFESpace(W_b2)

  # btilde space
  W_btilde = TestFESpace(model,
      ReferenceFE(lagrangian ,Float64,(p)), #2*(p+1)
      conformity=:L2)
  R_btilde = TransientTrialFESpace(W_btilde)

  # entropy space
  W_S = TestFESpace(model,
    ReferenceFE(lagrangian ,Float64,3*(p)), #2*(p+1)
    conformity=:L2)
  R_S = TransientTrialFESpace(W_S)

  # q space
  P_q = TestFESpace(model,
      ReferenceFE(lagrangian,Float64,p+1),
      conformity=:H1)
  H_q = TransientTrialFESpace(P_q)


  ###
  l2(w) = sqrt(sum( ∫( w⊙w )dΩ ))

  # b1 = \bar{b}     = 0.5 (b^n + b^k)
  # b2 = \hat{b}     = 1/3( (b^n)^2 + b^n b^k + (b^k)^2 )
  # b3 = \tilde{b}   -->  \tilde{b} \bar{b} = \hat{b}
  # bn               --> b^n h^n = B^n
  # bk = b_{n+1}^{k} --> b^k h^k = B^k



  ########################################
  # Prognostic # u,h,B
  ########################################
  X_prog = TransientMultiFieldFESpace([U,R,R]) # u, h, B
  Y_prog = MultiFieldFESpace([V,W,W]) # u, h, B

  res_prog((t,dt),(u0,h0,B0),(u,h,B),(T),(b1),(b2),(b3),(F,Φ,q,ω),(v,w,r), (b)) = (
  ∫( u⋅v - u0⋅v )dΩ
  + ∫( dt*(q*(vecPerp∘(F)⋅v) ) )dΩ
  - ∫( dt*(∇⋅v)*Φ  )dΩ
  + ∫( dt*0.5*(∇(T)⋅v)*b1  )dΩ - ∫( dt*0.5*(∇⋅v)*(b3*T) )dΩ - ∫( dt*0.5*T*(∇(b1)⋅v) )dΩ
  + ∫( h*w - h0*w )dΩ
  + ∫( dt*(∇⋅F)*w  )dΩ
  + ∫( B*r - B0*r )dΩ
  - ∫( dt*0.5*(∇(r)⋅F)*b1 )dΩ + ∫( dt*0.5*(b3*r)*(∇⋅F) )dΩ  + ∫( dt*0.5*r*(F⋅∇(b1))  )dΩ
  - ∫( dt*0.5*(mean(v*b1)⋅jump(T*n_Λ ) ) )dΛ + ∫( dt*0.5*(mean(v*T)⋅jump(b1*n_Λ ) ) )dΛ # u central
  + ∫( dt*0.5*(mean(F*b1)⋅jump(r*n_Λ ) ) )dΛ - ∫( dt*0.5*(mean(F*r)⋅jump(b1*n_Λ ))  )dΛ # B central
  #  + ∫( dt*0.5*(my_sign∘( (F⋅n_Λ).plus )*( (F⋅n_Λ).plus ) )*jump(b1*n_Λ )⋅jump(r*n_Λ )   )dΛ # B upwinding
  #  - ∫( dt*0.5*(my_sign∘( (F⋅n_Λ).plus )*( (v⋅n_Λ).plus ) )*jump(b1*n_Λ )⋅jump(T*n_Λ )   )dΛ # u upwinding
  )


  c = 0.5
  jac_prog((t,dt),(u0,h0,B0),(u,h,B),(du,dh,dB),(v,w,r),b,(F,Φ,q,ω),b3) = (
  ∫( du⋅v  )dΩ
  + ∫( (c*dt)*(ω*(vecPerp∘(du)⋅v) )  )dΩ
  - ∫( ((c*dt)*(1/2))*dB*(∇⋅v) )dΩ
  - ∫( ((c*dt)*(1/2)*b*dh)*(∇⋅v )  )dΩ
  + ∫( dh*w   )dΩ
  + ∫( (c*dt)*h0*(∇⋅du)*w  )dΩ
  + ∫( dB*r )dΩ
  + ∫( ((c*dt)*b*h0)*(∇⋅du)*r )dΩ
  )

  #   jac_prog((t,dt),(u0,h0,B0),(u,h,B),(du,dh,dB),(v,w,r),b) = ( ## Not a function of diagnostic variables
  #           ∫( du⋅v  )dΩ
  #         + ∫( (0.5*dt)*(f*(vecPerp∘(du)⋅v) )  )dΩ
  #         - ∫( ((0.5*dt)*(1/2))*dB*(∇⋅v) )dΩ
  #         - ∫( ((0.5*dt)*(1/2)*g)*dh*(∇⋅v)  )dΩ
  #         + ∫( dh*w  )dΩ
  #         + ∫( (0.5*dt)*H0*(∇⋅du)*w  )dΩ
  #         + ∫( dB*r )dΩ
  #         + ∫( ((0.5*dt)*g*H0)*(∇⋅du)*r )dΩ
  # )


  ########################################
  # Diagnostics # F,Φ,q,ω
  # diagnosed at every newton iteration
  ########################################
  X_diag = TransientMultiFieldFESpace([U_F,R_Φ,H_q,H_q]) # F, Φ, q, ω
  Y_diag = MultiFieldFESpace([V_F,W_Φ,P_q,P_q]) # F, Φ, q, ω

  res_diag((t,dt),(u0,h0,B0),(u,h,B),(F,Φ,q,ω),(s,ψ,p,ϕ)) = (
  ∫( F⋅s - (1.0/3.0)*h0*(u0⋅s) - (1.0/6.0)*h0*(u⋅s)
      - (1/6)*h*(u0⋅s) - (1/3)*h*(u⋅s) )dΩ
  + ∫( Φ*ψ - (1.0/6.0)*(u0⋅u0)*ψ - (1.0/6.0)*(u0⋅u)*ψ
      - (1.0/6.0)*(u⋅u)*ψ
      - 0.25*B0*ψ - 0.25*B*ψ )dΩ
  + ∫( 0.5*q*h*p + 0.5*q*h0*p
  + 0.5*(gradPerp∘(∇(p))⋅u) + 0.5*(gradPerp∘(∇(p))⋅u0)
  - f*p )dΩ
  + ∫( ω*ϕ
  + 0.5*(gradPerp∘(∇(ϕ))⋅u) + 0.5*(gradPerp∘(∇(ϕ))⋅u0)
  - f*ϕ )dΩ
  )

  jac_diag((t,dt),(u0,h0,B0),(u,h,B),(F,Φ,q,ω),(dF,dΦ,dq,dω),(s,ψ,p,ϕ)) = (
  ∫( dF⋅s )dΩ
  + ∫( dΦ*ψ  )dΩ
  + ∫( 0.5*dq*h*p + 0.5*dq*h0*p )dΩ
  + ∫( dω*ϕ )dΩ
  )

  ########################################
  # Diagnostics # b
  ########################################
  X_diagb = R_b # b
  Y_diagb = W_b # b

  res_diagb((t,dt),(u0,h0,B0),(u,h,B),b,l) = (
  ∫( (b*( h  ) )*l - ( B )*l   )dΩ
  )

  jac_diagb((t,dt),(u0,h0,B0),(u,h,B),b,db,l) = (
  ∫( (db*(h) )*l    )dΩ
  )

  ########################################
  # Diagnostics # T, bbar (b1)
  # diagnosed after each solve
  ########################################
  Tfunc((u0,h0,B0),(u,h,B)) = 0.25*(h + h0 )
  bbarfunc(b0,b) = 0.5*( b0 + b )
  Tspace = R
  bbarspace = R_b

  ########################################
  # Diagnostics # bhat (b2)
  # diagnosed after each bbar compute
  ########################################

  X_diagw = R_b2 # b2
  Y_diagw = W_b2 # b2

  res_diagw((t,dt),(u0,h0,B0),(u,h,B),b0,b,b2,l2) = (
  ∫( b2*l2 - (1.0/3.0)*( b0*b0 + b0*b + b*b )*l2   )dΩ
  # ∫( b2*l2 - (1.0/2.0)*( b0*b0 + b*b )*l2   )dΩ
  )

  jac_diagw((t,dt),(u0,h0,B0),(u,h,B),b2,db2,l2) = (
  ∫( db2*l2  )dΩ
  )


  ########################################
  # Diagnostics # btilde (b3)
  # diagnosed after each bhat compute
  ########################################

  X_diagz = R_btilde # b3
  Y_diagz = W_btilde # b3

  res_diagz((t,dt),(F,Φ,q,ω),b1,b2,b3,l3,b) = (
  ∫( ((b3*b1))*l3 - b2*l3   )dΩ
  )

  jac_diagz((t,dt),(F,Φ,q,ω),b1,b2,b3,db3,l3,b) = (
  ∫( (db3*b1)*l3   )dΩ
  )



  ########################################
  # Casimirs
  ########################################
  energy(u,h,B) =  sum( ∫(0.5*h*(u⋅u) + 0.5*h*B  )dΩ)
  Mass(h) =    sum( ∫( h )dΩ)
  vorticity(u,h,q) = sum( ∫( h*q-f )dΩ) #sum( ∫( dotPerp∘(∇(u)) )dΩ)
  vorticity2(u,h,q) = sum( ∫( h*q )dΩ)
  vorticityq(q) = sum( ∫( q )dΩ)
  densityB(h,b) =   sum( ∫( h*b )dΩ)
  entropy_internal(b1,b2,b3,F) = sum( ∫( (∇⋅F)*(-1/2*b2)  )dΩ
                      + ∫( 0.5*(b3*b1)*(∇⋅F) )dΩ
                      + ∫( 0.5*b1*(F⋅∇(b1))  )dΩ
                      - ∫( 0.5*(∇(b1)⋅F)*b1 )dΩ
  )

  entropy_center(b1,F) = sum( ∫( 0.5*(mean(F*b1)⋅jump(b1*n_Λ ) ) )dΛ - ∫( 0.5*(mean(F*b1)⋅jump(b1*n_Λ ))  )dΛ   )

  entropy_upwind(b,F) = sum( ∫( (my_sign∘( (F⋅n_Λ).plus )*( (F⋅n_Λ).plus ) )*jump(b*n_Λ )⋅jump(b*n_Λ )   )dΛ )
  corio(f) =  sum( ∫( f )dΩ)

  ########################################
  # Sovlers
  ########################################
  # ode solvers
  ls = PETScLinearSolver(petsc_ls_from_options_c)
  nls_ls = PETScLinearSolver(petsc_ls_from_options_g)
  nls = GridapSolvers.NonlinearSolvers.NewtonSolver(nls_ls;maxiter=nls_tols.maxiter,atol=nls_tols.atol,rtol=nls_tols.rtol,verbose=i_am_main(ranks))

  # initial condition solvers
  _cg = PETScLinearSolver(petsc_ls_from_options_c)
  _gmres = PETScLinearSolver(petsc_gmres_jacobi)
  _nls_gmres = GridapSolvers.NonlinearSolvers.NewtonSolver(_gmres;maxiter=nls_tols.maxiter,atol=1e-14,rtol=1.e-14,verbose=i_am_main(ranks))
  _nls_cg = GridapSolvers.NonlinearSolvers.NewtonSolver(_cg;maxiter=nls_tols.maxiter,atol=1e-14,rtol=1.e-14,verbose=i_am_main(ranks))


  ########################################
  # Initial conditions
  ########################################
  xh0 = FEFunction(X_prog, my_metadata.free_values)
  uh0, hh0, Bh0 = xh0


  # T
  Th0 = Tfunc(xh0,xh0)

  _res_diagb(b,l) = res_diagb((0.0,0.0),xh0,xh0,(b),(l) )
  _jac_diagb(b,db,l) = jac_diagb((0.0,0.0),xh0,xh0,(b),(db),(l))
  _op_diagb = FEOperator(_res_diagb,_jac_diagb,X_diagb(0.0),Y_diagb)
  bh0 = solve(_nls_cg,_op_diagb)

  b1h0 = interpolate_everywhere(bbarfunc(bh0,bh0), bbarspace(0.0))

  res_diag00((F,Φ,q,ω),(s,ψ,p,ϕ)) = (
  res_diag((0.0,0.0),xh0,xh0,(F,Φ,q,ω),(s,ψ,p,ϕ))
  )
  jac_diag00((F,Φ,q,ω),(dF,dΦ,dq,dω),(s,ψ,p,ϕ)) =  (
  jac_diag((0.0,0.0),xh0,xh0,(F,Φ,q,ω),(dF,dΦ,dq,dω),(s,ψ,p,ϕ))
  )
  op_diag00 = FEOperator(res_diag00,jac_diag00,X_diag(0.0),Y_diag)
  yh0 = solve(_nls_cg,op_diag00)
  Fh0,Φh0,qh0 = yh0

  _res_diagw(b2,l2) = res_diagw((0.0,0.0),xh0,xh0,bh0,bh0,b2,l2)
  _jac_diagw(b2,db2,l2) = jac_diagw((0.0,0.0),xh0,xh0,b2,db2,l2)
  op_diagw0 = FEOperator(_res_diagw,_jac_diagw,X_diagw(0.0),Y_diagw)
  wh0 = solve(_nls_cg,op_diagw0)
  b2h0 = wh0

  _res_diagz((b3),(l3)) = res_diagz((0.0,0.0),yh0,b1h0,b2h0,b3,l3,bh0)
  _jac_diagz((b3),(db3),(l3)) = jac_diagz((0.0,0.0),yh0,b1h0,b2h0,b3,db3,l3,bh0)
  op_diagz0 = FEOperator(_res_diagz,_jac_diagz,X_diagz(0.0),Y_diagz)
  zh0 = solve(_nls_gmres,op_diagz0)
  b3h0 = zh0


  i_am_main(ranks) && println("done")


  ########################################
  # Operators
  ########################################
  op_prog = FEOperator(res_prog,jac_prog,X_prog,Y_prog)
  op_diag = FEOperator(res_diag,jac_diag,X_diag,Y_diag)
  op_diagb = FEOperator(res_diagb,jac_diagb,X_diagb,Y_diagb)
  op_diagw = FEOperator(res_diagw,jac_diagw,X_diagw,Y_diagw)
  op_diagz = FEOperator(res_diagz,jac_diagz,X_diagz,Y_diagz)

  op_tsw = TSWOperator(op_prog,op_diag,op_diagb,op_diagw,op_diagz,Tfunc,Tspace,bbarfunc,bbarspace)
  opT = TransientTSWOperator(op_tsw, const_jac=const_jac)

  t0 = t

  ode_solver = TSWPoissonIntegrator(nls,ls,dt)

  sol_t  = solve(ode_solver,opT,t0,tF,xh0, bh0, yh0, wh0, zh0)

  i_am_main(ranks) && println("initial iterate")
  it = iterate(sol_t)


  while !isnothing(it)

    el2_uold = el2_u
    el2_hold = el2_h
    el2_Bold = el2_B
    Told = T


    data, state = it
    t, xh = data
    _tF, stateF, state0, uF, odecache,
    bF,yF,wF,zF ,
    diagnosticsbF,diagnosticsF,diagnosticswF,diagnosticszF,
    diagnosticsb0, diagnostics0,diagonsticsw0,diagnosticsz0 = state[2]


    td = t/convert2seconds

    # extract solution at time t
    xh = FEFunction(X_prog(t),stateF[1])
    uh,hh,Bh = xh

    T = t
    el2_u = l2( uh0 - uh )/l2(uh0)
    el2_h = l2( hh0 - hh)/l2(hh0)
    el2_B = l2( Bh0 - Bh)/l2(Bh0)

    i_am_main(ranks) && println("t = ", td, "; tF = ", tF)
    i_am_main(ranks) && println(el2_u, "; ", el2_h, "; ", el2_B)

    if el2_u > 100.0
      i_am_main(ranks) && println("breaking early")
      break
    end
    if el2_h > 100.0
      i_am_main(ranks) && println("breaking early")
      break
    end
    if el2_B > 100.0
      i_am_main(ranks) && println("breaking early")
      break
    end

    if mod(counter,out_freq) == 0

      output = @strdict el2_u el2_h el2_B T el2_uold el2_hold el2_Bold Told
      i_am_main(ranks) && safesave(datadir(out_dir, ("tsw_convergence.jld2")), output)

      # save dofs
      i_am_main(ranks) && println("saving dofs and meta data")
      output_dof = @strdict t n p tF dt g H0 f L domain counter convert2seconds
      i_am_main(ranks) && safesave(out_dir*"/dofs.jld2", output_dof)
      psave(dir_meta, (xh.metadata.free_values))

    end


    it = iterate(sol_t, state)
    counter = counter + 1

  end


  results = @strdict el2_u el2_h el2_B T el2_uold el2_hold el2_Bold Told
  return results
end

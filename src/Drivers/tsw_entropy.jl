function main_tsw_entropy(;nprocs,testcase=instability,ps=[1],ns=[16],CFLs=[0.2],tF=[0.01],
      const_jac=true,
      out_loc="temp",out_freq=1,
      options=options_cg_gmres,
      upwinding_parms = (;upwinding=false,const_jac=false,ε=1e-4,soft=false),
      nls_tols = (;atol=1e-16,rtol=1e-11,maxiter=50))


  allparams = Dict(
    "n" => ns,
    "p" => ps,
    "tF" => tF,
    "CFL" => CFLs,
    "out_loc" => out_loc,
    "out_freq" => out_freq,
    "const_jac" => const_jac,
    "nls_tols" => nls_tols
  )

  dicts = dict_list(allparams)

  ranks = with_mpi() do distribute
    distribute(LinearIndices((prod(nprocs),)))
  end

  GridapPETSc.Init(args=split(options))

  main_tsw_entropy(dicts,ranks,nprocs,options,out_loc,testcase,upwinding_parms)

  GridapPETSc.Finalize()
  GridapPETSc.gridap_petsc_gc()
end

function main_tsw_entropy(dicts,ranks,nprocs,options,out_loc,testcase,upwinding_parms)
  i_am_main(ranks) && println("thermal shallow water -- entropy conservation")
  i_am_main(ranks) && println("--START--")

  out_dir = datadir(out_loc)
  (i_am_main(ranks) && !isdir(out_dir))  && mkdir(out_dir)

  for (i, d) in enumerate(dicts)
    s = run_tsw_entropy(d,ranks,nprocs,options,out_dir,testcase,upwinding_parms)
    i_am_main(ranks) && wsave(datadir(out_loc, savename(d,"jld2")), s)
  end

  i_am_main(ranks) && println("--DONE--")

end


function run_tsw_entropy(case::Dict,ranks,nprocs,options,out_dir,testcase::tswtestcase,upwinding_parms)

  @unpack n, p, tF, CFL, out_loc, out_freq, const_jac, nls_tols = case

  u, h, B, H0, g, f, L, simName, convert2seconds, domain = get_testcase(testcase)

  dx = L/n
  uu = sqrt(g*H0)
  C = CFL/(uu*p^2)
  dt = C*dx
  tFd = tF*convert2seconds
  i_am_main(ranks) && println("tf = ", tFd)
  i_am_main(ranks) && println("CFL = ",CFL,"; dt = ", dt, "; dx = ", dx)

  i_am_main(ranks) && println("running tsw $(simName) testcase")

  results = tsw_testcase_entropy(ranks,nprocs,options,n,dt,tFd,p,g,H0,f,L,
            domain,u,h,B,convert2seconds,out_dir,simName,out_freq,const_jac,upwinding_parms,nls_tols)

  simparams = @strdict dx dt C simName

  merge(case,simparams,results)

end


function tsw_testcase_entropy(ranks,nprocs,options,n,dt,tF,p,g,H0,f,L,domain,u,h,B,convert2seconds,
  out_dir,simName,out_freq,
  const_jac::Bool,upwinding_parms,nls_tols)


  ts = []
  Cs = []


  degree = 5*(p+1)


  partition = (n, n)
  model = CartesianDiscreteModel(ranks, nprocs,domain, partition, isperiodic=(true,true))

  Ω = Triangulation(model)
  dΩ = Measure(Ω, degree)
  Λ = SkeletonTriangulation(model)
  dΛ = Measure(Λ,degree)
  n_Λ = get_normal_vector(Λ)

  # make upwinding parms cell fields
  upwinding,_const_jac,_ε,_soft = upwinding_parms
  ε = CellField(_ε,Ω)
  soft = CellField(_soft,Ω)

  spaces, multi_spaces = get_FEspaces(model,p)

  ###

  # b1 = \bar{b}     = 0.5 (b^n + b^k)
  # b2 = \hat{b}     = 1/3( (b^n)^2 + b^n b^k + (b^k)^2 )
  # b3 = \tilde{b}   -->  \tilde{b} \bar{b} = \hat{b}
  # bn               --> b^n h^n = B^n
  # bk = b_{n+1}^{k} --> b^k h^k = B^k


  ########################################
  # Prognostic # u,h,B
  ########################################
  X_prog = multi_spaces.X_prog
  Y_prog = multi_spaces.Y_prog
  c = 0.5 # for jacobian

  function res_prog(dΩ,dΛ,n_Λ)
    _res_prog((t,dt),(u0,h0,B0),(u,h,B),(T),(b1),(b3),(F,Φ,q,ω),(v,w,r), (b), (b0)) = (
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
    )

  end

  function res_prog_upwinding(dΩ,dΛ,n_Λ,_ε,_soft)
    ε = CellField(_ε,Ω)
    soft = CellField(_soft,Ω)

    _res_prog((t,dt),(u0,h0,B0),(u,h,B),(T),(b1),(b3),(F,Φ,q,ω),(v,w,r), (b), (b0)) = (
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
      + ∫( dt*0.5*( upwinding_sign∘( (F⋅n_Λ).plus, ε, soft )*( (F⋅n_Λ).plus ) )*jump(b1*n_Λ )⋅jump(r*n_Λ )   )dΛ # B upwinding
      - ∫( dt*0.5*( upwinding_sign∘( (F⋅n_Λ).plus, ε, soft )*( (v⋅n_Λ).plus ) )*jump(b1*n_Λ )⋅jump(T*n_Λ )   )dΛ # u upwinding
    )

  end

  function jac_prog(dΩ,c)
    _jac_prog((t,dt),(u0,h0,B0),(u,h,B),(du,dh,dB),(v,w,r),(b),(F,Φ,q,ω),b3,b1) = (
        ∫( du⋅v  )dΩ
      + ∫( (c*dt)*(ω*(vecPerp∘(du)⋅v) )  )dΩ
      - ∫( ((c*dt)*(1/2))*dB*(∇⋅v) )dΩ
      - ∫( ((c*dt)*(1/2)*b1*dh)*(∇⋅v )  )dΩ
      + ∫( dh*w   )dΩ
      + ∫( (c*dt)*h0*(∇⋅du)*w  )dΩ
      + ∫( dB*r )dΩ
      + ∫( ((c*dt)*b1*h0)*(∇⋅du)*r )dΩ
    )

  end

  function jac_prog_upwinding(dΩ,c,dΛ,n_Λ,ε,soft)

    _jac_prog((t,dt),(u0,h0,B0),(u,h,B),(du,dh,dB),(v,w,r),(b),(F,Φ,q,ω),b3,b1) = (
        ∫( du⋅v  )dΩ
      + ∫( (c*dt)*(ω*(vecPerp∘(du)⋅v) )  )dΩ
      - ∫( (c*dt)*(∇⋅v)*(0.5*dB)  )dΩ
      + ∫( (c*dt)*0.5*(0.5*∇(dh)⋅v)*b  )dΩ - ∫( (c*dt)*0.5*(∇⋅v)*(b3*0.5*dh) )dΩ - ∫( (c*dt)*0.5*0.5*dh*(∇(b)⋅v) )dΩ
      + ∫( dh*w   )dΩ
      + ∫( (c*dt)*h0*(∇⋅du)*w  )dΩ
      + ∫( dB*r )dΩ
      - ∫( (c*dt)*0.5*(∇(r)⋅du)*(b*h0) )dΩ + ∫( (c*dt)*0.5*(b3*h0*r)*(∇⋅du) )dΩ  + ∫( (c*dt)*(0.5*r*h0)*(du⋅∇(b))  )dΩ
      - ∫( (c*dt)*0.5*(mean(v*b)⋅jump(0.5*dh*n_Λ ) ) )dΛ + ∫( (c*dt)*0.5*(mean(v*0.5*dh)⋅jump(b*n_Λ ) ) )dΛ # u central
      + ∫( (c*dt)*0.5*(mean(h0*du*b)⋅jump(r*n_Λ ) ) )dΛ - ∫( (c*dt)*0.5*(mean(h0*du*r)⋅jump(b*n_Λ ))  )dΛ # B central
      + ∫( (c*dt)*0.5*( upwinding_sign∘( (F⋅n_Λ).plus, ε, soft )*( ( (h0*du)⋅n_Λ).plus)  )*(jump(b*n_Λ )⋅jump(r*n_Λ ) )   )dΛ # B upwinding
      - ∫( (c*dt)*0.5*( upwinding_sign∘( (F⋅n_Λ).plus, ε, soft )*( (v⋅n_Λ).plus ) )*(jump(b*n_Λ )⋅jump( (0.5*dh)*n_Λ ))   )dΛ # u upwinding
    )
  end

  op_prog = FEOperator(res_prog(dΩ,dΛ,n_Λ),jac_prog(dΩ,c),X_prog,Y_prog)

  if upwinding
    i_am_main(ranks) && println("Using upwinding")
    op_prog = FEOperator(res_prog_upwinding(dΩ,dΛ,n_Λ,_ε,_soft),jac_prog_upwinding(dΩ,c,dΛ,n_Λ,ε,soft),X_prog,Y_prog)
  end


  ########################################
  # Diagnostics # F,Φ,q,ω
  # diagnosed at every newton iteration
  ########################################
  X_diag = multi_spaces.X_diag
  Y_diag = multi_spaces.Y_diag
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
  op_diag = FEOperator(res_diag,jac_diag,X_diag,Y_diag)

  ########################################
  # Diagnostics # b
  ########################################
  X_diagb = spaces.R # b
  Y_diagb = spaces.W # b

  res_diagb((t,dt),(u0,h0,B0),(u,h,B),b,l,b0) = (
    ∫( (b*( h  ) )*l - ( B )*l   )dΩ
  )

  jac_diagb((t,dt),(u0,h0,B0),(u,h,B),b,db,l,b0) = (
    ∫( (db*(h) )*l    )dΩ
  )
  op_diagb = FEOperator(res_diagb,jac_diagb,X_diagb,Y_diagb)


  ########################################
  # Diagnostics # T, bbar (b1)
  # diagnosed after each solve
  ########################################
  Tspace = spaces.R
  bbarspace = spaces.R

  ########################################
  # Diagnostics # bhat (b2)
  # diagnosed after each bbar compute
  ########################################

  X_diagw = spaces.R # b2
  Y_diagw = spaces.W # b2
  op_diagw = FEOperator(res_diagw(dΩ),jac_diagw(dΩ),X_diagw,Y_diagw)

  ########################################
  # Diagnostics # btilde (b3)
  # diagnosed after each bhat compute
  ########################################

  X_diagz = spaces.R # b3
  Y_diagz = spaces.W # b3

  res_diagz((t,dt),(F,Φ,q,ω),b1,b3,l3,(b),(b0)) = (
    ∫( ( ((b3*b1 ))*l3 - (1.0/2.0)*( b0*b0 + b*b )*l3 )   )dΩ
  )

  jac_diagz((t,dt),(F,Φ,q,ω),b1,b3,db3,l3,(b)) = (
    ∫( (db3*b1)*l3   )dΩ
  )
  op_diagz = FEOperator(res_diagz,jac_diagz,X_diagz,Y_diagz)


  ########################################
  # Sovlers
  ########################################
  solvers = get_solvers(nls_tols,ranks)
  ode_solvers, IC_solvers = solvers
  nls,ls = ode_solvers
  _nls_cg,_nls_gmres,_cg = IC_solvers



  ########################################
  # Initial conditions
  ########################################
  a((u,h,B),(v,w,r)) = ∫( u⋅v + h*w + B*r)dΩ
  l((v,w,r)) = ∫( u(0.0)⋅v + h(0.0)*w + B(0.0)*r  )dΩ
  op = AffineFEOperator(a,l,X_prog(0.0),Y_prog(0.0))
  xh0 = solve(_cg,op)
  uh0, hh0, Bh0 = xh0


  _res_diagb(b,l) = res_diagb((0.0,0.0),xh0,xh0,(b),(l) ,hh0)
  _jac_diagb(b,db,l) = jac_diagb((0.0,0.0),xh0,xh0,(b),(db),(l),hh0)
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

  wh0 = b1h0

  _res_diagz((b3),(l3)) = res_diagz((0.0,0.0),yh0,b1h0,b3,l3,bh0,bh0)
  _jac_diagz((b3),(db3),(l3)) = jac_diagz((0.0,0.0),yh0,b1h0,b3,db3,l3,bh0)
  op_diagz0 = FEOperator(_res_diagz,_jac_diagz,X_diagz(0.0),Y_diagz)
  zh0 = solve(_nls_gmres,op_diagz0)


  ########################################
  # Casimirs
  ########################################
  casimirs0 = compute_casimirs(uh0,hh0,Bh0,bh0,zh0,Fh0,dΩ,dΛ,n_Λ,bh0,upwinding,ε,soft,hh0,Bh0,dt)

  ########################################
  # Operators
  ########################################

  op_tsw = TSWOperator(op_prog,op_diag,op_diagb,op_diagw,op_diagz,Tfunc,Tspace,bbarfunc,bbarspace)
  opT = TransientTSWOperator(op_tsw, const_jac=const_jac)

  t0 = 0.0
  ode_solver = TSWPoissonIntegrator(nls,(ls,ls),dt)

  sol_t  = solve(ode_solver,opT,t0,tF,xh0, bh0, yh0, wh0, zh0)

  i_am_main(ranks) && println("initial iterate...")
  it = iterate(sol_t)
  i_am_main(ranks) && println("...done")


  createpvd(ranks,out_dir*"/tsw_$simName") do pvd

    pvd[0] = createvtk(Ω,out_dir*"/tsw_$(simName)_0.vtu",
                        cellfields=["u"=>uh0, "h"=>hh0, "B" =>Bh0,
                        "F"=>Fh0, "Phi"=>Φh0,
                        "q"=>qh0, "b"=>bh0],append=false)


    push!(Cs,casimirs0)
    push!(ts,t0)
    it = iterate(sol_t)
    counter = 1

    while !isnothing(it)
      data, state = it
      t, xh = data
      _tF, stateF, state0, uF, odecache,
              bF,yF,wF,zF ,
              diagnosticsbF,diagnosticsF,diagnosticswF,diagnosticszF,
              diagnosticsb0, diagnostics0,diagonsticsw0,diagnosticsz0 = state[2]


      td = t/convert2seconds

      # initial condition to time step
      u0,h0,B0 = FEFunction(X_prog(t-dt),state0[1])
      F0,Φ0,q0 = FEFunction(X_diag(t-dt), diagnostics0[1])
      b0 = FEFunction(X_diagb(t-dt), diagnosticsb0[1])

      # extract solution at time t
      b00 = FEFunction(X_diagb(t), diagnosticsb0[1])
      xh = FEFunction(X_prog(t),stateF[1])
      uh,hh,Bh = xh
      Fh,Φh,qh = FEFunction(X_diag(t), diagnosticsF[1])
      bh = FEFunction(X_diagb(t), diagnosticsbF[1])
      wh = FEFunction(X_diagw(t), diagnosticswF[1])
      zh = FEFunction(X_diagz(t), diagnosticszF[1])

      b3h = zh

      casimirs = compute_casimirs(uh,hh,Bh,bh,b3h,Fh,dΩ,dΛ,n_Λ,b00,upwinding,ε,soft,h0,B0,dt)

      i_am_main(ranks) && println("t = ", t)

      i_am_main(ranks) && println("Normalised energy = ", (casimirs.E - casimirs0.E)/casimirs0.E)
      i_am_main(ranks) && println("Normalised entropy = ", (casimirs.S - casimirs0.S)/casimirs0.S)
      i_am_main(ranks) && println("dSdt = ", casimirs.Sinternal_btilde + casimirs.Sinternal2 + casimirs.Scenter + casimirs.Supwind)



      if mod(counter,out_freq) == 0

        push!(Cs,casimirs)
        push!(ts,t)

        pvd[td] = createvtk(Ω,out_dir*"/tsw_$(simName)_$t.vtu",
                        cellfields=["u"=>uh, "h"=>hh, "B" => Bh,
                                    "F"=>Fh, "Phi"=>Φh,
                                    "q"=>qh, "b"=> bh],append=false)

        output = @strdict ts Cs
        i_am_main(ranks) && safesave(datadir(out_dir, ("tsw_entropy.jld2")), output)
      end

      it = iterate(sol_t, state)
      counter = counter + 1

    end


  end

  results = @strdict ts Cs

  return results
end

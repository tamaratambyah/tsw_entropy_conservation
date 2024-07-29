function gradPerp(∇ϕ::VectorValue{2})
  """
  # ϕ     = scalar function of x,y
  # ∇ϕ    = ( ∂ϕ/∂x, ∂ϕ/∂y )
  # ∇^⟂ϕ  = ( -∂ϕ/∂y, ∂ϕ/∂x  ) """
  # xt = Point(1.0,1.0)
  # ϕ(x) = x[1] + 2*x[2]
  # qq = ∇(ϕ)(xt)
  # VectorValue(-qq[2],qq[1])
  # pts = get_cell_points(Ω)
  # gg = evaluate( gradPerp∘(∇(ϕ)), pts)

  v = VectorValue( -deepcopy(∇ϕ[2]), deepcopy(∇ϕ[1]))
  return v
end


function vecPerp(u)
  # u   = (u1, u2)
  # u^T = (-u2, u1)
  uT = VectorValue(-deepcopy(u[2]),deepcopy(u[1]))
  return uT
end


# function my_sign(Fn)
#   # including the factor 1/2 here
#   # stable for both (F⋅n_Λ).plus and (F⋅n_Λ).minus
#   #
#   # When evaluating (F⋅n_Λ).plus on Λ, return a vector VectorValue
#   # the enteries in x,y components seem to be equal, so just take the x component

#   c = 0.5

#   if Fn < 0.0
#     c = -0.5
#   end

#   return c
# end
function my_sign_og(Fn)
  # including the factor 1/2 here
  # stable for both (F⋅n_Λ).plus and (F⋅n_Λ).minus
  #
  # When evaluating (F⋅n_Λ).plus on Λ, return a vector VectorValue
  # the enteries in x,y components seem to be equal, so just take the x component

  c = 0.0

  if Fn < 0.0
    c = -0.5
  elseif Fn > 0.0
    c = 0.5
  end

  return c
end

function my_sign(Fn)
  # including the factor 1/2 here
  # stable for both (F⋅n_Λ).plus and (F⋅n_Λ).minus
  #
  # When evaluating (F⋅n_Λ).plus on Λ, return a vector VectorValue
  # the enteries in x,y components seem to be equal, so just take the x component

  c = 0.0

  if Fn < -1e-4
    c = -0.5
  elseif Fn > 1e-4
    c = 0.5
  end

  return c
end

function my_sign_jac(Fn)
  # including the factor 1/2 here
  # stable for both (F⋅n_Λ).plus and (F⋅n_Λ).minus
  #
  # When evaluating (F⋅n_Λ).plus on Λ, return a vector VectorValue
  # the enteries in x,y components seem to be equal, so just take the x component

  c = 0.5*Fn/(sqrt(Fn^2 + (1e-3)^2 ) )

  return c
end

function save_tsw_test(n,p,u,h,B,out_dir,ranks,nprocs,options,domain)


  i_am_main(ranks) && println("saving test set up")

  dir_meta = datadir(out_dir*"/meta")


  degree = 10*(p+1)
  partition = (n, n)
  model = CartesianDiscreteModel(ranks,nprocs,domain, partition,isperiodic=(true,true))

  Ω = Triangulation(model)
  dΩ = Measure(Ω, degree)
  Λ = SkeletonTriangulation(model)
  dΛ = Measure(Λ,degree)
  n_Λ = get_normal_vector(Λ)

  V = TestFESpace(model,
                    ReferenceFE(raviart_thomas,Float64,p),
                    conformity=:Hdiv)
  U = TransientTrialFESpace(V)

  W = TestFESpace(model,
                  ReferenceFE(lagrangian ,Float64,p),
                  conformity=:L2)
  R = TransientTrialFESpace(W)

  P = TestFESpace(model,
                  ReferenceFE(lagrangian,Float64,p+1),
                  conformity=:H1)
  H = TransientTrialFESpace(P)


  X_prog = TransientMultiFieldFESpace([U,R,R]) # u, h, B
  Y_prog = MultiFieldFESpace([V,W,W]) # u, h, B

  _cg = PETScLinearSolver(petsc_ls_from_options_c)

  a((u,h,B),(v,w,r)) = ∫( u⋅v + h*w + B*r)dΩ
  l((v,w,r)) = ∫( u(0.0)⋅v + h(0.0)*w + B(0.0)*r  )dΩ
  op = AffineFEOperator(a,l,X_prog(0.0),Y_prog(0.0))
  xh0 = solve(_cg,op)
  uh0, hh0, Bh0 = xh0

  #### save meta data
  psave(dir_meta, (xh0.metadata.free_values))

  ### plot
  writevtk(Ω,out_dir*"/plot_og/tsw_plot_og.vtu",
                    cellfields=["u"=>xh0[1], "h"=>xh0[2], "B"=>xh0[3] ])
  i_am_main(ranks) && println("Successfully finished save test")

end

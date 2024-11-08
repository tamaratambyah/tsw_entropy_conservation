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

function upwinding_sign(Fn,ε,soft)
  c = 0.0

  if Fn < -ε
    c = -0.5
  elseif Fn > ε
    c = 0.5
  end

  if soft
    c = 0.5*Fn/(sqrt(Fn^2 + (ε)^2 ) )
  end

  return c

end

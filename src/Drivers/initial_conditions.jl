
function get_testcase(testcase::Merging)
  println("merging vortex")
  ### Double vortex

  # H0 = 2#750
  # g = 1.0#9.80616
  # f = 1.0#0.00006147
  # l = 1.0#5000*1e3
  # dh = 0.025 #0.075
  # L = 1.0

  # in km
  L = 5000*1e3
  f = 0.00006147
  H0 = 750
  dh = 75
  g = 9.80616
  l = 5000*1e3


  sigmax = 3/40*l
  sigmay = 3/40*l
  ox = 0.1
  oy = 0.1
  m = 0.5
  simName = "vortex"

  xc1 = (0.5 - ox)*l
  xc2 = (0.5 + ox)*l
  yc1 = (0.5 - oy)*l
  yc2 = (0.5 + oy)*l

  x1(x) = l/(π*sigmax)*sin( π/l*(x[1] -xc1 )  )
  x2(x) = l/(π*sigmax)*sin( π/l*(x[1] -xc2 )  )
  x11(x) = l/(2*π*sigmax)*sin( 2*π/l*(x[1] -xc1 )  )
  x22(x) = l/(2*π*sigmax)*sin( 2*π/l*(x[1] -xc2 )  )

  y1(x) = l/(π*sigmay)*sin( π/l*(x[2] -yc1 )  )
  y2(x) = l/(π*sigmay)*sin( π/l*(x[2] -yc2 )  )
  y11(x) = l/(2*π*sigmay)*sin( 2*π/l*(x[2] -yc1 )  )
  y22(x) = l/(2*π*sigmay)*sin( 2*π/l*(x[2] -yc2 )  )


  h(t) = x -> H0 - dh*( exp( -m*( x1(x)^2 + y1(x)^2  )  )
                  +  exp(  -m*( x2(x)^2 + y2(x)^2 )  )
                  - 4*π*sigmax*sigmay/l^2 )
  u1(x) = -g*dh/(f*sigmay)*( y11(x)*exp( -m*( x1(x)^2 + y1(x)^2 ) )
                          + y22(x)*exp( -m*( x2(x)^2 + y2(x)^2 )  )  )
  u2(x) = g*dh/(f*sigmax)*(  x11(x)*exp( -m*( x1(x)^2 + y1(x)^2 ) )
                          + x22(x)*exp( -m*( x2(x)^2 + y2(x)^2 )  )  )
  u(t) = x -> VectorValue(u1(x), u2(x) )
  # b(x,t) = g*( 1 + 0.05*sin( 2*π/l*( x[1]-l/2 ) )  )
  B(t) = x -> ( g*(H0 - dh*( exp( -m*( x1(x)^2 + y1(x)^2  )  )
                +  exp(  -m*( x2(x)^2 + y2(x)^2 )  )
                - 4*π*sigmax*sigmay/l^2 ))
          + g*0.05*sin( 2*π/l*( x[1]-l/2 ) )*(H0 - dh*( exp( -m*( x1(x)^2 + y1(x)^2  )  )
                                              +  exp(  -m*( x2(x)^2 + y2(x)^2 )  )
                                              - 4*π*sigmax*sigmay/l^2 ))
  )

  convert2seconds = (3600*24)
  domain = (0.0, L, 0.0, L)
  return u, h, B, H0, g, f, L, simName,convert2seconds, domain
end

function get_testcase(testcase::Convergence)
  println("rescaled convergence test")
  # balanced test case
  a = 6371120
  _H0 = 5960
  _g = 9.80616 #
  c = 0.05
  _f =  6.147e-5
  u0 = 20.0

  Ro_Bu = (u0*_f*a)/(_g*_H0)
  # println(Ro_Bu)

  L = 2*π
  H0 = 1.0
  g = 1.0
  f = Ro_Bu

  simName = "rescaled convergence"
  u(t) = x -> VectorValue(cos(x[2] ) , 0.0 )
  h(t) = x -> H0 - Ro_Bu*sin(x[2])
  b(t) = x ->  1 + c*(1/h^2)
  ## B(t) = b*h = 1*(h) + c*(1/h)
  B(t) = x -> 1*(H0 - Ro_Bu*sin(x[2]))  + c/(H0 - Ro_Bu*sin(x[2]))

  convert2seconds = 1.0
  domain =  (0.0, L, 0.0, L)
  return u, h, B, H0, g, f, L, simName, convert2seconds, domain
end



# function get_testcase(testcase::Convergence)

#   # balanced test case
#   a = 6371120
#   H0 = 5960
#   g = 9.80616 #
#   c = 0.05
#   f = 6.147e-5
#   L = 2*π*a
#   u0 = 20.0
#   simName = "convergence"
#   u(t) = x -> VectorValue(u0*cos(x[2]/a), 0.0 )
#   h(t) = x -> H0 - a*f*u0/(g)*sin(x[2]/a)
#   ## b(t) = x -> g*( 1 + c*H0^2/h(x,t)^2  )
#   ## B(t) = b*h = g*h + cH0^2/h
#   B(t) = x -> g*(H0 - a*f*u0/(g)*sin(x[2]/a))  + c*H0^2/(H0 - a*f*u0/(g)*sin(x[2]/a))

#   convert2seconds = (3600*24)
#   domain =  (0.0, L, 0.0, L)
#   return u, h, B, H0, g, f, L, simName, convert2seconds, domain
# end

r(x) = sqrt((x[1])^2 + (x[2])^2)

function get_testcase(testcase::Instability)
  simName = "intstability"
  β = 2.0
  L = 8.0
  g = 1.0
  H0 = 1.0
  f = 1.0
  s0 = 1.0
  U = 0.1
  Ro = 0.1
  Bu = 1.0
  l = 4.0
  rc = 0.5


  sf(x) = -1.0*exp( -60*(sqrt(x[1]^2 + x[2]^2)-rc)^2 )*sin( 6*π*(sqrt(x[1]^2 + x[2]^2)-rc)  )
  db(x) = -0.01*sf(x)*cos(l*(atan(x[2],x[1])))
  dh(x) = 0.01*sf(x)*cos(l*(atan(x[2],x[1])))
  du(x) = -0.01*sf(x)*cos(l*(atan(x[2],x[1])))
  dv(x) = -0.01*sf(x)*cos(l*(atan(x[2],x[1])))
  dB(x) = db(x)*dh(x)

  u1(x) = -1.0*U*r(x)*exp( (1.0 - (r(x))^β )/β )*sin(atan(x[2],x[1])) + du(x)
  u2(x) = U*r(x)*exp( (1.0 - (r(x))^β )/β )*cos(atan(x[2],x[1])) + dv(x)

  u(t) = x -> VectorValue(u1(x), u2(x))
  h(t) = x -> H0 + dh(x)
  # b = s0 - 2.0*s0*Ro/Bu*( exp( (1 - (r(x))^2 )/2 ) + Ro*0.5*exp( 1 - (r(x))^2  )   )
  B(t) = x -> (s0*H0 - H0*2.0*s0*Ro/Bu*( exp( (1 - (r(x))^2 )/2 ) + Ro*0.5*exp( 1 - (r(x))^2  )   ) + H0*db(x)
             + s0*dh(x) - dh(x)*2.0*s0*Ro/Bu*( exp( (1 - (r(x))^2 )/2 ) + Ro*0.5*exp( 1 - (r(x))^2  )   ) + dh(x)*db(x)
                )

  convert2seconds = 1.0
  domain = (-L/2, L/2, -L/2, L/2)
  return u, h, B, H0, g, f, L, simName, convert2seconds, domain
end

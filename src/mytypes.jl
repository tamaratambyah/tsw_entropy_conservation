abstract type tswconvergence end
struct Convergence <: tswconvergence end

const convergence = Convergence()

abstract type tswtestcase  end

struct Vortex <: tswtestcase end
struct Instability <: tswtestcase end

const vortex = Vortex()
const instability = Instability()

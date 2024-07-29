abstract type tswconvergence end
struct Convergence <: tswconvergence end

const convergence = Convergence()

abstract type tswconvergencerestarted end
struct ConvergenceRestarted <: tswconvergencerestarted end

const convergencerestarted = ConvergenceRestarted()


abstract type tswtestcase  end

struct Merging <: tswtestcase end
struct Instability <: tswtestcase end

const merging = Merging()
const instability = Instability()

abstract type tswrestart  end

struct Restarted <: tswrestart end
const restarted = Restarted()

# Import the package
import Optimization as OPT
import OptimizationNLopt
import ForwardDiff

# Define the problem to optimize
L(u, p) = (p[1] - u[1])^2 + p[2] * (u[2] - u[1]^2)^2
u0 = zeros(2)
p = [1.0, 100.0]
optfun = OPT.OptimizationFunction(L, OPT.AutoForwardDiff())
prob = OPT.OptimizationProblem(optfun, u0, p, lb = [-1.0, -1.0], ub = [1.0, 1.0])

# Solve the optimization problem
sol = OPT.solve(prob, OptimizationNLopt.NLopt.LD_LBFGS())

# Analyze the solution
@show sol.u, L(sol.u, p)
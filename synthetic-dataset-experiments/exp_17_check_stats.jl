
import AdvancedHMC
import LogDensityProblems
import LinearAlgebra

struct TestProblem end
LogDensityProblems.logdensity(p::TestProblem, θ) = -sum(abs2, θ) / 2
LogDensityProblems.dimension(p::TestProblem) = 10
LogDensityProblems.capabilities(::Type{TestProblem}) = LogDensityProblems.LogDensityOrder{0}()

p = TestProblem()
d = 10
initial_theta = rand(d)
metric = AdvancedHMC.DiagEuclideanMetric(d)
h = AdvancedHMC.Hamiltonian(metric, LogDensityProblems.logdensity, LogDensityProblems.ADgradient(:ForwardDiff, p))
integrator = AdvancedHMC.Leapfrog(0.1)
kernel = AdvancedHMC.HMCKernel(AdvancedHMC.Trajectory{AdvancedHMC.MultinomialTS}(integrator, AdvancedHMC.GeneralisedNoUTurn()))
adaptor = AdvancedHMC.StanHMCAdaptor(AdvancedHMC.MassMatrixAdaptor(metric), AdvancedHMC.StepSizeAdaptor(0.80, integrator))

samples, stats = AdvancedHMC.sample(h, kernel, initial_theta, 10, adaptor, 10; progress=false)

println("Stats type: ", typeof(stats[1]))
println("Stats keys: ", propertynames(stats[1]))

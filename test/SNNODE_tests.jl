using .NeuralPDE2
using Flux
using Random
import OptimizationOptimisers

Random.seed!(100)

# Deterministic equation
linear = (u, p, t) -> cos(2pi * t)
null_f = (u, p, t) -> 0.0f0
tspan = (0.0f0, 1.0f0)
u0 = 0.0f0
prob = SDEProblem(linear, null_f, u0, tspan, p=)
chain = Flux.Chain(Dense(1, 5, σ), Dense(5, 1))
# luxchain = Lux.Chain(Lux.Dense(1, 5, Lux.σ), Lux.Dense(5, 1))
opt = OptimizationOptimisers.Adam(0.1, (0.9, 0.95))

sol = solve(prob, NNODE(chain, opt), dt = 1 / 20.0f0, verbose = true,
            abstol = 1.0f-10, maxiters = 200)
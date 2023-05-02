using .NeuralPDE2
using Flux
using Random
import OptimizationOptimisers
using Plots

# Random.seed!(100)

# # Deterministic equation
# linear = (u, p, t) -> cos(2pi * t)
# null_f = (u, p, t) -> 0.0f0
# tspan = (0.0f0, 1.0f0)
# u0 = 0.0f0
# prob = SDEProblem(linear, null_f, u0, tspan)
# chain = Flux.Chain(Dense(1, 5, σ), Dense(5, 1))
# # luxchain = Lux.Chain(Lux.Dense(1, 5, Lux.σ), Lux.Dense(5, 1))
# opt = OptimizationOptimisers.Adam(0.1, (0.9, 0.95))

# sol = solve(prob, NNODE(chain, opt), dt = 1 / 20.0f0, verbose = true,
#             abstol = 1.0f-10, maxiters = 200)

# # analytic solution
# u(t) = 1/(2pi) * sin(2pi * t)
# t = range(0, 1, length = 100)

# # plot solution
# using Plots

# plot(sol, vars = (0, 1), label = "NNODE")
# plot!(t, u.(t), label = "analytic")


# Stochastic equation - black scholes
mu = 1.5
sigma = 2
f = (u, p, t) -> mu * u
g = (u, p, t) -> sigma * u
tspan = (0.0f0, 1.0f0)
u0 = 1.0f0
prob = SDEProblem(f, g, u0, tspan)
chain = Flux.Chain(Dense(1, 25, σ), Dense(25, 75, σ), Dense(75, 150, σ), Dense(150, 1))
# luxchain = Lux.Chain(Lux.Dense(1, 5, Lux.σ), Lux.Dense(5, 1))
opt = OptimizationOptimisers.Adam(0.01, (0.9, 0.95))

sol = solve(prob, NNODE(chain, opt, autodiff=false), dt = 1 / 20.0f0, verbose = true,
            abstol = 1.0f-10, maxiters = 10000)

# analytic solution
dt=1/100.0f0
u(t) = exp((mu - 0.5 * sigma ^ 2) * t + sigma * sqrt(dt) * randn())
t = range(0, 1, length = 100)

# plot solution


plot(sol, vars = (0, 1), label = "NNODE")
plot!(t, u.(t), label = "analytic")
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


# # Stochastic equation - black scholes
# mu = 3
# sigma = 1/2

# t_length=500
# t = range(0, 1, length = t_length)
# dt=1.0f0/t_length

# f = (u, p, t) -> mu * u
# g = (u, p, t) -> sigma * u
# tspan = (0.0f0, 1.0f0)
# u0 = 0.5f0
# prob = SDEProblem(f, g, u0, tspan)
# # chain = Flux.Chain(Dense(1, 25, σ), Dense(25, 75, σ), Dense(75, 150, σ), Dense(150, 1))
# chain = Chain(
#     Dense(1, 25, σ),
#     Dropout(0.2),
#     Dense(25, 75, σ),
#     Dropout(0.2),
#     Dense(75, 150, σ),
#     Dropout(0.2),
#     Dense(150, 1)
# )
# # luxchain = Lux.Chain(Lux.Dense(1, 5, Lux.σ), Lux.Dense(5, 1))
# opt = OptimizationOptimisers.Adam(0.5, (0.9, 0.95))

# sol = solve(prob, NNODE(chain, opt, autodiff=false), dt = dt, verbose = true,
#             abstol = 1.0f-3, maxiters = 5000)

# # analytic solution
# u(t) = 0.5*exp((mu - 0.5 * sigma ^ 2) * t + sigma * sqrt(dt) * randn())

# # plot solution
# plot(t, u.(t), label = "analytic")
# plot!(sol, vars = (0, 1), label = "NNODE")

# Stochastic equation 2 - example 2 
# alpha = 1/10
# beta = 1/20

# t_length=500
# t = range(0, 1, length = t_length)
# dt=1.0f0/t_length

# f = (u, p, t) -> -(alpha^2)*sin(u)*(cos(u))^3
# g = (u, p, t) -> alpha*(cos(u))^2
# tspan = (0.0f0, 1.0f0)
# u0 = 0.5f0
# prob = SDEProblem(f, g, u0, tspan)
# # chain = Flux.Chain(Dense(1, 25, σ), Dense(25, 75, σ), Dense(75, 150, σ), Dense(150, 1))
# chain = Chain(
#     Dense(1, 25, σ),
#     Dropout(0.2),
#     Dense(25, 75, σ),
#     Dropout(0.2),
#     Dense(75, 1)
# )
# # luxchain = Lux.Chain(Lux.Dense(1, 5, Lux.σ), Lux.Dense(5, 1))
# opt = OptimizationOptimisers.Adam(0.00005, (0.9, 0.95))

# nnode = NNODE(chain, opt, autodiff=false)

# sol = solve(prob, nnode, dt = dt, verbose = true,
#             abstol = 1.0f-3, maxiters = 20000)

# # analytic solution
# u(t) = atan(alpha*sqrt(dt)*randn()+tan(u0))

# # plot solution
# plot(t, u.(t), label = "", color = "green", alpha=0.05)

# for i in 1:100
#     u(t) = atan(alpha*sqrt(dt)*randn()+tan(u0))
#     plot!(t, u.(t), label = "", color = "green", alpha = 0.05)
# end
# plot!(sol, vars = (0, 1), label = "NNODE", color="red")


# # Stochastic equation 2 - example 3
# alpha = 1/10
# beta = 1/20

# t_length=100
# t = range(0, 1, length = t_length)
# dt=1.0f0/t_length

# f = (u, p, t) -> (beta/sqrt(1+t)-u/(2*(1+t)))
# g = (u, p, t) -> alpha*beta/sqrt(1+t)
# tspan = (0.0f0, 1.0f0)
# u0 = 0.5f0
# prob = SDEProblem(f, g, u0, tspan)
# # chain = Flux.Chain(Dense(1, 25, σ), Dense(25, 75, σ), Dense(75, 150, σ), Dense(150, 1))
# chain = Chain(
#     Dense(1, 25, σ),
#     Dropout(0.2),
#     Dense(25, 75, σ),
#     Dropout(0.2),
#     Dense(75, 150, σ),
#     Dropout(0.2),
#     Dense(150, 1)
# )
# # luxchain = Lux.Chain(Lux.Dense(1, 5, Lux.σ), Lux.Dense(5, 1))
# opt = OptimizationOptimisers.Adam(0.005, (0.9, 0.95))

# sol = solve(prob, NNODE(chain, opt, autodiff=false), dt = dt, verbose = true,
#             abstol = 1.0f-3, maxiters = 5000)

# # analytic solution
# u(t) = u0/(sqrt(1+t))+beta*(t+alpha*sqrt(dt)*randn())/sqrt(1+t)

# # plot solution
# plot(t, u.(t), label = "", color = "green", alpha=0.05)

# for i in 1:100
#     u(t) = u0/(sqrt(1+t))+beta*(t+alpha*sqrt(dt)*randn())/sqrt(1+t)
#     plot!(t, u.(t), label = "", color = "green", alpha = 0.05)
# end
# plot!()
# plot!(sol, vars = (0, 1), label = "NNODE", color="red")


# plot(sol, idxs = (0, 1))

# Stochastic equation - Ornstein-Uhlenbeck Process
theta = 2.5
mu = 1/20
sigma = 1/20

t_length=25
t = range(0, 1, length = t_length)
dt=1.0f0/t_length

f = (u, p, t) -> theta*(mu-u)
g = (u, p, t) ->  sigma
tspan = (0.0f0, 1.0f0)
u0 = 0.5f0
prob = SDEProblem(f, g, u0, tspan)
# chain = Flux.Chain(Dense(1, 25, σ), Dense(25, 75, σ), Dense(75, 150, σ), Dense(150, 1))
chain = Chain(
    Dense(1, 25, σ),
    # Dropout(0.2),
    # Dense(25, 75, σ),
    # Dropout(0.2),
    Dense(25, 1)
)
opt = OptimizationOptimisers.Adam(0.001, (0.9, 0.95))
# try other optimizers
# opt = OptimizationOptimisers.RMSProp(0.0005)

nnode = NNODE(chain, opt, autodiff=false)

sol = solve(prob, nnode, dt = dt, verbose = true,
            abstol = 0.1, maxiters = 5000)

# analytic solution
# u(t) = mu+(u0-mu)*exp(-theta*t)+sigma*sqrt(1-exp(-2*theta*t))*randn()
u(t) = u0 * exp(-theta * t) + mu * (1 - exp.(-theta * t)) + sigma * exp(-theta * t) * randn()


# plot solution
plot(t, u.(t), label = "", color = "green", alpha=0.1)

for i in 1:100
    plot!(t, u.(t), label = "", color = "green", alpha = 0.1)
end
plot!(sol, vars = (0, 1), label = "NNODE", color="red")
# save the plot
savefig("test_sdes/ou_fit2.png")
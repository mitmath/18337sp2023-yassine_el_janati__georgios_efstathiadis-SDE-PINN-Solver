using .NeuralPDE2
using Flux
using Random
import OptimizationOptimisers
using Plots


# Stochastic equation - black scholes
alpha = 0.1
beta = 0.2

t_length=100
t = range(0, 1, length = t_length)
dt=1.0f0/t_length

f = (u, p, t) -> alpha * u
g = (u, p, t) -> beta * u
tspan = (0.0f0, 1.0f0)
u0 = 0.5f0

n_terms = 4

prob = SDEProblem(f, g, u0, tspan, [n_terms])
chain = Chain(
    Dense(1, 25, σ),
    Dense(25, 75, σ),
    Dense(75, 150, σ),
    Dense(150, 1)
)
opt = OptimizationOptimisers.Adam(0.0001, (0.9, 0.95))

nnode = NNODE(chain, opt, autodiff=false)

sol = solve(prob, nnode, dt = dt, verbose = true,
            abstol = 1.0f-3, maxiters = 2000)

# analytic solution
u(t) = u0*exp((beta - 0.5 * alpha ^ 2) * t + alpha * sqrt(dt) * randn())

u_values = [u.(t) for _ in 1:100];

# plot solution
plot(t, u_values[1], label = "", color = "green", alpha=0.1)

for i in 2:100
    plot!(t, u_values[i], label = "", color = "green", alpha = 0.1)
end

# plot average solution
mean_u = sum(u_values) / length(u_values)
plot!(t, fill(mean_u, length(t)), label = "", color="green")

plot!(sol, vars = (0, 1), label = "NNODE", color="red")

savefig("test_sdes\\example1.png")

# calculate difference between analytic and neural solution
diff = abs.(mean_u .- sol.(t))
diff

# calculate RMSE
using Statistics
rmse = sqrt(mean(diff .^ 2))
rmse

# R^2
R2 = 1 - sum(diff .^ 2) / sum((mean_u .- mean(mean_u)) .^ 2)
R2
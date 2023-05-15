using .NeuralPDE2
using Flux
using Random
import OptimizationOptimisers
using Plots


# Stochastic equation - example 2
theta = 1/10

t_length=25
t = range(0, 1, length = t_length)
dt=1.0f0/t_length

f = (u, p, t) -> -theta^2 * sin(u) * cos(u)^3 
g = (u, p, t) ->  theta * cos(u)^2
tspan = (0.0f0, 1.0f0)
u0 = 0.5f0

n_terms = 4

prob = SDEProblem(f, g, u0, tspan, [n_terms])
chain = Chain(
    Dense(1, 25, relu),
    Dense(25, 1, σ)
)
opt = OptimizationOptimisers.Adam(0.001, (0.9, 0.95))

nnode = NNODE(chain, opt, autodiff=false)

sol = solve(prob, nnode, dt = dt, verbose = true,
            abstol = 1.0f-4, maxiters = 5000)

# analytic solution
u(t) = atan(theta*randn() + tan(u0))

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

savefig("test_sdes\\example2.png")

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
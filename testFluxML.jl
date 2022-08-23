using DifferentialEquations,Flux, DiffEqFlux,  Plots

function lotka_volterra(du,u,p,t)
    x, y = u
    α, β, δ, γ = p
    du[1] = dx = α*x - β*x*y
    du[2] = dy = -δ*y + γ*x*y
  end
  u0 = [1.0,1.0]
  tspan = (0.0,10.0)
  p = [1.5,1.0,3.0,1.0]
  prob = ODEProblem(lotka_volterra,u0,tspan,p)



u0=[1.0,1.0]
# p = [1.91, 1.91, 1.25, 1.3]
p = [1.5,1.0,3.0,1.0]
prob = ODEProblem(lotka_volterra,u0,tspan,p)
sol = solve(prob,Tsit5(),saveat=0.1)
# x_tuzi = sol[1,:] # length 101 vector
# y_lang = sol[2,:]
plot(sol)
savefig("lotka1.pdf")


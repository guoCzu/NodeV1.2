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



# p = [1.5,1.2,3.0,1.3]
# prob = ODEProblem(lotka_volterra,u0,tspan,p)
sol = solve(prob,Tsit5(),saveat=0.1)
A = sol[1,:] # length 101 vector


# p = [2.2, 1.0, 2.0, 0.4] # 初始參數向量
params = Flux.params(p)

function predict_rd() # 我們的單層神經網路
  solve(prob,Tsit5(),p=p,saveat=0.1)[1,:] # [1,:] 返回的是x的值，即兔子的数量
#   return  a131
end

# a121 = predict_rd()
# println("a121= ",a121)
# println("size of a121= ",size(a121))

loss_rd() = sum(abs2,x-1 for x in predict_rd()) # 損失函數


data = Iterators.repeated((), 100)
opt = ADAM(0.1)
cb = function () # 用 callback function 來觀察訓練情況
  display(loss_rd())
  # 利用 `remake` 來再造我們的 `prob` 並放入目前的參數 `p`
  display(plot(solve(remake(prob,p=p),Tsit5(),saveat=0.1),ylim=(0.91275,1.041)))
end

# 顯示初始參數的微分方程
cb()

Flux.train!(loss_rd, params, data, opt, cb = cb)

println("params= ",params)
# 训练完成后， param始终的四个参数的值为：[1.9091575347771435, 1.9052182143342058, 1.2470867951611126, 1.3046869450726664]


# 用该组参数重新求解微分方程

# u0=[1.2,1.3]
# # p = [1.9091575347771435, 1.9052182143342058, 1.2470867951611126, 1.3046869450726664]
# prob = ODEProblem(lotka_volterra,u0,tspan,p)
# sol = solve(prob,Tsit5(),saveat=0.1)
# # x_tuzi = sol[1,:] # length 101 vector
# # y_lang = sol[2,:]
# plot(sol)

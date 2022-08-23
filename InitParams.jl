include("MyConstants.jl")
include("Forward.jl")
using Distributions

function InitParams()
    Nx = MyConst.Nx # x 的点数

    # kL=[1,3,5,3,1] # 第一个和最后一个是输入和输出层神经元个数，中间3,5,3是隐藏层神经元个数。
    kL = MyConst.kL
    nL = length(kL) # 神经网络总层数
    ###
    # n1 = Normal(0,1/sqrt(H1)) # μ = 0.0 σ=1.0
    w = Array{Any}(undef,nL-1)
    b = Array{Any}(undef,nL)
    x = Array{Any}(undef,nL)
    h = Array{Any}(undef,nL)
    ###
    # err_w = Array{Any}(undef,nL)
    # err_b = Array{Any}(undef,nL)
    ###
    # tp_w = typeof(zeros(Float32,kL[1],kL[2]))
    # w[1] = zeros(Float32,kL[1],kL[2])
    # tp_b = typeof(zeros(Float32,kL[1]))
    b[1] = zeros(Float32,kL[1])
    # tp_h = typeof(zeros(Float32,kL[1]))
    h[1] = zeros(Float32,kL[1])
    # tp_x = typeof(zeros(Float32,kL[1]))
    x[1] = zeros(Float32,kL[1])
    
    ###
    for l=2:nL  # 层指标，出掉了输入层和输出层
        # tp_w = typeof(zeros(Float32,kL[l-1],kL[l]))
        w[l-1] = zeros(Float32,kL[l-1],kL[l])
        # tp_b = typeof(zeros(Float32,kL[l]))
        b[l] = zeros(Float32,kL[l])
        # tp_h = typeof(zeros(Float32,kL[l]))
        h[l] = zeros(Float32,kL[l])
        # tp_x = typeof(zeros(Float32,kL[l]))
        x[l] = zeros(Float32,kL[l])

    end
    ### 初始化
    for l=1:nL-1
        distr = Normal(0,1/sqrt(kL[l+1]))
        w[l] = rand(distr,(kL[l],kL[l+1])) 
        b[l+1] = randn(kL[l+1])
    end

        # bH1 = randn(H1,1)
        # # weights # w_H = normrnd(0,1/sqrt(H),[H,1]);
        # n1 = Normal(0,1/sqrt(H1)) # μ = 0.0 σ=1.0
        # wH1 = rand(n1,(H1,1)) #(1,H)'
        # ###
        # bH2 = randn((H2,1))
        # n2 = Normal(0,1/sqrt(H1)) # μ = 0.0 σ=1.0
        # wH2 = rand(n2,(H2,H1)) #(1,H)'
        # ###
        # bH3 = randn(H3,1)
        # n3 = Normal(0,1/sqrt(H2)) # μ = 0.0 σ=1.0
        # wH3 = rand(n3,(H3,H2)) #(1,H)'

    # end

    # h,x =   feedForward(w,b,h,x,1);
    return  w,b,h,x 
end



# w,b,h,x  = InitParams()
# Nx = MyConst.Nx
# # h,x = 0.0,0.0
# for ix = 1:1
#     h,x =   feedForward(w,b,h,x,ix);
# end
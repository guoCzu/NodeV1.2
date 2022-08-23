include("Forward.jl")
include("Backpropagate.jl")
include("MyConstants.jl")
include("Funcs.jl")
using Plots
using Distributions
using Dates
# Equation Parameters
# y' = f(y,x)
# -----------
# 
# -----------
# clear everything
function Train(f,a0,df_dy,C_1,w,b,h,x)
    
    nBP = MyConst.nBP

    # Number of Training Points
    Nx = MyConst.Nx;
    aOut = zeros(Nx)

    # Network Parameters
    # intial learning rate
    eta = MyConst.eta
    # drop rate
    droprate = MyConst.droprate
    # # hidden layer 神经数。
    xIn = x[1]
    # feedforward over batches
    for ix = 1:Nx
        h,x =   feedForward(w,b,h,x,ix);
    end
    # return 0
    ############################## 
    # # backpropagation algorithm
    for ibp = 1:nBP
        w,b,h,x = backPropagate(w,b,h,x,Nx,f,a0,df_dy,C_1,eta,droprate,ibp);
        
    end
    # feedforward over training inputs
    for ix = 1:Nx
        h,x = feedForward(w,b,h,x,ix);
        aOut[ix] = x[end][1]
    end

    return aOut
end

#     # Plot Actual vs. ANN Solution       
#     p0 = plot(1)
#     p2 = plot!(p0,x,y(x),color="red",label = "analysis solution",legend=true)
#     p2 = plot!(p2,x,C_1 .+ x.*a_out,color="black",label = "neural solution",legend=true)
#     # savefig("curve.png")
#     xlabel!("x")
#     ylabel!("y")
#     display(p2)
#     savefig("p2.png")
#     title!("Exact vs. ANN-computed solution to y' = y")
#     # exit()
#     # legend!("Exact","ANN","location","northwest")
# # ######################################
# #     # Error Plot
#     n_err = N;
#     # sample
#     x_err = Array{Float64}(undef,N)
#     x_err .= x
#     # x_err = 0:1/n_err:1 -1/n_err
#     # x_err = linspace(0,1,n_err)";
#     a_out_err = zeros(n_err,1);
#     # feedforward over error-evaluating inputs
#     for i = 1:n_err
#         a_H,z_H,a_out_err[i],z_out = feedForward(w_H,b_H,w_out,x_err[i]);
#     end
#     # get errors
#     err = abs.(y(x_err) .- (C_1 .+ x_err.*a_out_err));

#     p2= plot!(p2,x_err,err,color=:green)
#     xlabel!("x")
#     ylabel!("error")

#     title!("Absolute Error of ANN-computed solution to y"*" = y")
##################################################################
# #     # Extrapolation Plot
#     m = 2* N;
#     # ex = Array{Float64}(undef,N)
#     # ex .= x
#     # ex = linspace(0,10,N)';
#     ex = range(1, 10, length=m)
#     a_out = zeros(m,1);
#     # feedforward over extrapolation points
#     for i = 1:m
#         a_H,z_H,a_out[i],z_out = feedForward(w_H,b_H,w_out,ex[i]);
#     end

#     # plot!(ex,y(ex))
#     p2 = plot!(p2,ex,IC .+ ex.*a_out)
#     xlabel!("x")
#     ylabel!("y")

#     title!("Extrapolation of ANN-computed solution to y' = y")
#     # legend("Exact","ANN","location","northwest")
# ######################################
    # return w_H
# end


##############################

# println("开始运行。 ",now())
# main()
# println("结束运行。 ",now())
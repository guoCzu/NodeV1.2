
function ToPlot(C_1, x,y,a_out)
    
    # Plot Actual vs. ANN Solution       
    p0 = plot(1,legend=false)
    p2 = plot!(p0,x,y(x),color="red",label = "analysis solution",legend=true)
    p2 = plot!(p2,x,C_1.(x) .+ x.*a_out,color="black",label = "neural solution",legend=true)
    # savefig("curve.png")
    xlabel!("x")
    ylabel!("y")
    display(p2)
    savefig("p2.png")
    title!("Exact vs. ANN-computed solution to y' = y")
    # exit()
    # legend!("Exact","ANN","location","northwest")
# ######################################
#     # Error Plot
    # n_err = MyConst.Nx
    # # sample
    # x_err = Array{Float64}(undef,n_err)
    # x_err .= x
    # # x_err = 0:1/n_err:1 -1/n_err
    # # x_err = linspace(0,1,n_err)";
    # a_out_err = zeros(n_err,1);
    # feedforward over error-evaluating inputs
    # for i = 1:n_err
    #     a_H,z_H,a_out_err[i],z_out = feedForward(w_H,b_H,w_out,x_err[i]);
    # end
    # get errors
    # err = abs.(y(x_err) .- (C_1 .+ x_err.*a_out_err));

    # p2= plot!(p2,x_err,err,color=:green)
    # xlabel!("x")
    # ylabel!("error")

    # title!("Absolute Error of ANN-computed solution to y"*" = y")

    
end
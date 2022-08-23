

function InitX(x)
    # Training Points
    Nx = MyConst.Nx 

    xIn= Array{Float64}(undef,Nx)
    dx = 1.0/Nx
    for j = 1:Nx
        xIn[j] = (j-1) * dx
    end
    x[1] = xIn
    return x
    
end
########################
function InitValues() # 微分方程本身的初始条件
    # Initial Condition
    # IC = -1.0;
    C_1 = -1.0 

    return C_1
    
end
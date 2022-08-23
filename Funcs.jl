# 微分方程本身所用的数学函数，

function Funcs()

    f(x,y) = sin(x) ;
    # f(x,y) = x * sin(x)
    
    a0(x) = -1.0 ; b(x) = 0.0 #4.0 
    # partial of function wrt y
    df_dy(x,y) =  0.0;
    # exact solution!!
    y(x) =  -cos.(x) ;  # 1./(1+exp(-x));

    return f,a0,df_dy,y
    
end
include("Sig.jl")
function feedForward(w,b,h,x,ix);
    # nHidLayers = length(bHs) # 隐藏层个数
    # zHs = Array{typeof(bHs[1])}(undef,(nHidLayers,1))
    # aHs = Array{typeof(bHs[1])}(undef,(nHidLayers,1))
    # for i =1:nHidLayers
    #     zHs[i] = zeros(length(bHs[i]),1)
    #     aHs[i] = zeros(length(bHs[i]),1)
    # end
    # 第一层传输
    # zHs[1] = wHs[1]*x0 .+ bHs[1]
    # aHs[1] = sig.(zHs[1])
    kL = MyConst.kL
    nL = length(kL)
    
    for l =2:nL
        for kNow=1:kL[l]
            # h[l][kPre] = sum(w)
            temp1 = 0.0 
            for kPre=1:kL[l-1]
                # println("l, kNow, kPre= ",l, " ",kNow , " ",kPre)
                if l==2 

                    temp1 = temp1 + w[l-1][kPre,kNow]*x[l-1][ix]
                else 
                    temp1 = temp1 + w[l-1][kPre,kNow]*x[l-1][kPre]
                end
                
            end
            h[l][kNow] =  temp1 + b[l][kNow]
            if l<nL 
                x[l][kNow] = sig(h[l][kNow])
            else 
                x[l][kNow] = h[l][kNow]
            end
            
        end
        ###
        # if l<nL && l>=2
        #     x[l] = sig.(h[l])
        # elseif l==nL 
        #     x[l] = h[l]
        # end
    end
    ###
    # z_out = w_out' * aHs[nHidLayers] ;
    a_out = x[nL][1]
    # # weighted inputs to hidden layer

    
    return h,x
end
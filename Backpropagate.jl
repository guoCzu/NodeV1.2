include("Forward.jl")
include("Sig.jl")
include("Dsig.jl")


function backPropagate(w,b,h,x,Nx,f,A_0,df_dy,C_1,eta,droprate,i)
    kL = MyConst.kL 
    Nx = MyConst.Nx
    nL = length(kL)
    ###
    # grad = 1.0 #暂时先记为1.0 
    ###
    δb = deepcopy(b)
    δw = deepcopy(w)
    δx = deepcopy(x)
    ###
    xIn = x[1]
    # trial solution derivatives
    dytrial_dx = 0;
    ###
    for ix = 1:Nx # x 的100个点。
        # feedforward 
        # xIn = x[1][ix]
        h,x = feedForward(w,b,h,x,ix);
        # trial solutions
        # aO 即为文献中的 Ψ(xi)=a_out =aO
        aO = x[end][1]
        y_trial = A_0(x[1][ix]) + x[1][ix] * aO  
        # trial solution derivative
        # daO_dx = x[end][1]/x[1][ix]
        dytrial_dx = 2.0* aO #+ xIn[i] * daO_dx  ;
        grad = 2.0*(dytrial_dx - f(x[1][ix],y_trial))
        
        
        for l=nL:-1:2
            if l==nL 
                for kNow=1:kL[l]
                    δx[l][kNow] = grad * 1.0 
                    δb[l][kNow] = grad * dsig(h[l][kNow]) # 本来是1.0 如果没有 输出层的偏置，则为0.0  
                    b[l][kNow] = (b[l][kNow]  - eta* δb[l][kNow]) * 0.0  # 设置为0
                end
            elseif l<nL && l>=2
                for kNow=1:kL[l] 
                    temp1 = 0.0 
                    for kNext=1:kL[l+1]
                        temp1 = temp1 + δx[l+1][kNext] * dsig(h[l+1][kNext]) * w[l][kNow,kNext]
                    end
                    δx[l][kNow] =  temp1
                    δb[l][kNow] = δx[l][kNow] * dsig(h[l][kNow])
                    b[l][kNow] = b[l][kNow]  - eta* δb[l][kNow]
                    for kNext=1:kL[l+1] 
                        δw[l][kNow,kNext] = δx[l+1][kNext] * dsig(h[l+1][kNext])*x[l][kNow]
                        w[l][kNow,kNext] = w[l][kNow,kNext]  - eta* δw[l][kNow,kNext]
                    end
                end
            elseif l==1
            end
            ############# 更新 参数 #############


        end
        ############################
        
    end


    return w,b,h,x
end








########################################################################
# function backPropagateBak(w,b,h,x,Nx,f,a,b,df_dy,C_1,eta,droprate,i)
#     # (wHs, bHs, wO,n,x,f,a,b,df_dy,C_1,eta,droprate,epoch)
    
#     kL = MyConst.kL 
    
#     ###
#     # trial solutions
#     y_trial = 0;
    
#     # trial solution derivatives
#     dytrial_dx = 0;
    
#     # feedforward
#     # activations
#     # a_H = zeros(H,1);
#     # # weighted inputs
#     # z_H = zeros(H,1);
#     # activations
#     aO = 0; #  a_out
#     # weighted input
#     zO = 0;
    
#     # output layer error
#     # err_out = 0;
#     # hidden layer error
#     err_wH3 = zeros(H3,H2);
#     err_wH2 = zeros(H2,H1);
#     err_wH1 = zeros(H1,1);
#     err_bH3 = zeros(H3,1)
#     err_bH2 = zeros(H2,1)
#     err_bH1 = zeros(H1,1)
#     err_wO = zeros(H3,1)
    
#     # # grad of output layer weights
#     # dwO = zeros(H3,1);
#     # # grad of hidden layer weights
#     # dwH3 = zeros(H3,H2);
#     # dwH2 = zeros(H2,H1);
#     # dwH1 = zeros(H1);
#     # # grad of hidden layer biases
#     # dbH3 = zeros(H3,1);
#     # dbH2 = zeros(H2,1);
#     # dbH1 = zeros(H1,1);

#     # gradient of Network wrt error

#     # Cost Function Gradient
#     # E1_0(x,a_out,da_out_dx,d2a_out_dx2,y_trial) = a(x)*B + b(x)*A + x* b(x) *B # A,B 是初始值

#     # grad_C1(dytrial_dx,y_trial,x) = 1-pf(x)*x
#     ###
#     pSzwO = size(wO)
#     daO_dwO = Array{Float64}(undef,pSzwO[1],pSzwO[2])
#     daOx_dwO = Array{Float64}(undef,pSzwO[1],pSzwO[2])
#     pSzwH3 = size(wHs[3]) # 网络参数的矩阵。
#     daO_dwH3 = Array{Float64}(undef,pSzwH3[1],pSzwH3[2])
#     daOx_dwH3 = Array{Float64}(undef,pSzwH3[1],pSzwH3[2])
#     pSzwH2 = size(wHs[2]) # 网络参数的矩阵。
#     daO_dwH2 = Array{Float64}(undef,pSzwH2[1],pSzwH2[2])
#     daOx_dwH2 = Array{Float64}(undef,pSzwH2[1],pSzwH2[2])
#     pSzwH1 = size(wHs[1]) # 网络参数的矩阵。
#     daO_dwH1 = Array{Float64}(undef,pSzwH1[1])
#     daOx_dwH1 = Array{Float64}(undef,pSzwH1[1])
#     pSzbH3 = size(bHs[3]) # 网络参数的矩阵。
#     daO_dbH3 = Array{Float64}(undef,pSzbH3[1])
#     daOx_dbH3 =  Array{Float64}(undef,pSzbH3[1])
#     pSzbH2 = size(bHs[2]) # 网络参数的矩阵。
#     daO_dbH2 = Array{Float64}(undef,pSzbH2[1])
#     daOx_dbH2 =  Array{Float64}(undef,pSzbH2[1])
#     pSzbH1 = size(bHs[1]) # 网络参数的矩阵。
#     daO_dbH1 = Array{Float64}(undef,pSzbH1[1])
#     daOx_dbH1 =  Array{Float64}(undef,pSzbH1[1])
#     ###


#     # drop learning rate by half every 20 epochs
#     # eta = eta*(1/2)^floor(epoch/droprate);
#     # eta = eta * 1.0/2^(floor((epoch+1)/5000))

#     # loop over 
#     for i = 1:n # x 的100个点。
#         # feedforward 
#         # current
#         aHs,zHs,aO,zO = feedForward(wHs, bHs,wO,x[i]);
#         # aO = a_out 
#         zH1,zH2,zH3 = zHs[1],zHs[2],zHs[3]
#         aH1,aH2,aH3 = aHs[1],aHs[2],aHs[3]
        
#         wOT = wO' #  wO 转置一下。
#         daO_dx = wOT *  (dsig.(zH3) .* (wH3 * (dsig.(zH2) .* (wH2 * (dsig.(zH1) .* wH1)) )))
#         daO_dx = daO_dx[1]
#         # daO_dx = da_out_dx
#         # trial solutions
#         # a_out 即为文献中的 Ψ(xi)=a_out 
#         y_trial = C_1 + x[i] * aO  
#         # trial solution derivative
#         dytrial_dx = aO + x[i] * daO_dx  ;
#         # trial solution 2 derivative
#         # wO 
        
#         daO_dwO = sig.(zH3)
#         daOx_dwO = dsig.(zH3) .* (wH3*(dsig.(zH2).* (wH2*(dsig.(zH1).*wH1)))) # da_out_x/dwO
#         # wH

#         for j = 1:pSzwH3[2]
#             # tp1 = wOT * dsig.(zH3) * aH2[j]
#             daO_dwH3[:,j]  .= (wOT * dsig.(zH3) )* aH2[j]
#             tp1 = aH2 .* (wH2 * (aH1 .* wH1))
#             daOx_dwH3[:,j] .= (wOT * (d2sig.(zH3).* (wH3 * (dsig.(zH2) .* (wH2*(dsig.(zH1).* wH1)) )))*aH2[j]
#                            .+ (wOT *  dsig.(zH3)) * tp1[j] )
#         end                   

#         for j = 1:pSzwH2[2]
#             daO_dwH2[:,j] .= wOT * (dsig.(zH3) .* (wH3 * dsig.(zH2))) * aH1[j]
#             tp1 = aH1 .* wH1
#             daOx_dwH2[:,j] .= (wOT * (dsig.(zH3).* (wH3 * (d2sig.(zH2) .* (wH2*(dsig.(zH1).* wH1)) )))*aH1[j]
#                            .+  wOT * (dsig.(zH3).* (wH3 *  dsig.(zH2))) * tp1[j] )
#         end
        
#         # for j = 1:pSz[1]
#             daO_dwH1[:] .= wOT * (dsig.(zH3) .* (wH3 * (dsig.(zH2) .* (wH2 * dsig.(zH1))))) * x[i]
#             daOx_dwH2 .= wOT * (dsig.(zH3).* (wH3 * (d2sig.(zH2) .* (wH2*(d2sig.(zH1)+aH1)) )))
                           
#         # end
#         # bH
        
#         # for j = 1:pSz[1]
#             daO_dbH3[:] .= wOT * dsig.(zH3) 
#             daOx_dbH3[:] .= wOT*(d2sig.(zH3).*(wH3*(dsig.(zH2).*(wH2*(dsig.(zH1).*wH1)))))  # .* (wH3 * dsig.(zH2))) .* (wH2[j] * dsig.(zH1)) * x[i]
#         # end
        
#         # for j = 1:pSz[1]
#             daO_dbH2[:] .= wOT * (dsig.(zH3) .* (wH3 * dsig.(zH2)))  #) .* (wH2[j] * dsig.(zH1)) * x[i]
#             daOx_dbH2[:] .= wOT*(d2sig.(zH3).*(wH3*(d2sig.(zH2).*(wH2*(dsig.(zH1).*wH1)))))  # .* (wH3 * dsig.(zH2))) .* (wH2[j] * dsig.(zH1)) * x[i]
#         # end
        
#         # for j = 1:pSz[1]
#             daO_dbH1[:] .= wOT * (dsig.(zH3) .* (wH3 * (dsig.(zH2) .* (wH2 * dsig.(zH1))))) 
#             daOx_dbH1[:] .= wOT*(d2sig.(zH3).*(wH3*(d2sig.(zH2).*(wH2*(d2sig.(zH1).*wH1)))))  # .* (wH3 * dsig.(zH2))) .* (wH2[j] * dsig.(zH1)) * x[i]
#         # end

#         # gradient of network wrt output
#         dE1_daO = a(x[i])+x[i]*b(x[i])-x[i]*df_dy(x[i],y_trial)
#         dE1_daOx = x[i]*a(x[i])
#         # dE1_da_out2P = x[i]^2
#         # dE1_df = df_dy(x[i],y_trial)*x[i]^2
#         dE1 = [dE1_daO, dE1_daOx]
#         daO_dwOV = [daO_dwO , daOx_dwO  ] # dp 代表 parameter的倒数
#         daO_dwH3V = [daO_dwH3 ,daOx_dwH3 ]
#         daO_dwH2V = [daO_dwH2 ,daOx_dwH2 ]
#         daO_dwH1V = [daO_dwH1 ,daOx_dwH1 ]
#         daO_dbH3V = [daO_dbH3 ,daOx_dbH3  ]
#         daO_dbH2V = [daO_dbH2 ,daOx_dbH2  ]
#         daO_dbH1V = [daO_dbH1 ,daOx_dbH1  ]
#         ###
#         E1 =  a(x[i]) * dytrial_dx + b(x[i]) * y_trial - f(x[i],y_trial)
#         # error of layers
#         # output layer
#         # err_wO1 = grad_N0 * grad_N1 * sig(z_H)
#         # err_wO2 = grad_N0 * grad_N2 * (dsig(z_H) .* w_H)
#         # err_wO = err_wO1 .+ err_wO2 
#         kk=1
#         err_wO = 2 * E1 *  dE1[1:kk]' *   daO_dwOV[1:kk]
#         # hidden layer
#         err_wH3 = 2 * E1 *  dE1[1:kk]' *   daO_dwH3V[1:kk] 
#         err_wH2 = 2 * E1 *  dE1[1:kk]' *   daO_dwH2V[1:kk] 
#         err_wH1 = 2 * E1 *  dE1[1:kk]' *   daO_dwH1V[1:kk] 
#         ###
#         err_bH3 = 2 * E1 *  dE1[1:kk]' *   daO_dbH3V[1:kk]
#         err_bH2 = 2 * E1 *  dE1[1:kk]' *   daO_dbH2V[1:kk]
#         err_bH1 = 2 * E1 *  dE1[1:kk]' *   daO_dbH1V[1:kk]


#         # gradients of network parameters
#         # update llhlearning rate
        
#         # output layer weights
#         # dw_out = a_H*err_out;
#         dwO = err_wO;
#         # hidden layer weights
#         dwH3 = err_wH3;
#         dwH2 = err_wH2;
#         dwH1 = err_wH1;
#         # hidden layer bias
#         dbH3 = err_bH3;
#         dbH2 = err_bH2;
#         dbH1 = err_bH1;
                
#         # gradient descent
#         # output layer weights
#         wO = wO - eta*err_wO;
#         # hidden layer weights
#         wH3 = wH3 - eta*err_wH3;
#         wH2 = wH2 - eta*err_wH2;
#         wH1 = wH1 - eta*err_wH1;
#         # hidden layer bias
#         bH3 = bH3 - eta*err_bH3;
#         bH2 = bH2 - eta*err_bH2;
#         bH1 = bH1 - eta*err_bH1;
#         ###
#         wHs = [wH1,wH2,wH3 ]
#         bHs = [bH1,bH2,bH3 ]
#     end
#     # wHs = [wH1,wH2,wH3 ]
#     # bHs = [bH1,bH2,bH3 ]
#     return w,b,h,x 
# end
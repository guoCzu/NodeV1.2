

function C_idx(In1,H1,H2,H3,O1) # 各个层的神经元的下标

    C_in_idx = [In1];     # 输入层的神经元个数
    C_H_idx = [H1,H2,H3]  # 隐藏层的神经元的个数
    C_O_idx = [O1]        # 输出层的神经元的个数

    return C_in_idx,C_H_idx,C_O_idx
    
end
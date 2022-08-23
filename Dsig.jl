function dsig(z)
    a = sig(z) * (1.0 -sig(z));
    return a 
end
#######################
function d2sig(z)
    a = sig(z) * (1.0 -sig(z)) * (1.0 - 2.0* sig(z) );
    return a 
end
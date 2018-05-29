function [f1 f2] = NLfunForwardL(x,g,params)
e = params.numr*params.nr;
L = x;
R = params.Rv;
L = reshape(L,params.numr,params.nr);
R = reshape(R,params.numc,params.nr);
if isempty(g)
    f1 = params.afun(L*R');
    f2 = 0;
else 
    fp = params.afunT(g);
    f1 = vec(fp*R);
    f2 = vec(fp);
end
end
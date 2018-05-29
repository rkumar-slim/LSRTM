function [f1 f2] = NLfunForwardR(x,g,params)
e = params.numr*params.nr;
L = params.Lv;
R = x;
L = reshape(L,params.numr,params.nr);
R = reshape(R,params.numc,params.nr);
if isempty(g)
    f1 = params.afun(L*R');
    f2 = 0;
else 
    fp = params.afunT(g);
    f1 = vec(fp'*L);
    f2 = vec(fp);
end
end
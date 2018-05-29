function [f1,f2] = NLfunForwardLS(x,g,params)
e = params.numr*params.nr;
L = x(1:e);
R = x(e+1:end);
L = reshape(L,params.numr,params.nr);
R = reshape(R,params.numc,params.nr);
if isempty(g)
    f1 = vec(params.W*(L*R'));
    f2 = 0;
else 
    fp = params.W'*params.afunT(g);
    f1 = [vec(fp*R); vec(fp'*L)];
    f2 = vec(fp);
end
end


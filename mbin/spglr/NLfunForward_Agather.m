function [f1 f2] = NLfunForward_Agather(x,g,params)
e = params.numr*params.nr;
L = x(1:e);
R = x(e+1:end);
L = reshape(L,params.numr,params.nr);
R = reshape(R,params.numc,params.nr);
if isempty(g)
    f1 = vec(L*R');
    f2 = 0;
else 
    fp = reshape(g,params.numr,params.numc);
    f1 = [vec(fp*R); vec(fp'*L)];
    f2 = vec(fp);
end
end
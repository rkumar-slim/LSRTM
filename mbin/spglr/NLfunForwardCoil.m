function [f1,f2] = NLfunForwardCoil(x,g,params)

e = params.numr*params.nr;
L = x(1:e);
R = x(e+1:end);
L = reshape(L,params.numr,params.nr);
R = reshape(R,params.numc,params.nr);
if isempty(g)
    f1 = params.P*(vec(L*R'));
    f1 = Srcreg( f1, params, 1 );
    f1 = Recreg( f1, params, 1 );
    f2 = 0;
else
    fp = Recreg( g, params, -1 );
    fp = Srcreg( fp, params, -1 );
    fp = reshape(params.P'*vec(fp),params.numr,params.numc);
    f1 = [vec(fp*R);vec(fp'*L)];
    f2 = vec(fp);
end
end
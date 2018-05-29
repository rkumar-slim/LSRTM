function [f1,f2] = NLfunForwardinverp(x,g,params)
e = params.numr*params.nr;
L = x(1:e);
R = x(e+1:end);
L = reshape(L,params.numr,params.nr);
R = reshape(R,params.numc,params.nr);
if isempty(g)
    if params.modehelm==1
        f1 = params.Res*(params.MH'*vec(L*R'));
    else
        datapred = params.MH'*vec(L*R');
        f1       = [sqrt(params.lambda)*params.Res*(datapred);(datapred)];
    end
    f2 = 0;
else 
    if params.modehelm==1
        fp = reshape(params.MH*(params.Res'*g(:)),params.numr,params.numc);
        f1 = [vec(fp*R); vec(fp'*L)];
        f2 = vec(fp);
    else
        adjM = @(x)reshape(params.MH*x(:),params.numr,params.numc);% used to be MH'
        g1 = g(1:size(params.Res,1));
        g2 = g(size(params.Res,1)+1:end);
        gL = adjM(g2)*R + params.lambda*(adjM(params.Res'*(g1)))*R;
        gR = (adjM(g2))'*L + params.lambda*(adjM(params.Res'*(g1)))'*L;
        f1 = [vec(gL);vec(gR)];
        f2 = params.lambda*adjM(params.Res'*(g1));
    end
end
end
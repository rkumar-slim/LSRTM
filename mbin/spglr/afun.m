function [x] = afun(x,Ind,params)
% e = params.numr*params.nr;
% L = x(1:e);
% R = x(e+1:end);
% L = reshape(L,params.numr,params.nr);
% R = reshape(R,params.numc,params.nr);
%if params.logical==1
    x = vec(x);
    x(Ind)=0;
%else
%    x = vec(x);
%    x(Ind)=0;
%end
end
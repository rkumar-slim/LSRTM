function [f g] = genNorm(r, params)
p = params.p; % norm of p-norm
f = norm(r,p);

g = (r.^(p-1))*f^(1-p);
%g = r./f;

end
function x = NucNorm_project(x,weights,tau,params)
x = reshape(x,params.m, params.n);
[U S V] = svds(x,params.k);
s = diag(S);
clear S;

if isreal(s)
   s = oneProjector(s,weights,tau);
else
   sa  = abs(s);
   idx = sa < eps;
   sc  = oneProjector(sa,weights,tau);
   sc  = sc ./ sa; sc(idx) = 0;
   s   = s .* sc;
end

x = U*diag(s)*V';
x = x(:);

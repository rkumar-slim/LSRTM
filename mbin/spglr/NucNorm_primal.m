function p = NucNorm_primal(x,weights,params)
x = reshape(x,params.m, params.n);
x = svds(x,params.k);

p = norm(x.*weights,1);

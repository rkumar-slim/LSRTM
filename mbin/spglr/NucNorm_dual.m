function d = NucNorm_dual(x,weights,params)
x = reshape(x,params.m, params.n);
x = svds(x,1);

d = norm(x./weights,inf);

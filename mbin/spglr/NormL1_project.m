function x = NormL1_project(x,weights,tau)

if nargin == 2
    tau = weights;
    weights = 1;
end
    
if isreal(x)
   x = oneProjector(x,weights,tau);
else
   xa  = abs(x);
   idx = xa < eps;
   xc  = oneProjector(xa,weights,tau);
   xc  = xc ./ xa; xc(idx) = 0;
   x   = x .* xc;
end

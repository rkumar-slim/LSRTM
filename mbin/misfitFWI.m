function [f,g] = misfitRTM(m,Q,D,model)

% consensus optimization FWI LR
%
% use:
%   [f,g] = misfitRTM(m,Q,D,model,params)
%
% input:
%   m  - reference model
%   Q  - source
%   D  - data
%   model - struct with model paramaters
%   params.C - aqcuisition mask as matrix of size nrec x nsrc
%
% output:
%   f - function value
%   g - gradient

% predected data
Dsim = F(m,Q,model);

% Residual
Res = D - Dsim;

% objective
f = 0.5*norm(Res)^2;

% jacobian operator
opJ = oppDF(m,Q,model);

% gradient
g   = opJ'*Res;

end

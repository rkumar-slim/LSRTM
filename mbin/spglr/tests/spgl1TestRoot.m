function [ok] = spgl1TestRoot()

m = 120; n = 512; k = 20; % m rows, n cols, k nonzeros.
p = randperm(n); x0 = zeros(n,1); x0(p(1:k)) = sign(randn(k,1));
A  = randn(m,n); [Q,R] = qr(A',0);  A = Q';
b  = A*x0 + 0.005 * randn(m,1);


opts = spgSetParms('optTol',1e-4, ...
                   'project', @NormL1_project, ...
                   'primal_norm', @NormL1_primal, ...
                   'dual_norm', @NormL1_dual, ...
                   'subspaceMin', 0 ...
                   );



%% Run three separate root finding problems

%sigma = 1e-1;
sigma = 1e-5;


%params.funForward = @funForward;
params.funPenalty = @funLS;


[xLS, r, g, info] = spgl1(A, b, [], sigma, [], opts, params);
[xLS, r, g, info] = spgl1Orig(A, b, [], sigma, 0*x0, opts);


end


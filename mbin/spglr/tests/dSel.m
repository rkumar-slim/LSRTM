function [ok] = dantzigSelector()

m = 100; n = 500; k = 10; % m rows, n cols, k nonzeros.
p = randperm(n); x0 = zeros(n,1); x0(p(1:k)) = sign(randn(k,1));
A  = randn(m,n); [Q,R] = qr(A',0);  A = Q';
b  = A*x0 + 0.01 * randn(m,1);

bTild = A'*b;  % b for dantzig selector
ATild = A'*A;  % A for dantzig selector

p = 10;
params.p = p;   % p for genNorm

funOne = @funLS;
funTwo = @genNorm;


% opts = spgSetParms('optTol',1e-5, ...
%                    'project', @NormL1_project, ...
%                    'primal_norm', @NormL1_primal, ...
%                    'dual_norm', @NormL1_dual, ...
%                    'subspaceMin', 0, ...
%                    'funPenalty', funOne ... 
%                    );


% params.nu = 1e-2;
% params.hub = 0.005;
% params.rho = 1;

%% Run three separate root finding problems

%sigma = funOne(Err, params);


% first run: dantzig with 2-norm, using funLS
%params.funPenalty = funOne;
%[xLS, r, g, info] = spgl1(A, b, [], sigma, [], opts, params);



% second run: dantzig with 2-norm, using genNorm
opts = spgSetParms('optTol',1e-5, ...
                   'project', @NormL1_project, ...
                   'primal_norm', @NormL1_primal, ...
                   'dual_norm', @NormL1_dual, ...
                   'subspaceMin', 0, ...
                   'funPenalty', funTwo ... 
                   );

sigma = .005;
[xGN, r, g, info] = spgl1(ATild, bTild, [], sigma, [], opts, params);


%  cvx_begin
%    variables xCVX(n)
%    minimize( norm(xCVX,1))  
%    subject to 
%     norms(ATild*xCVX - bTild, p) <= sigma
%  cvx_end


 cvx_begin
   variables xDantzig(n)
   minimize( norm(xDantzig,1))  
   subject to 
    norms(ATild*xDantzig - bTild, inf) <= sigma
 cvx_end

 
 
fprintf('one norm of SPGL1: %5.3f, one-norm of inf-Dantzig: %5.3f\n', norm(xGN, 1), norm(xDantzig, 1));

fprintf('norm diff between inf-Dantzig and SPGL1: %5.3f \n', norm(xGN - xDantzig, inf));

%fprintf('norm diff between P-Dantzig and Dantzig: %5.3f\n', norm(xCVX - xDantzig, inf));



%% Plot the results

figure(1)
plot(1:n, x0 + 2,  1:n, xGN(1:n), 1:n, xDantzig(1:n) - 2,   '-k', 'Linewidth', 1.5);

legend('true', 'P-Dantz',  'Dantzig');

figure(2);
plot(1:m, b - A*x0 + 1.5, 1:m, b - A*xGN(1:n), 1:m, b - A*xDantzig(1:n) - 1.5,   '-k', 'Linewidth', 2); 
legend('true', 'P-Dantz',  'Dantzig');

% 
% figure(1)
% plot(1:n, x0 + 2,  1:n, xGN(1:n) - 2, 1:n, xCVX(1:n), 1:n, xDantzig(1:n) - 4,   '-k', 'Linewidth', 1.5);
% 
% legend('true', 'P-Dantz', 'CVX-P-Dantzig', 'Dantzig');
% 
% figure(2);
% plot(1:m, b - A*x0 + 1.5, 1:m, b - A*xGN(1:n) - 1.5, 1:m, b - A*xCVX(1:n), 1:m, b - A*xDantzig(1:n) - 3,   '-k', 'Linewidth', 2); 
% legend('true', 'P-Dantz', 'CVX-P-Dantzig', 'Dantzig');
% 

end


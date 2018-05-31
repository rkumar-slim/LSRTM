clear all; clc;
load OED_shepp.mat
nImg = [64 64];
idk = randperm(size(A,1));
idk = idk(1:5000);
A = sparse(A(idk,:));
A = sprandn(prod(nImg),prod(nImg),9/prod(nImg));
A = A + 10*speye(size(A));
%%
K = convFFT(nImg,[3 3 1 1]);
% generate data
data = A*x_true;
data = data +0* randn(size(data))/10;
layer = varNetLayer(A,data,K,'activation',@quadActivation);
net = ResNN(layer,50,1e-3);

%% test first
% theta = 1*[0 -1 0; -1 4 -1; 0 -1 0];
theta = 1*[0 0 0; 0 1 0; 0 0 0];
theta = repmat(vec(theta),net.nt,1);
tic;
Y = apply(net,theta,A'*data);
toc;
%%
figure(1); clf;
subplot(2,2,1)
montageArray(reshape(A'*data,nImg(1),nImg(2),[]))
title('backprojection')
colorbar
subplot(2,2,2)
montageArray(reshape(Y,nImg(1),nImg(2),[]))
title('var net')
colorbar
subplot(2,2,3)
montageArray(reshape(x_true,nImg(1),nImg(2),[]))
title('true')
colorbar

subplot(2,2,4)
montageArray(reshape(A\data,nImg(1),nImg(2),[]))
title('LS solution')
colorbar

%% 
pReg = tikhonovReg(opEye(nTheta(net)),1e-2);
layer = varNetLayer(A,data(:,1:15),K,'activation',@quadActivation);
net = ResNN(layer,50,1e-3);
fctn = mseObjFctn(net,pReg,A'*data(:,1:15),x_true(:,1:15));
% [isOK,his] = checkDerivative(fctn,initTheta(net),'out',1);

%%
opt = newton('out',1,'maxIter',10);
th0 = theta;
thOpt = solve(opt,fctn,th0);

%%
figure(3); clf;
subplot(1,2,1);
montageArray(reshape(th0,3,3,[]))
title('starting guess')
subplot(1,2,2);
montageArray(reshape(thOpt,3,3,[]))
title('optimized kernels')


%%
layer = varNetLayer(A,data,K,'activation',@quadActivation);
net = ResNN(layer,50,1e-3);

tic;
Yopt = apply(net,thOpt,A'*data);
toc;
tic;
Y0 = apply(net,th0,A'*data);
toc;

figure(4); clf;
subplot(2,2,1)
montageArray(reshape(A'*data,nImg(1),nImg(2),[]))
title('backprojection')
colorbar
subplot(2,2,2)
montageArray(reshape(Y0,nImg(1),nImg(2),[]))
title('initial theta')
caxis([0 1])
colorbar
subplot(2,2,3)
montageArray(reshape(Yopt,nImg(1),nImg(2),[]))
title('optimal theta')
caxis([0 1])
colorbar

subplot(2,2,4)
montageArray(reshape(x_true,nImg(1),nImg(2),[]))
title('true')
colorbar

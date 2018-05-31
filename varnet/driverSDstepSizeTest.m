clear all; clc;
load OED_shepp.mat
nImg = [64 64];
 idk = randperm(size(A,1));
 idk = idk(1:768);
 A = sparse(A(idk,:));
%  A = sprandn(prod(nImg),prod(nImg),9/prod(nImg));
% A = A + 10*speye(size(A));
%% manipualte last image
figure(1); clf;

xt =reshape(x_true(:,end),nImg);
xt(45:50,40:41) = 1;
xt(47:48,40:45) = 1;
imagesc(xt)
x_true(:,end)=xt(:);
%%
K = convFFT(nImg,[5 5 1 1],'Q',eye(25)-ones(25)/25);
% generate data

data = A*x_true;
data = data +0.01* randn(size(data))/10;
M     = scalingKernel([size(x_true,1) size(x_true,1)]);
layer = varNetLayerPC(A,data,M,K,'activation',@quadActivation);
net = ResNN(layer,20,2e-3,'outTimes',ones(20,1));

%% test first
%  theta = 1e-1*[0 -1 0; -1 4 -1; 0 -1 0]; 
%  theta = 1*[0 0 0; 0 1 0; 0 0 0];
theta = zeros(5,5);
theta(3,3)=1;
theta = repmat([vec(theta);.5*initTheta(M)],net.nt,1);
tic;
[~,Y] = apply(net,theta,0*A'*data);
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
net.layer.b = data(:,1:15);
pReg = tikhonovReg(opEye(nTheta(net)),1e-4);
fctn = mseObjFctn(net,pReg,0*A'*data(:,1:15),repmat(x_true(:,1:15),net.nt,1));
% [isOK,his] = checkDerivative(fctn,initTheta(net),'out',1);

%%
opt = newton('out',1,'maxIter',10);
th0 = theta;
thOpt = solve(opt,fctn,th0);

%%
th0 = reshape(th0,[],net.nt);
thOpt = reshape(thOpt,[],net.nt);
figure(3); clf;
subplot(2,2,1);
montageArray(reshape(K.Q*th0(1:25,:),5,5,[]))
title('starting guess')
colorbar
subplot(2,2,2);
montageArray(reshape(K.Q*thOpt(1:25,:),5,5,[]))
title('optimized kernels')
colorbar
subplot(2,2,3);
imagesc(th0(26:end,:));
title('scaling, th0')
colorbar
subplot(2,2,4);
imagesc(thOpt(26:end,:));
title('scaling, thOpt')
colorbar
%%
net.layer.b = data;

tic;
[~,Yopt] = apply(net,thOpt(:),0*A'*data);
toc;
tic;
[~,Y0] = apply(net,th0,0*A'*data);
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
%% intermediate iterates
[Yopt,~,Yall] = apply(net,thOpt(:),0*A'*data);
[Y0,~,Yall0] = apply(net,th0(:),0*A'*data);

id = 20;
Yimg = [];
Yimg0 = [];
for k=1:size(Yall,1)
    Yimg = [Yimg Yall{k,1}(:,id)];
    Yimg0 = [Yimg0 Yall0{k,1}(:,id)];
end

figure(5); clf;
subplot(4,1,1);
montageArray(reshape(Yimg,nImg(1),nImg(2),[]),20);
axis equal tight
title('reconstructions per iter')

subplot(4,1,2);
resOpt = sum((Yimg-x_true(:,id)).^2,1);
montageArray(reshape(Yimg-x_true(:,id),nImg(1),nImg(2),[]),20);
axis equal tight
title('residual per iter')

subplot(4,1,3);
montageArray(reshape(Yimg0,nImg(1),nImg(2),[]),20);
axis equal tight
title('reconstructions per iter')

subplot(4,1,4);
res0 = sum((Yimg0-x_true(:,id)).^2,1);
montageArray(reshape(Yimg0-x_true(:,id),nImg(1),nImg(2),[]),20);
axis equal tight
title('residual per iter')

%% try SD with exact ls
d0   = data(:,end);
Kop  = getOp(K,K.Q*theta(1:25));
xc   = zeros(size(A,2),1);
his = zeros(net.nt,4);
Xall = zeros(prod(nImg),1);
for k=1:net.nt
    %misfit
    res = A*xc-d0;
    Dc  = 0.5*(res'*res);
    dD  = A'*res;
    % regularizer
    res = Kop*xc;
    Rc  = 0.5*(res'*res);
    dR  = Kop'*res;
    H   = @(x)  A'*(A*x) + Kop'*(Kop*x);
    % putting things together
    Jc = Dc + Rc;
    dJ = dD+dR;
    s  = dJ;
    %exact LS
    mu = ((s'*dJ)/(2*s'*H(s)));
    
    xc = xc - mu*s;
    
    his(k,:) = [Jc Dc mu norm(xc-x_true(:,end)).^2];
    fprintf('%d\t%1.2e\t%1.2e\t%1.2e\t%1.2e\n',k,his(k,:));
     Xall(:,k) = xc(:);
end
    
%%
figure(5); clf;
plot(his(:,end),'-+','LineWidth',2);
hold on
plot(resOpt,'-o','linewidth',2)
% plot(res0,');,
legend('steepest descent','learned reconstruction')
%%
figure(6); clf;
montageArray(reshape(Xall,nImg(1),nImg(2),[]),20);
axis equal tight
caxis([0 1])
%%
figure(6); clf;
montageArray(reshape(xc,nImg));
caxis([0 1])
    

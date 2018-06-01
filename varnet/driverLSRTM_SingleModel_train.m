clear all ;clc;
%curdir = pwd;
% addpath(genpath('/home/rajivkumar/LSRTM'));
% addpath(genpath('/home/rajivkumar/Meganet.m'));
% random seed
s = RandStream('mt19937ar','Seed',1);
RandStream.setGlobalStream(s);

% load true model
load marmousi.mat
n = size(v);
v = v(1:300,1:1000);
% construct cells of different models
C = mat2cell(v,[100 100 100],[100 100 100 100 100 100 100 100 100 100]);
% smoothing levels
slevel = [5 10 15 20 50 100];
%% different models
nsub   = [100 100];
Models = cell(prod(size(C)),length(slevel));
Modelt = cell(prod(size(C)),length(slevel));
for i = 1:length(slevel)
    % smoothing operator
    S      = opKron(opSmooth(nsub(2),slevel(i)),opSmooth(nsub(1),slevel(i)));
    for j = 1:prod(size(C))
        m           = C{j};
        m           = 1e6./m.^2;
        Models{j,i} = S*vec(m);
        Modelt{j,i} = m(:);
    end
end
%% training data
perc    = 0.1; % percentage of training sample selections
index   = randperm(prod(size(C))*length(slevel));
index   = index(1:floor(length(index)*perc));
index = [1 ];
[In,Jn] = ind2sub([prod(size(C)) length(slevel)],index);
length(index)
% creat ebig matirx of test true models
Mt_train = zeros(100*100,length(index));
for i = 1:length(index)
    Mt_train(:,i) = Modelt{In(i),Jn(i)};
end
% create big matrix of test smooth models
Ms_train = zeros(100*100,length(index));
for i = 1:length(index)
    Ms_train(:,i) = Models{In(i),Jn(i)};
end

%% create synthetic data for testing and define model parameters
z                       = 0:5:495;
x                       = 0:5:495;
n                       = nsub;
model.n                 = n;
model.o                 = [0 0];
model.d                 = [5 5];
model.xt                = x;
model.zt                = z;
model.nb                = [60 60;60 60];
model.freq              = [5:1:40];
model.nf                = numel(model.freq);
model.f0                = 20; %peak freq of ricker wavelet
model.t0                = 0; %phase shift of wavelet in seconds
%receivers and sources near the top
model.zsrc              = model.d(2);
model.xsrc              = model.xt(1:10:end);
model.zrec              = model.d(2);
model.xrec              = model.xt(1:end);
ns                      = length(model.xsrc);
nr                      = length(model.xrec);
Q                       = speye(ns);
model.unit              = 's2/km2';
model.nsamples          = length(index);
model.nsrc              = size(Q,2);
model.nrec              = length(model.zrec)*length(model.xrec);
model.nfreq             = length(model.freq);
model.datan             = model.nsrc*model.nrec*model.nfreq;
%% generate true seismic data
Df1                     = Fm(Mt_train,Q,model);
Df2                     = Fm(Ms_train,Q,model);
b                       = Df1 - Df2;
b                       = reshape(gather(b),model.nsrc*model.nrec*model.nfreq,model.nsamples);
%% Lu factorization
tic;[LL,UU,Pp,Qp,Rr,dH] = LUFact(Ms_train,Q,model);toc
A                       = oppDFLU(Ms_train,Q,LL,UU,Pp,Qp,Rr,dH,model);

%% r
nImg                    = nsub;
K                       = convFFT(nImg,[5 5 1 1],'Q',eye(25)-ones(25)/25);
x_true                  = Mt_train - Ms_train;
data                    = b;
data                    = data +0.01* randn(size(data))/10;
fprintf('computing backprojection solution...\n');
tic;
xBackProj               = A'*data(:);
dBackProj               = A*xBackProj;
timeBP = toc;
fprintf('\t...misfit=%1.2e, error=%1.2e, time(A''A)=%1.2f\n',...
    norm(xBackProj(:)-x_true(:)),0.5*norm(data(:)-dBackProj(:)).^2,timeBP);
% return
%% test first

M                       = scalingKernel([size(x_true,1) size(x_true,1)]);
layer                   = varNetLayerPC_seismic(A,data,M,K,model,'activation',@quadActivation);
net                     = ResNN(layer,20,3e-9);

%%

theta = zeros(5,5);
theta(3,3)=1;
theta = repmat([vec(theta);1],net.nt,1);
x0 = zeros(prod(model.n),model.nsamples);

fprintf('running VarNet with initial weights...\n');
tic;
xVarNet0 = apply(net,theta,x0);
dVarNet0 = A*xVarNet0;
timeVN0 = toc;
fprintf('\t...misfit=%1.2e, error=%1.2e, time(A''A)=%1.2f\n',...
    norm(xVarNet0(:)-x_true(:)),0.5*norm(data(:)-dVarNet0(:)).^2,timeVN0);

%%
figure(1); clf;
subplot(2,2,1)
montageArray(reshape(xBackProj,nImg(1),nImg(2),[]))
title('backprojection')
colorbar
subplot(2,2,2)
montageArray(reshape(xVarNet0,nImg(1),nImg(2),[]))
title('var net')
colorbar
subplot(2,2,3)
montageArray(reshape(x_true,nImg(1),nImg(2),[]))
title('true')
colorbar

%%
net.layer.b = data;
pReg = tikhonovReg(opEye(nTheta(net)),1e-4);
fctn = mseObjFctn(net,pReg,zeros(prod(model.n),model.nsamples),x_true);
% [isOK,his] = checkDerivative(fctn,initTheta(net),'out',1);

%%
opt = sd('out',1,'maxIter',10,'maxStep',1);
th0 = theta;
thOpt = solve(opt,fctn,th0);

%%
fprintf('running VarNet with learned weights...\n');
tic;
xVarNet = apply(net,thOpt,x0);
dVarNet = A*xVarNet;
timeVN = toc;
fprintf('\t...misfit=%1.2e, error=%1.2e, time(A''A)=%1.2f\n',...
    norm(xVarNet(:)-x_true(:)),0.5*norm(data(:)-dVarNet(:)).^2,timeVN);

save singleModelTrain-20 xVarNet xVarNet0 thOpt th0
return

% standard FWI example
clear all;clc
pwd = curdir;
addpath(genpath(pwd));

% random seed
s = RandStream('mt19937ar','Seed',1);
RandStream.setGlobalStream(s);

% define model
z             = 0:5:1000;
x             = 0:5:1000;
v             = 1500*ones(length(z),length(x));
v(50:end,:)   = 1800;
v(80:end,:)   = 2200;

%% define model paramater
n             = size(v);
model.n       = n;
model.o       = [0 0];
model.d       = [10 10];
model.xt      = x;
model.zt      = z;
model.nb      = [60 60;60 60];

% set up frequency
model.freq    = [5:0.25:40];
model.nf      = numel(model.freq);
model.f0      = 20; %peak freq of ricker wavelet
model.t0      = 0; %phase shift of wavelet in seconds

%receivers and sources near the top
model.zsrc   = model.d(2);
model.xsrc   = model.xt(1:5:end);
model.zrec   = model.d(2);
model.xrec   = model.xt(1:2:end);
ns           = length(model.xsrc);
nr           = length(model.xrec);
Q            = speye(ns);
model.unit   = 's2/km2';
m            = 1e6./v(:)^2;
% smooth model
S           = opKron(opSmooth(n(2),10),opSmooth(n(1),10));
m0          = S*m;

% generate true seismic data
D           = F(m,Q,model);

% generate data in background model
D0          = F(m0,Q,model);

% linearized Data
b           = D - D0;

% curvelet operator
C           = opWavelet(n(1),n(2),'HAAR',8,5);

% jacobian (operator A)
J           = oppDF(m0,Q,model);
A           = J*C';

% plain RTM
dm          = J'*b;
dm          = reshape(dm,n);
figure(1);imagesc(x,z,diff(dm,1));title('RTM');

% Sparsity promition based LSRTM
opts         = spgSetParms('iterations',20);
[x,r,g,info] = spgl1(A,b,0,1e-3,[],opts);

% LSRTM image
LSdm           = reshape(real(C'*x),n);
LSdm           = reshape(LSdm,n);
figure(2);imagesc(x,z,diff(LSdm,1));title('LSRTM');

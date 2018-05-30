% standard FWI example
clear all;clc
curdir = pwd;
addpath(genpath(curdir));
% random seed
s = RandStream('mt19937ar','Seed',1);
RandStream.setGlobalStream(s);
% define model
z             = 0:5:500;
x             = 0:5:500;
v             = 1500*ones(length(z),length(x));
v(50:end,:)   = 1800;
v(80:end,:)   = 2200;
%% define model paramater
n             = size(v);
model.n       = n;
model.o       = [0 0];
model.d       = [5 5];
model.xt      = x;
model.zt      = z;
model.nb      = [60 60;60 60];
% set up frequency
model.freq    = [5:1:40];
model.nf      = numel(model.freq);
model.f0      = 20; %peak freq of ricker wavelet
model.t0      = 0; %phase shift of wavelet in seconds
%receivers and sources near the top
model.zsrc   = model.d(2);
model.xsrc   = model.xt(1:10:end);
model.zrec   = model.d(2);
model.xrec   = model.xt(1:end);
ns           = length(model.xsrc);
nr           = length(model.xrec);
Q            = speye(ns);
model.unit   = 's2/km2';
m            = 1e6./v(:).^2; % velocity to slowness^2
% smooth model
S            = opKron(opSmooth(n(2),10),opSmooth(n(1),10));
m0           = S*m;
% generate true seismic data
D           = F(m,Q,model);
% generate data in background model
D0          = F(m0,Q,model);
% linearized Data
b           = D - D0;
%% LU factorization
tic;[LL,UU,Pp,Qp,Rr,dH] = LUFact(m0,Q,model);toc
J           = oppDFLU(m0,Q,LL,UU,Pp,Qp,Rr,dH,model);
tic;dm      = J'*b;toc
dd1 = J*dm;
% dm          = reshape(dm,n);
% figure(2);imagesc(x,z,diff(dm,1));title('LU');
%% no LU factorization
J           = oppDF(m0,Q,model);
tic;dm1      = J'*b;toc;
dd2 = J*dm1;
% dm          = reshape(dm,n);
% figure(1);imagesc(x,z,diff(dm,1));title('no LU');
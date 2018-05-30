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
v1             = 1500*ones(length(z),length(x));
v1(30:end,:)   = 1800;
v1(60:end,:)   = 2200;

v2             = 1500*ones(length(z),length(x));
v2(50:end,:)   = 1800;
v2(80:end,:)   = 2200;

v3             = 1500*ones(length(z),length(x));
v3(20:end,:)   = 1800;
v3(80:end,:)   = 2200;

v4             = 1500*ones(length(z),length(x));
v4(50:end,:)   = 1800;
v4(90:end,:)   = 2200;
%% define model paramater
n             = size(v1);
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
model.nsamples = 4;
m1           = 1e6./v1(:).^2; % velocity to slowness^2
m2           = 1e6./v2(:).^2; % velocity to slowness^2
m3           = 1e6./v3(:).^2; % velocity to slowness^2
m4           = 1e6./v4(:).^2; % velocity to slowness^2
% smooth model
S            = opKron(opSmooth(n(2),10),opSmooth(n(1),10));
m01          = S*m1;
m02          = S*m2;
m03          = S*m3;
m04          = S*m4;

%% generate true seismic data
Df1          = Fm([m1 m2 m3 m4],Q,model);
Df2          = Fm([m01 m02 m03 m04],Q,model);
bf           = Df1 - Df2;
%% Lu factorization
tic;[LL,UU,Pp,Qp,Rr,dH] = LUFactm([m01 m02 m03 m04],Q,model);toc
%% all in one go approach 
tic;output       = DFm([m01 m02 m03 m04],Q,bf,-1,model);toc;
tic;bfd       = DFm([m01 m02 m03 m04],Q,output,1,model);toc;
output       = reshape(output,model.n(1),model.n(2),model.nsamples);
figure(1);imagesc(x,z,diff(output(:,:,1),1));title('1');
figure(2);imagesc(x,z,diff(output(:,:,2),1));title('2');
figure(3);imagesc(x,z,diff(output(:,:,3),1));title('3');
figure(4);imagesc(x,z,diff(output(:,:,4),1));title('4');
%% LU factorization
tic;output       = DFmLU([m01 m02 m03 m04],Q,bf,-1,LL,UU,Pp,Qp,Rr,dH,model);toc;
tic;bfdlu        = DFmLU([m01 m02 m03 m04],Q,output,1,LL,UU,Pp,Qp,Rr,dH,model);toc;
output       = reshape(output,model.n(1),model.n(2),model.nsamples);
figure(5);imagesc(x,z,diff(output(:,:,1),1));title('1lu');
figure(6);imagesc(x,z,diff(output(:,:,2),1));title('2lu');
figure(7);imagesc(x,z,diff(output(:,:,3),1));title('3lu');
figure(8);imagesc(x,z,diff(output(:,:,4),1));title('4lu');


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
D1           = F(m1,Q,model);
D2           = F(m2,Q,model);
D3           = F(m3,Q,model);
D4           = F(m4,Q,model);

% generate data in background model
D01          = F(m01,Q,model);
D02          = F(m02,Q,model);
D03          = F(m03,Q,model);
D04          = F(m04,Q,model);

% linearized Data
b1           = D1 - D01;
b2           = D2 - D02;
b3           = D3 - D03;
b4           = D4 - D04;
%% serial
J1           = oppDF(m01,Q,model);
tic;dm1      = J1'*b1;toc;
dm1          = reshape(dm1,n);
figure(1);imagesc(x,z,diff(dm1,1));title('1');

J2           = oppDF(m02,Q,model);
tic;dm2      = J2'*b2;toc;
dm2          = reshape(dm2,n);
figure(2);imagesc(x,z,diff(dm2,1));title('2');

J3           = oppDF(m03,Q,model);
tic;dm3      = J3'*b3;toc;
dm3          = reshape(dm3,n);
figure(3);imagesc(x,z,diff(dm3,1));title('3');

J4           = oppDF(m04,Q,model);
tic;dm4      = J4'*b4;toc;
dm4          = reshape(dm4,n);
figure(4);imagesc(x,z,diff(dm4,1));title('4');
%% all in one go approach 
Df1          = Fm([m1 m2 m3 m4],Q,model);
Df2          = Fm([m01 m02 m03 m04],Q,model);
bf           = Df1 - Df2;
output       = DFm([m01 m02 m03 m04],Q,bf,-1,model);
output       = reshape(output,model.n(1),model.n(2),model.nsamples);
figure(5);imagesc(x,z,diff(output(:,:,1),1));title('1j');
figure(6);imagesc(x,z,diff(output(:,:,2),1));title('2j');
figure(7);imagesc(x,z,diff(output(:,:,3),1));title('3j');
figure(8);imagesc(x,z,diff(output(:,:,4),1));title('4j');



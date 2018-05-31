clear all ;clc;
curdir = pwd;
addpath(genpath(curdir));

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
        Modelt{j,i} = m;
    end
end
%% training data
perc    = 0.5; % percentage of training sample selections
index   = randperm(prod(size(C))*length(slevel));
index   = index(1:floor(length(index)*perc));
[In,Jn] = ind2sub([prod(size(C)) length(slevel)],index);

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
z              = 0:5:495;
x              = 0:5:495;
n              = nsub;
model.n        = n;
model.o        = [0 0];
model.d        = [5 5];
model.xt       = x;
model.zt       = z;
model.nb       = [60 60;60 60];
% model.freq    = [5:20:40];
model.freq     = 5;
model.nf       = numel(model.freq);
model.f0       = 20; %peak freq of ricker wavelet
model.t0       = 0; %phase shift of wavelet in seconds
%receivers and sources near the top
model.zsrc     = model.d(2);
model.xsrc     = model.xt(1:10:end);
model.zrec     = model.d(2);
model.xrec     = model.xt(1:end);
ns             = length(model.xsrc);
nr             = length(model.xrec);
Q              = speye(ns);
model.unit     = 's2/km2';
model.nsamples = length(index);
%% generate true seismic data
Df1            = Fm(Mt_train,Q,model);
Df2            = Fm(Ms_train,Q,model);
bf             = Df1 - Df2;
%% Lu factorization
tic;[LL,UU,Pp,Qp,Rr,dH] = LUFact(Ms_train,Q,model);toc
J                       = oppDFLU(Ms_train,Q,LL,UU,Pp,Qp,Rr,dH,model);
% adjoint
%tic;output              = J'*bf;toc;
% forward
%tic;bfdlu               = J*output;toc;
%% variational network

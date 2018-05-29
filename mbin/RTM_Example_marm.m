% standard FWI example
clear all;clc
addpath(genpath('/Users/rjbaraldi12/Documents/UW/sasha_reading/Inversion/interpolation/Joint_Inver_Intrp/mbin'));
addpath(genpath('/Users/rjbaraldi12/Documents/UW/sasha_reading/minConf'));
addpath(genpath('/Users/rjbaraldi12/Dropbox/Interpolation/spot')); %% need minConf
addpath(genpath('/Users/rjbaraldi12/Dropbox/Interpolation/pSpot')); %% need minConf
load BG.mat
% random seed
s = RandStream('mt19937ar','Seed',1);
RandStream.setGlobalStream(s);
%% define model paramater
n             = size(v);
model.n       = n; %(z,x)
model.o       = [0 0];
model.d       = [12.5 12.5];
model.xt      = 0:model.d(2):(model.n(2)-1)*model.d(2);
model.zt      = 0:model.d(1):(model.n(1)-1)*model.d(1);
model.nb      = [60 60;60 60];
% set up frequency
freq          = [(3:0.25:4)' (4:0.25:5)' (5:0.25:6)' (6:0.25:7)' (7:0.25:8)' (8:0.25:9)' (9:0.25:10)',...
                (10:0.25:11)' (11:0.25:12)' (12:0.25:13)' (13:0.25:14)' (14:0.25:15)' (15:0.25:16)' (16:0.25:17)',...
                (17:0.25:18)' (18:0.25:19)' (19:0.25:20)' (20:0.25:21)' (21:0.25:22)' (22:0.25:23)' (23:0.25:24)' (24:0.25:25)'];
model.nf      = numel(freq);
model.f0      = 10; %peak freq of ricker wavelet
model.t0      = 0; %phase shift of wavelet in seconds

%receivers and sources near the top
model.zsrc   = model.d(2);
model.xsrc   = model.xt(1:3:end);
model.zrec   = model.d(2);
model.xrec   = model.xt(1:2:end);
ns           = length(model.xsrc); model.ns = ns;
nr           = length(model.xrec); model.nr = nr;
model.nsim   = 10; %number of simultaneous shots
model.redraw = 1; %redraw flag
ssW          = 1;
%background velocity and slowness squared
model.vtrue  = v(:);

model.vmin  = min(v(:));
model.vmax  = max(v(:));
model.mtrue = 1e6./model.vtrue.^2;
model.mmin  = 1e6./model.vmax.^2;
model.mmax  = 1e6./model.vmin.^2;
model.unit  = 's2/km2';
%changed this -> assume no free surface
model.freesurface = 0; 
% smooth model
S           = opKron(opSmooth(n(2),50),opSmooth(n(1),50));
m0          = S*model.mtrue;
%% run inversion
for i = 1:size(freq,2)
    % minimize over m
    fprintf('\n **** SOLVING FOR m : freq batch %d ****** \n',i);
    model.freq  = freq(:,i);
    
    %initialize stuff
    Q         = speye(model.ns); %source weight (simultaneous sources)
    
    % generate true seismic data
    D         = F(model.mtrue(:),Q,model);
    fprintf('\n **** data generated for freq batch %d ****** \n',i);
    % define the gaussian random matrix
    model.ssW = 1;
    
    % optimization options - SPG
    options.maxIter  = 40;
    options.memory   = 5;
    options.testOpt  = 0;
    options.progTol  = 1e-14;
    options.evol_tol = 1e-14;
    oit              = 0;
    boptions.LB = model.mmin*ones(size(m0));
    boptions.UB = model.mmax*ones(size(m0));
    boptions.l  = model.mmin;
    boptions.u  = model.mmax;
    funProj     = @(x) boundProjectstan(x,boptions);  % incorporates water velocity constraints
    fh          = @(x) misfitFWI(x,Q,D,model);
    m0          = minConf_SPG(fh,m0,funProj,options);
    figure(1);imagesc(reshape(m0,model.n));drawnow;
    save(['update_' num2str(i)] , 'm0');
end


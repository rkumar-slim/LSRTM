function [LL,UU,Pp,Qp,Rr,dH] = LUFactm(m,Q,model)
% Frequency domain FD modeling operator
%
% use:
%   D = F(m,Q,model)
% input:
%   m                 - vector with gridded squared slowness in [km^2/s^2]
%   Q                 - source matrix. size(Q,1) must match source grid
%                       definition, size(Q,2) determines the number of
%                       sources, if size(Q,3)>1, it represents a
%                       frequency-dependent source and has to be
%                       distributed over the last dimension.
%   model.{o,d,n}     - physical grid: z = ox(1) + [0:nx(1)-1]*dx(1), etc.
%   model.nb          - number of points to add for absorbing boundary
%   model.freq        - frequencies
%   model.f0          - peak frequency of Ricker wavelet, 0 for no wavelet.
%   model.t0          - phase shift [s] of wavelet.
%   model.{zsrc,xsrc} - vectors describing source array
%   model.{zrec,xrec} - vectors describing receiver array.
%
% output:
%   D  - Data cube (nrec x nsrc x nfreq) as (distributed) vector. nsrc  = size(Q,2);
%                                                                 nrec  = length(zrec)*length(xrec)
%                                                                 nfreq = length(freq)
% comp. grid
dt = model.d;
nt = model.n+2*model.nb(1,:);
nfreq  = length(model.freq);

% define wavelet
w = exp(1i*2*pi*model.freq*model.t0);
if model.f0
    % Ricker wavelet with peak-frequency model.f0
    w = (model.freq).^2.*exp(-(model.freq/model.f0).^2).*w;
end

% mapping from source/receiver/physical grid to comp. grid
Px = opKron(opExtension(model.n(2),model.nb(1,2)),opExtension(model.n(1),model.nb(1,1)));
% model parameter: slowness [s/m] on computational grid.
mu = Px*m;

% distribute frequencies according to standard distribution
freq = distributed(model.freq);
spmd
    codistr  = codistributor1d(2,[],[prod(nt)*prod(nt)*model.nsamples,nfreq]);
    freqloc  = getLocalPart(freq);
    nfreqloc = length(freqloc);
    LLloc    = [];
    UUloc    = [];
    Pploc    = [];
    Qploc    = [];
    Rrloc    = [];
    dHloc    = [];
    for k = 1:nfreqloc
        LL = []; UU = []; Rr = []; Qp = []; Pp = []; dH = [];
        for i = 1:model.nsamples
            [Hk, dHk]             = Helm2D_opt(mu(:,i),dt,nt,model.nb,model.unit,freqloc(k),model.f0);
            [LL1,UU1,Pp1,Qp1,Rr1] = lu(Hk);
            LL                    = [LL vec(LL1)];
            UU                    = [UU vec(UU1)];
            Qp                    = [Qp vec(Qp1)];
            Pp                    = [Pp vec(Pp1)];
            Rr                    = [Rr vec(Rr1)];
            dH                    = [dH vec(dHk)];
        end
        LLloc            = [LLloc vec(LL)];
        UUloc            = [UUloc vec(UU)];
        Pploc            = [Pploc vec(Pp)];
        Qploc            = [Qploc vec(Qp)];
        Rrloc            = [Rrloc vec(Rr)];
        dHloc            = [dHloc vec(dH)];
    end
    LL    = codistributed.build(LLloc,codistr,'noCommunication');
    UU    = codistributed.build(UUloc,codistr,'noCommunication');
    Pp    = codistributed.build(Pploc,codistr,'noCommunication');
    Qp    = codistributed.build(Qploc,codistr,'noCommunication');
    Rr    = codistributed.build(Rrloc,codistr,'noCommunication');
    dH    = codistributed.build(dHloc,codistr,'noCommunication');
end

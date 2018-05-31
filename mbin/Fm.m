function D = Fm(m,Q,model)
% Frequency domain FD modeling operator on multiple models
%
% use:
%   D = F(m,Q,model)
% input:
%   m                 - Matrix with gridded squared slowness in [km^2/s^2],
%   column represents the number of sample in a training network
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
ot = model.o-model.nb(1,:).*model.d;
dt = model.d;
nt = model.n+2*model.nb(1,:);
[zt,xt] = odn2grid(ot,dt,nt);

% data size
nsrc   = size(Q,2);
nrec   = length(model.zrec)*length(model.xrec);
nfreq  = length(model.freq);

% define wavelet
w = exp(1i*2*pi*model.freq*model.t0);
if model.f0
    % Ricker wavelet with peak-frequency model.f0
    w = (model.freq).^2.*exp(-(model.freq/model.f0).^2).*w;
end

% mapping from source/receiver/physical grid to comp. grid
Pr = opKron(opLInterp1D(xt,model.xrec),opLInterp1D(zt,model.zrec));
Ps = opKron(opLInterp1D(xt,model.xsrc),opLInterp1D(zt,model.zsrc));
Px = opKron(opExtension(model.n(2),model.nb(1,2)),opExtension(model.n(1),model.nb(1,1)));
% model parameter: slowness [s/m] on computational grid.
mu = Px*m;

% distribute frequencies according to standard distribution
freq = distributed(model.freq);
w    = distributed(w);
spmd
    codistr  = codistributor1d(2,[],[nsrc*nrec,nfreq,model.nsamples]);
    freqloc  = getLocalPart(freq);
    wloc     = getLocalPart(w);
    nfreqloc = length(freqloc);
    Dloc     = zeros(nrec*nsrc,nfreqloc,model.nsamples);
    for i = 1:model.nsamples
        for k = 1:nfreqloc
            Hk        = Helm2D_opt(mu(:,i),dt,nt,model.nb,model.unit,freqloc(k),model.f0);
            Uk        = Hk\(wloc(k)*(Ps'*Q));
            Dloc(:,k,i) = vec(Pr*Uk);
        end
    end
    D = codistributed.build(Dloc,codistr,'noCommunication');
end

% vectorize output, gather if needed
D = vec(D);

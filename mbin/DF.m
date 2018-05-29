function output = DF(m,Q,input,flag,model)
% Frequency domain modeling in the Born approximation. This is the
% Jacobian of F(m,Q,model).
%
% use:
%   output = DF(m,Q,input,flag,model,{gather})
% input:
%   m                 - vector with gridded squared slowness in [km^2/s^2]
%   Q                 - source matrix. size(Q,1) must match source grid
%                       definition, size(Q,2) determines the number of
%                       sources, if size(Q,3)>1, it represents a
%                       frequency-dependent source and has to be
%                       distributed over the last dimension.
%   input             - flag= 1: vector with gridded slowness perturbation
%                       flag=-1: vectorized data cube of size nrec xnrec x nfreq
%   flag              -  1: forward mode
%                       -1: adjoint mode
%   model.{o,d,n}     - regular grid: z = ox(1) + [0:nx(1)-1]*dx(1), etc
%   model.nb          - number of extra points for absorbing boundary on each side
%   model.freq        - frequencies
%   model.f0          - peak frequency of Ricker wavelet, 0 for no wavelet.
%   model.t0          - phase shift [s] of wavelet.
%   model.{zsrc,xsrc} - vectors describing source array
%   model.{zrec,xrec} - vectors describing receiver array



if nargin < 6
    dogather = 0;
end

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

if flag==1
    % solve Helmholtz for each frequency in parallel
    spmd
        codistr   = codistributor1d(2,codistributor1d.unsetPartition,[nsrc*nrec,nfreq]);
        freqloc   = getLocalPart(freq);
        wloc      = getLocalPart(w);
        nfreqloc  = length(freqloc);
        outputloc = zeros(nsrc*nrec,nfreqloc);
            for k = 1: nfreqloc
                [Hk, dHk] = Helm2D_opt(mu,dt,nt,model.nb,model.unit,freqloc(k),model.f0);
                U0k       = Hk\(wloc(k)*(Ps'*Q));
                Sk        = -(dHk*(U0k.*repmat(Px*input,1,nsrc)));
                U1k       = Hk\Sk;
                outputloc(:,k) = vec(Pr*U1k);
            end

        output = codistributed.build(outputloc,codistr,'noCommunication');
    end
    output = vec(output);
else
    spmd
        freqloc   = getLocalPart(freq);
        wloc      = getLocalPart(w);
        nfreqloc  = length(freqloc);
        outputloc = zeros(prod(model.n),1);
        inputloc  = getLocalPart(input);
            for k = 1:nfreqloc
                inputloc  = reshape(inputloc,[nsrc*nrec,nfreqloc]);
                [Hk, dHk] = Helm2D_opt(mu,dt,nt,model.nb,model.unit,freqloc(k),model.f0);
                U0k       = Hk\(wloc(k)*(Ps'*Q));
                Sk        = -Pr'*reshape(inputloc(:,k),[nrec nsrc]);
                V0k       = Hk'\Sk;
                r         = real(sum(conj(U0k).*(dHk'*V0k)),2));
                outputloc = outputloc + Px'*r;
            end
        output = pSPOT.utils.global_sum(outputloc);
    end
    output = output{1};
end

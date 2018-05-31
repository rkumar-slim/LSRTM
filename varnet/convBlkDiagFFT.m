classdef convBlkDiagFFT < convKernel 
    % classdef convBlkDiagFFT < convKernel
    %
    % 2D convolutions applied to all channels (no coupling). Computed using FFTs
    %
    % Transforms feature using affine linear mapping
    %
    %     Y(theta,Y0) =  K(theta_1) * Y0 
    %
    %  where 
    % 
    %      K - convolution matrix (computed using FFTs for periodic bc)
    
    properties
        S 
    end
    
    methods
        function this = convBlkDiagFFT(varargin)
            this@convKernel(varargin{:});
            if nargout==0 && nargin==0
                this.runMinimalExample;
                return;
            end
            this.S = gpuVar(this.useGPU, this.precision, getEigs(this));
            
        end
        function S = getEigs(this)
            S = zeros(prod(this.nImg),prod(this.sK(1:2)));
            for k=1:prod(this.sK(1:2))
                Kk = zeros(this.sK(1:2));
                Kk(k) = 1;
                Ak = getConvMatPeriodic(Kk,[this.nImg 1]);
                
                S(:,k) = vec(fft2( reshape(full(Ak(:,1)),this.nImg(1:2)) ));
            end
        end
        function this = gpuVar(this,useGPU,precision)
            if strcmp(this.precision,'double') && (isa(gather(this.S),'single'))
                this.S = getEigs(this);
            end
            this.S = gpuVar(useGPU,precision,this.S);
        end
        
        function runMinimalExample(~)
            nImg   = [16 16];
            sK     = [3 3,2];
            kernel = feval(mfilename,nImg,sK);
            theta1 = rand(sK); 
            theta1(:,1,:) = -1; theta1(:,3,:) = 1;
            theta  = theta1(:);

            I  = rand(nImgIn(kernel)); I(4:12,4:12,:) = 2;
            Ik = reshape(Amv(kernel,theta,I),kernel.nImgOut());
            figure(1); clf;
            subplot(1,2,1);
            imagesc(I(:,:,1));
            title('input');
            
            subplot(1,2,2);
            imagesc(Ik(:,:,1));
            title('output');
        end
        
        function Y = Amv(this,theta,Y)
            nex   = numel(Y)/prod(nImgIn(this));
            
            % compute convolution
            theta    = reshape(theta, [prod(this.sK(1:2)),this.sK(3)]);
            Yh = ifft2(reshape(Y,[nImgIn(this) nex]));

            Sk = reshape(this.S*theta,nImgIn(this));
            T  = Sk .* Yh;
            Y = real(fft2(T));
            Y  = reshape(Y,[],nex);
        end
        
        function Z = implicitTimeStep(this,theta,Y,h)
           % A = F*diag(S*theta)*F'
           % inv(h*ATA + I) = F*diag(1./(1+h*abs(S*theta)^2))*F'
           %
           nex   = numel(Y)/prod(nImgIn(this));
             
           % compute convolution
           theta    = reshape(theta, [prod(this.sK(1:2)),this.sK(3)]);
           Yh = ifft2(reshape(Y,[nImgIn(this) nex]));

           Sk = reshape(this.S*theta,nImgIn(this));
           T  = Yh./(h*abs(Sk).^2+1);
           Z = real(fft2(T));
           Z  = reshape(Z,[],nex);
        end

        
        function Z = ATmv(this,theta,Z)
            nex =  numel(Z)/prod(nImgOut(this));
            theta    = reshape(theta, [prod(this.sK(1:2)),this.sK(3)]);
            
            Yh = fft2(reshape(Z,[this.nImgOut nex]));
  
            Sk = reshape(this.S*theta,nImgOut(this));
            T  = Sk.*Yh;
            Z = real(ifft2(T));
            Z = reshape(Z,[],nex);
        end
        
        function dY = Jthetamv(this,dtheta,~,Y,~)
            nex    =  numel(Y)/nFeatIn(this);
            Y      = reshape(Y,[],nex);
            dY     = getOp(this,dtheta)*Y;
        end
        
        function dtheta = JthetaTmv(this,Z,~,Y,~)
            %  derivative of Z*(A(theta)*Y) w.r.t. theta    
            nex    =  numel(Y)/nFeatIn(this);
            
            % F*(diag(F'*Y)*(S*theta))
            % JTZ = S'*diag(conj(F'*Y))*F'*Z
            
            nI = this.nImgOut;
            Zh = ifft2(reshape(Z,[nI nex]));
            Yh = ifft2(reshape(Y,[nI nex]));
            T  = reshape(conj(Yh).*Zh,prod(nI(1:2)),nI(3)*nex); 
            Sk = reshape(this.S'*T,size(this.S,2),nI(3),nex);
            dtheta = real(sum(Sk,3))*prod(nI(1:2));
 
        end
        
        function n = nImgOut(this)
           n = [this.nImg(1:2)./this.stride this.sK(3)];
        end
 
        function theta = initTheta(this)
           theta = randn(this.sK);
        end
        

    end
end



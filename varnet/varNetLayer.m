classdef varNetLayer < abstractMeganetElement
    % classdef varNetLayer < abstractMeganetElement
    %
    % Implementation of variational network layer
    %
    % Y(theta,Y0) = - A'*(A*Y0-b) - K(th1)'(activation( K(th1)*Y0 + trafo.Bin*th2)))
    %
    
    
    properties
        A              % forward operator
        b              % data
        activation     % activation function
        K              % Kernel model, e.g., convMod
        Bin            % Bias inside the nonlinearity
        useGPU
        precision
        storeInterm   % flag for storing intermediates
    end
    methods
        function this = varNetLayer(A,b,K,varargin)
            if nargin==0
                help(mfilename)
                return;
            end
            useGPU    = [];
            precision = [];
            Bin       = [];
            activation = @tanhActivation;
            storeInterm=0;
            for k=1:2:length(varargin)     % overwrites default parameter
                eval([varargin{k},'=varargin{',int2str(k+1),'};']);
            end
            if not(isempty(useGPU))
                K.useGPU = useGPU;
            end
            if not(isempty(precision))
                K.precision = precision;
            end
            
            
            this.activation = activation;
            this.K = K;
            if not(exist('Bin','var')) || isempty(Bin)
                Bin = zeros(nFeatOut(K),0);
            end
            this.storeInterm=storeInterm;
            
            [this.A,this.b,this.Bin] = gpuVar(this.K.useGPU,this.K.precision,A,b,Bin);
        end
        
        function [th1,th2] = split(this,theta)
            th1 = theta(1:nTheta(this.K));
            cnt = numel(th1);
            th2 = theta(cnt+(1:size(this.Bin,2)));
        end
        
        function [Z,QZ,tmp] = apply(this,theta,Y,varargin)
            
            QZ =[]; tmp =  cell(1,2);
            nex        = numel(Y)/nFeatIn(this);
            Y          = reshape(Y,[],nex);
            storedAct  = (nargout>1);
            
            % add the data term
            res = this.A*Y - this.b;
            Z1  = -this.A'*res;

            % regularizer
            [th1,th2] = split(this,theta);
            Kop    = getOp(this.K,th1);
            Y     = Kop*Y;
            if this.storeInterm
                tmp{1}    = Y;
            end
            if not(isempty(th2))
                Y     = Y + this.Bin*th2;
            end
            Z      = this.activation(Y,'doDerivative',storedAct);
            Z      = -(Kop'*Z);
            
            Z = Z + Z1;
        end

        
        function [A,dA,KY,KZ] = getTempsForSens(this,theta,Y,tmp)
            % re-computes temp variables needed for sensitivity computations
            %
            % Input:
            %   theta - current weights
            %   Y     - input features
            %   tmp    - either {K*Y,-K'*Z} stored during apply or empty
            %
            % Output:
            %   dA    - derivative of activation
            %   KY    - K(theta)*Y
            %   tmpNL - temp results of norm Layer
            
            nex = numel(Y)/nFeatIn(this);
            KZ = [];
            [th1, th2]  = split(this,theta);
            
            if not(this.storeInterm)
                Y = reshape(Y,[],nex);
                KY = getOp(this.K,th1)*Y;
            else
                KY = tmp{1};
            end
            
                KYn = KY;
            if not(isempty(th2))
                KYn = KYn + this.Bin*th2;
            end
            [A,dA] = this.activation( KYn );
        end
        
        function n = nTheta(this)
            n = nTheta(this.K) + size(this.Bin,2);
   
        end
        
        function n = nFeatIn(this)
            n = nFeatIn(this.K);
        end
        
        function n = nFeatOut(this)
            n = nFeatIn(this.K);
        end
        
        function n = nDataOut(this)
            n = nFeatIn(this);
        end
        
        function theta = initTheta(this)
            theta = [vec(initTheta(this.K)); ...
                     0.1*ones(size(this.Bin,2),1)];
           
          
        end
        
        function dY = Jthetamv(this,dtheta,theta,Y,KY)
            
            [th1, ]  = split(this,theta);
            [dth1,dth2] = split(this,dtheta);
            
            [A,dA,KY,KZ] = getTempsForSens(this,theta,Y,KY);
            
            Kop    = getOp(this.K,th1);
            dKop   = getOp(this.K,dth1);
            dY     = dKop*Y;
            dY     = dY + this.Bin*dth2;
            
            dY = -(Kop'*(dA.*dY) + dKop'*A);
        end
        
        function dZ = JYmv(this,dY,theta,Y,KY)
            nex       = numel(Y)/nFeatIn(this);
            Y   = reshape(Y,[],nex);
            [th1]  = split(this,theta);
            
            [A,dA,KY,KZ] = getTempsForSens(this,theta,Y,KY);
            nex = numel(dY)/nFeatIn(this);
            dY  = reshape(dY,[],nex);
            
            % derivative of misfit term
            dZ1 = -this.A'*(this.A*dY);
            
            % derivative of regularizer
            Kop = getOp(this.K,th1);
            dY = Kop*dY;
            dZ = -(Kop'*(dA.*dY));
            
            dZ = dZ1 + dZ;
        end
        
        function dY = Jmv(this,dtheta,dY,theta,Y,KY)
            [th1]  = split(this,theta);
            [dth1,dth2] = split(this,dtheta);
            
            [A,dA,KY,KZ] = getTempsForSens(this,theta,Y,KY);
            
            nex = numel(Y)/nFeatIn(this);
            Kop    = getOp(this.K,th1);
            dKop   = getOp(this.K,dth1);
            if numel(dY)>1
                dY  = reshape(dY,[],nex);
                KdY = Kop*dY;
                dY1 = - this.A'*(this.A*dY);
            else
                KdY = 0;
                dY1 = 0;
            end
            dY = dKop*Y+KdY;
            dY     = dY + this.Bin*dth2;
            dY = -(Kop'*(dA.*dY) + dKop'*A);
            dY = dY1 + dY;
        end
        
        
        function dtheta = JthetaTmv(this,Z,~,theta,Y,KY)
            [th1]  = split(this,theta);
            [A,dA,KY,KZ] = getTempsForSens(this,theta,Y,KY);
            
            nex       = numel(Y)/nFeatIn(this);
            Z         = reshape(Z,[],nex);
            Kop       = getOp(this.K,th1);
            
            dAZ       = dA.*(Kop*Z);
            dth2      = vec(sum(this.Bin'*dAZ,2));
            dth1      = JthetaTmv(this.K,dAZ,[],Y);
            dth1      = dth1 + JthetaTmv(this.K,A,[],Z);
            dtheta    = [-dth1(:); -dth2(:); ];
        end
        
        function dY = JYTmv(this,Z,~,theta,Y,KY)
            [th1]  = split(this,theta);
            [A,dA,KY,KZ] = getTempsForSens(this,theta,Y,KY);
            
            nex       = numel(Y)/nFeatIn(this);
            Z         = reshape(Z,[],nex);
            
            % misfit
            dY1 = -this.A'*(this.A*Z);
            
            % regularizer
            Kop       = getOp(this.K,th1);
            dAZ       = dA.*(Kop*Z);
            dY  = dY1 - (Kop'*dAZ);
        end
        
        function [dtheta,dY] = JTmv(this,Z,~,theta,Y,KY,doDerivative)
            if not(exist('doDerivative','var')) || isempty(doDerivative)
                doDerivative =[1;0];
            end
            [th1]  = split(this,theta);
            [A,dA,KY,KZ] = getTempsForSens(this,theta,Y,KY);
            
            dY = [];
            nex       = numel(Y)/nFeatIn(this);
            Z         = reshape(Z,[],nex);
            Kop       = getOp(this.K,th1);
            
            % misfit
            dY1  = -this.A'*(this.A*Z);
            
            dAZ       = dA.*(Kop*Z);
            dth2      = vec(sum(this.Bin'*dAZ,2));
            dth1      = JthetaTmv(this.K,dAZ,[],Y);
            
            dth1      = dth1 + JthetaTmv(this.K,A,[],Z);
            dtheta    = [-dth1(:); -dth2(:)];
            
            if nargout==2 || doDerivative(2)==1
                dY  = dY1 -(Kop'*dAZ);
            end
            if nargout==1 && all(doDerivative==1)
                dtheta = [dtheta(:);dY(:)];
            end
        end
        
        
        % ------- functions for handling GPU computing and precision ---- 
        function this = set.useGPU(this,value)
            if (value~=0) && (value~=1)
                error('useGPU must be 0 or 1.')
            else
                this.K.useGPU  = value;
                [this.Bin,this.Bout] = gpuVar(value,this.precision,this.Bin,this.Bout);
            end
        end
        function this = set.precision(this,value)
            if not(strcmp(value,'single') || strcmp(value,'double'))
                error('precision must be single or double.')
            else
                this.K.precision = value;
                [this.Bin,this.Bout] = gpuVar(this.useGPU,value,this.Bin,this.Bout);
            end
        end
        function useGPU = get.useGPU(this)
            useGPU = this.K.useGPU;
            
        end
        function precision = get.precision(this)
            precision = this.K.precision;
        end

    end
end



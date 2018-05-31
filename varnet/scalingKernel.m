classdef scalingKernel
    % classdef scalingKernel < handle
    % 
    % scaling
    %
    %   Y(theta,Y0)  = theta*Y0,    theta is scalar!
    %
    
    properties
        nK
        useGPU
        precision
    end
    
    methods
        function this = scalingKernel(nK,varargin)
            this.nK = nK;
            useGPU  = 0;
            precision = 'double';
            for k=1:2:length(varargin)     % overwrites default parameter
                eval([ varargin{k},'=varargin{',int2str(k+1),'};']);
            end
            this.useGPU = useGPU;
            this.precision = precision;
            
        end
        function this = gpuVar(this,useGPU,precision)
        end
        function this = set.useGPU(this,value)
            if (value~=0) && (value~=1)
                error('useGPU must be 0 or 1.')
            else
                this.useGPU = value;
            end
        end
        function this = set.precision(this,value)
            if not(strcmp(value,'single') || strcmp(value,'double'))
                error('precision must be single or double.')
            else
                this.precision = value;
            end
        end
        
        function n = nTheta(this)
            n= 1;
        end
        
        function n = nFeatIn(this)
            n = this.nK(2);
        end
        
        function n = nFeatOut(this)
            n = this.nK(1);
        end
        
        function theta = initTheta(this)
            theta = 1;
        end
            
        function A = getOp(this,theta)
            Amv = @(x) theta*x;
            A = LinearOperator(this.nK(1),this.nK(1),Amv,Amv);
        end
        
        function dY = Jthetamv(this,dtheta,~,Y,~)
            dY = dtheta*Y;
        end
        
        
       function dtheta = JthetaTmv(this,Z,~,Y,~)
            % Jacobian transpose matvec.
            nex    =  numel(Y)/nFeatIn(this);
            Y      = reshape(Y,[],nex);
            Z      = reshape(Z,[],nex);
            dtheta   = sum(vec(Y.*Z));
       end
        
           

    end
end


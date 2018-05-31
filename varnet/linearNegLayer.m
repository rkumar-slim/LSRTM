classdef linearNegLayer < abstractMeganetElement
    % classdef doubleSymLayer < abstractMegaNetElement
    %
    % Implementation of symmetric double layer model
    %
    %
    properties
        K              % Kernel model, e.g., convMod
        useGPU
        precision
    end
    methods
        function this = linearNegLayer(K,varargin)
            if nargin==0
                help(mfilename)
                return;
            end
            useGPU    = [];
            precision = [];
            for k=1:2:length(varargin)     % overwrites default parameter
                eval([varargin{k},'=varargin{',int2str(k+1),'};']);
            end
            if not(isempty(useGPU))
                K.useGPU = useGPU;
            end
            if not(isempty(precision))
                K.precision = precision;
            end
            
            this.K = K;
        end
        
        function [Z,Yt] = apply(this,theta,Y,varargin)
            nex    = numel(Y)/nFeatIn(this);
            Y      = reshape(Y,[],nex);
            
            Kop    = getOp(this.K,theta);
            Yt     = Kop * Y; 
            Z      = -(Kop'*Yt);
        end
        
        function n = nTheta(this)
            n = nTheta(this.K);
        end
        
        function n = nFeatIn(this)
            n = nFeatIn(this.K);
        end
        
        function n = nFeatOut(this)
            n = nFeatIn(this.K);
        end
        
        function theta = initTheta(this)
           theta = randn(nTheta(this),1);
        end
        
        function [dZ] = Jthetamv(this,dtheta,theta,Y,Yt)
            % Yt = K*Y
            % A = Yt, dA = 1;
            
            Kop    = getOp(this.K,theta);
            dKop   = getOp(this.K,dtheta);
            dZ     = dKop*Y;
            
            dZ = -(Kop'*dZ + dKop'*Yt);
        end
        
        function [dZ] = JYmv(this,dY,theta,~,~)
            %[~,dA] = this.activation(Yt);
            nex = numel(dY)/nFeatIn(this);
            dY  = reshape(dY,[],nex);
            
            Kop = getOp(this.K,theta);
            dZ = -(Kop'*(Kop*dY));            
        end
                
        function dtheta = JthetaTmv(this,Z,theta,Y,Yt)
            nex       = numel(Y)/nFeatIn(this);
            Z         = reshape(Z,[],nex);
            Kop       = getOp(this.K,theta);
            %[A,dA]    = this.activation(Yt);
            
            dAZ       = Kop*Z;
            dtheta    = JthetaTmv(this.K,dAZ,[],Y);
            
            dtheta    = -(dtheta + JthetaTmv(this.K,Yt,[],Z));
        end
        
        function dY = JYTmv(this,Z,theta,Y,~)
            nex       = numel(Y)/nFeatIn(this);
            Z         = reshape(Z,[],nex);
            Kop       = getOp(this.K,theta);
 
            dAZ       = Kop*Z;
            dY        = -(Kop'*dAZ);
        end
        
        %------- Inverse function for time stepping --------------------
        function[Yt] = applyInv(this,theta,Y,h)
            nex    = numel(Y)/nFeatIn(this);
            Y      = reshape(Y,[],nex);
            n      = size(Y,1);
            
            %Kop    = getOp(this.K,theta);
            Yt     = implicitTimeStep(this.K,theta,Y,h);
            %Yt     = (h*(Kop'*Kop) + speye(n))\Y; 
        end

       function [dZ] = iJthetamv(this,dtheta,theta,Y,h)
       % derivative of the inverse function
            nex    = numel(Y)/nFeatIn(this);
            Y      = reshape(Y,[],nex);

            Kop    = getOp(this.K,theta);
            dKop   = getOp(this.K,dtheta);
            %T      = h*(Kop'*Kop) + speye(size(Kop,2));
            W        = implicitTimeStep(this.K,theta,Y,h);
            Wt       =  dKop'*(Kop*W) + Kop'*(dKop*W);
            dZ       = implicitTimeStep(this.K,theta,Wt,h);
            dZ       = -h*dZ;
            %dZ     = -h * (T\ (dKop'*(Kop*W) + Kop'*(dKop*W)));
            
       end

       
      function dY = iJYmv(this,Z,theta,~,~,h)
            nex       = numel(Z)/nFeatIn(this);
            Z         = reshape(Z,[],nex);

            dY        = implicitTimeStep(this.K,theta,Z,h);
      end
      
      function dY = iJmv(this,dtheta,dY,theta,Y,h)
          Z1 = iJYmv(this,dY,theta,[],[],h);
          Z2 = iJthetamv(this,dtheta,theta,Y,h);
          dY = Z1+Z2;
      end
      
      function dY = iJYTmv(this,Z,theta,~,~,h)
            dY = iJYmv(this,Z,theta,[],[],h);
      end
      
      function dY     = iJthetaTmv(this,Z,theta,Y,~,h)
            nex       = numel(Y)/nFeatIn(this);
            Z         = reshape(Z,[],nex);
            
            Kop       = getOp(this.K,theta);

            TiY        = implicitTimeStep(this.K,theta,Y,h);
            TiZ        = implicitTimeStep(this.K,theta,Z,h);
            
            th1       = JthetaTmv(this.K,Kop*TiZ,[],TiY,[]);
            th2       = JthetaTmv(this.K,Kop*TiY,[],TiZ,[]);
            dY        = -h*(th1+th2);
            %dY =  -h*Kop*(TiZ*TiY' + TiY*TiZ'); 

      end
        
      function [dtheta,dY] = iJTmv(this,Z,theta,Y,h)
          dY     = iJYTmv(this,Z,theta,[],[],h);
          dtheta = iJthetaTmv(this,Z,theta,Y,[],h);
      end


        % ------- functions for handling GPU computing and precision ---- 
        function this = set.useGPU(this,value)
            if (value~=0) && (value~=1)
                error('useGPU must be 0 or 1.')
            else
                this.K.useGPU  = value;
            end
        end
        function this = set.precision(this,value)
            if not(strcmp(value,'single') || strcmp(value,'double'))
                error('precision must be single or double.')
            else
                this.K.precision = value;
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

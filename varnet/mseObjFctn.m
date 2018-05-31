classdef mseObjFctn < objFctn
    % classdef mseObjFctn < objFctn
    %
    % mean squared error for output of deep neural networks 
    %
    % J(theta) = 1/(2*n)* | Y(theta) - Ytrue | + Rtheta(theta)
    %
    
    properties
        net
        pRegTheta
        Y0
        Ytrue
        useGPU
        precision
    end
    
    methods
        function this = mseObjFctn(net,pRegTheta,Y0,Ytrue,varargin)
            if nargout==0 && nargin==0
                this.runMinimalExample;
                return;
            end
            useGPU = [];
            precision = [];
            for k=1:2:length(varargin)     % overwrites default parameter
                eval([varargin{k},'=varargin{',int2str(k+1),'};']);
            end
            this.net    = net;
            this.pRegTheta = pRegTheta;
            
            if not(isempty(useGPU))
                this.useGPU = useGPU;
            end
            
            if not(isempty(precision))
                this.precision=precision;
            end
            [Y0,Ytrue] = gpuVar(this.useGPU,this.precision,Y0,Ytrue);
            this.Y0         = Y0;
            this.Ytrue      = Ytrue;
        end
        
        function [Jc,para,dJ,H,PC] = eval(this,theta,idx)
            if not(exist('idx','var')) || isempty(idx)
                Y0    = this.Y0;
                Ytrue = this.Ytrue;
            else
                Y0    = this.Y0(:,idx);
                Ytrue = this.Ytrue(:,idx);
            end
            compGrad = nargout>2;
            compHess = nargout>3;
            
            dJ = [];  H = []; PC = [];
            nex = size(Y0,2);
            
            % evaluate loss
            if compGrad
                [YN,J] = linearizeTheta(this.net,theta,Y0);
                res    = YN-Ytrue;
                F      = (.5/nex)*norm(res,'fro')^2;
                dJ     = (J'*res)/nex;
                Jc   = F;
                if compHess
                    Hthmv = @(x) (J'*(J*x))/nex; %  JTmv(this.net, reshape(d2YF* Jmv(this.net,x,[],Kb,Yall,dA),size(YN)), Kb,Yall,dA);
                    H   = LinearOperator(numel(theta),numel(theta),Hthmv,Hthmv);
                end
            else
                YN     = apply(this.net,theta,Y0);
                res    = YN-Ytrue;
                F      = (.5/nex)*norm(res,'fro')^2;
                Jc    = F;
            end
            para = struct('F',F);

            % evaluate regularizer for DNN weights             
            if not(isempty(this.pRegTheta))
                [Rth,hisRth,dRth,d2Rth] = regularizer(this.pRegTheta,theta);
                Jc = Jc + Rth;
                if compGrad
                    dJ = dJ + dRth;
                end
                if compHess
                       H  = H + d2Rth;
                end
                para.Rth = Rth;
                para.hisRth = hisRth;
            end
            
            if nargout>4
                PC = getPC(this.pRegTheta);
            end
        end
        
        function [str,frmt] = hisNames(this)
            str = {'mse'};
            frmt= {'%-12.2e'};
            if not(isempty(this.pRegTheta))
                [s,f] = hisNames(this.pRegTheta);
                s{1} = [s{1} '(theta)'];
                str  = [str, s{:}];
                frmt = [frmt, f{:}];
            end
        end
        
        function his = hisVals(this,para)
            his = para.F;
            if not(isempty(this.pRegTheta))
                his = [his, hisVals(this.pRegTheta,para.hisRth)];
            end
        end
        
        function str = objName(this)
            str = 'mseObjFun';
        end
        
        % ------- functions for handling GPU computing and precision ----
        function this = set.useGPU(this,value)
            if isempty(value)
                return
            elseif(value~=0) && (value~=1)
                error('useGPU must be 0 or 1.')
            else
                if not(isempty(this.net)); this.net.useGPU       = value; end
                if not(isempty(this.pRegTheta)); this.pRegTheta.useGPU       = value; end
                
                [this.Y0,this.Ytrue] = gpuVar(value,this.precision,...
                                                         this.Y0,this.Ytrue);
            end
        end
        function this = set.precision(this,value)
            if isempty(value)
                return
            elseif not(strcmp(value,'single') || strcmp(value,'double'))
                error('precision must be single or double.')
            else
                if not(isempty(this.net)); this.net.precision       = value; end
                if not(isempty(this.pRegTheta)); this.pRegTheta.precision       = value; end
                
                [this.Y0,this.Ytrue] = gpuVar(this.useGPU,value,...
                                                         this.Y0,this.Ytrue);
            end
        end
        function useGPU = get.useGPU(this)
                useGPU = -ones(2,1);
                
                if not(isempty(this.net)) && not(isempty(this.net.useGPU))
                    useGPU(1) = this.net.useGPU;
                end
                if not(isempty(this.pRegTheta)) && not(isempty(this.pRegTheta.useGPU))
                    useGPU(2) = this.pRegTheta.useGPU;
                end
                
                useGPU = useGPU(useGPU>=0);
                if all(useGPU==1)
                    useGPU = 1;
                elseif all(useGPU==0)
                    useGPU = 0;
                else
                    error('useGPU flag must agree');
                end
        end
        function precision = get.precision(this)
            isSingle    = -ones(2,1);
            isSingle(1) = strcmp(this.net.precision,'single');
            if not(isempty(this.pRegTheta)) && not(isempty(this.pRegTheta.precision))
                isSingle(2) = strcmp(this.pRegTheta.precision,'single');
            end
            isSingle = isSingle(isSingle>=0);
            if all(isSingle==1)
                precision = 'single';
            elseif all(isSingle==0)
                precision = 'double';
            else
                error('precision flag must agree');
            end

        end
        function runMinimalExample(~)
            
            nex    = 100; nf =2;
            
            blocks    = cell(2,1);
            blocks{1} = NN({singleLayer(dense([2*nf nf]))});
            blocks{2} = ResNN(doubleLayer(dense([2*nf 2*nf]),dense([2*nf 2*nf])),2,.1);
            net    = Meganet(blocks);
            nth = nTheta(net);
            theta  = randn(nth,1);
            
            % training data
            Y = randn(nf,nex);
            C = zeros(nf,nex);
            C(1,Y(2,:)>Y(1,:).^2) = 1;
            C(2,Y(2,:)<=Y(1,:).^2) = 1;
            
            % validation data
            Yv = randn(nf,nex);
            Cv = zeros(nf,nex);
            Cv(1,Yv(2,:)>Yv(1,:).^2) = 1;
            Cv(2,Yv(2,:)<=Yv(1,:).^2) = 1;
            
            
            pLoss = regressionLoss();
            W = vec(randn(2,2*nf+1));
            pRegW  = tikhonovReg(opEye(numel(W)));
            pRegTheta    = tikhonovReg(opEye(numel(theta)));
            
            fctn = dnnObjFctn(net,pRegTheta,pLoss,pRegW,Y,C);
            fval = dnnObjFctn(net,[],pLoss,[],Yv,Cv);
            % [Jc,para,dJ,H,PC] = fctn([Kb(:);W(:)]);
            % checkDerivative(fctn,[Kb(:);W(:)])
            
            opt1  = newton('out',1,'maxIter',20);
            opt2  = sd('out',1,'maxIter',20);
            opt3  = nlcg('out',1,'maxIter',20);
            [KbWopt1,His1] = solve(opt1,fctn,[theta(:); W(:)],fval);
            [KbWopt2,His2] = solve(opt2,fctn,[theta(:); W(:)],fval);
            [KbWopt3,His3] = solve(opt3,fctn,[theta(:); W(:)],fval);
            
            figure(1); clf;
            subplot(1,3,1);
            semilogy(His1.his(:,2)); hold on;
            semilogy(His2.his(:,2)); 
            semilogy(His3.his(:,2)); hold off;
            legend('newton','sd','nlcg');
            title('objective');

            subplot(1,3,2);
            semilogy(His1.his(:,4)); hold on;
            semilogy(His2.his(:,4)); 
            semilogy(His3.his(:,4)); hold off;
            legend('newton','sd','nlcg');
            title('opt.cond');

            subplot(1,3,3);
            plot(His1.his(:,10)); hold on;
            plot(His1.his(:,end),'--'); hold on;
            plot(His2.his(:,8)); 
            plot(His2.his(:,end),'--');
            plot(His3.his(:,8)); 
            plot(His3.his(:,end),'--'); hold off;
            legend('newton-train','newton-val','sd-train','sd-val','nlcg-train','nlcg-val');
            title('loss');
        end
    end
end











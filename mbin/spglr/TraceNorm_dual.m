function d = TraceNorm_dual(x,weights, params)

% dual of trace norm is operator norm i.e maximum singular value
if params.mode==1
    x = reshape(x,params.numr,params.numc);
    d = svds(gather(x),1);
elseif params.mode==2 % svd on full matrix
    
    x = reshape(x(1:params.mhnumr*params.mhnumc),params.mhnumr,params.mhnumc)+reshape(x(params.mhnumr*params.mhnumc+1:end),params.mhnumr,params.mhnumc);
    d = svds(gather(x),1);
    
elseif params.mode==3 % random SVD
    x = reshape(x,params.numr,params.numc);
    row_oversample = ceil(2*params.nr*log(params.nr));
    row_order = randperm(params.numr);
    ind = row_order(1:row_oversample);
    x = x(ind,:);
    d = svds(gather(x),1);
else % random SVD with irbl svd library
    x = reshape(x,params.numr,params.numc);
    row_oversample = ceil(2*params.nr*log(params.nr));
    row_order = randperm(params.numr);
    if length(row_order)>row_oversample
        ind = row_order(1:row_oversample);
        x = x(ind,:);
    end
    opts.K = params.svdnum;
    opts.tol = params.svdtol;
    d = irblsvds(x,opts);
end


function W = drawWeights(model,ssW)
%redraw or use predefined weights W for simultaneous sources
if (model.redraw == 1)
    W = randn(model.ns,model.nsim);
else
    W = ssW;
end
end


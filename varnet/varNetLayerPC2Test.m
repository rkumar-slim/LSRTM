classdef varNetLayerPC2Test < layerTest
    % classdef varNetLayerPCTest < layerTest
    %
    methods (TestClassSetup)
        function addTrafos(testCase)
            ks    = cell(0,1);
            A = randn(14,14);
            xtrue = randn(14,10);
            b = A*xtrue;
            M = linearNegLayer(dense([14 14]));
            K = dense([14 14]);
            ks{end+1} = varNetLayerPC(A,b,M,K,'Bin',randn(14,3));
            testCase.layers = ks;
        end
    end
end
classdef varNetLayerTest < layerTest
    % classdef varNetLayerTest < layerTest
    %
    methods (TestClassSetup)
        function addTrafos(testCase)
            ks    = cell(0,1);
            A = randn(14,14);
            xtrue = randn(14,10);
            b = A*xtrue;
            ks{end+1} = varNetLayer(A,b,dense([14 14]),'Bin',randn(14,3));
            testCase.layers = ks;
        end
    end
end
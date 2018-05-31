classdef scalingKernelTest < kernelTest
	% classdef denseTest < kernelTest
	%
	% tests some dense kernels. Extend to cover more cases.
	methods (TestClassSetup)
        function addKernels(testCase)
            ks    = cell(1,1);
            ks{1} = scalingKernel([24 24]);
            testCase.kernels = ks;
        end
    end
end
using Plots

using Flux: onecold

using .DataPreparation: vocabulary, loadSamples, unflatten

include("DataPreparation.jl")
function trainingPairMatrix(samples)
	# Fill with good and usefull code
end

# trainingSamples, evaluationSamples = loadSamples("samples\\backwards_n=6_samples=500000Samples.dat",
trainingSamples, evaluationSamples = loadSamples("samples\\forward_n=6_samples=10000Samples.dat",
	evaluationFraction = 0,
	maxInputLength     = 100, #parameterSpecification.nInputLength,
	maxOutputLength    = 100, #parameterSpecification.nOutputLength,
)

labelSizes  = [size(sample, 2) for sample in trainingSamples[1]]
targetSizes = [size(sample, 2) for sample in trainingSamples[2]]

histogram(labelSizes,  title="Label Sizes")
histogram(targetSizes, title="Target Sizes")

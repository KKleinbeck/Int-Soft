using Parameters: @with_kw
using Flux: crossentropy, onecold
using CuArrays: @cufunc

device = "gpu"
tpu(x) = device == "gpu" ? gpu(x) : cpu(x)

@with_kw mutable struct TrainingParameters
	# Optimizer and loss function Hyperparameters
	η::Float32        = 1e-3    # learning rate
	wPenalty::Float32 = 1e-3

	# Data and Training related Hyperparameters
	batchsize::Int = 1
	shuffle::Bool  = true
	throttle::Int  = 30
	nEpochs::Int   = 10
end

function misclassificationRate(ŷ, y)
	ŷ = unflatten(ŷ)
	y = unflatten(y)
	ŷ, y = onecold(ŷ), onecold(y)

	seqLength = length(y)
	ŷ = (k = findfirst(x -> x == 2, ŷ)) == nothing ? ŷ : ŷ[1:k-1] # Token `2` is the <EOS> token
	y = (k = findfirst(x -> x == 2, y)) == nothing ? y : y[1:k-1]

	minLength = min(length(ŷ), length(y))
	(reduce(+, ŷ[1:minLength] .!= y[1:minLength]) + abs(length(ŷ) - length(y))) / seqLength
end

function expressionIsEqual(ŷ, y)
	ŷ = unflatten(ŷ)
	y = unflatten(y)
	ŷ, y = onecold(ŷ), onecold(y)

	ŷ = (k = findfirst(x -> x == 2, ŷ)) == nothing ? ŷ : ŷ[1:k-1] # Token `2` is the <EOS> token
	y = (k = findfirst(x -> x == 2, y)) == nothing ? y : y[1:k-1]

	length(ŷ) != length(y) && return false
	all(ŷ .== y)
end

function evaluate(x, y, model)
	# assume flattened samples here.
	@info("Target:")
	y = onecold(length(size(y)) == 1 ? unflatten(y) : y)
	y = (k = findfirst(x -> x == 2, y)) == nothing ? y : y[1:k-1]
	@info(y)

	@info("The model results in:")
	ŷ = model(x)
	ŷ = onecold(length(size(ŷ)) == 1 ? unflatten(ŷ) : ŷ)
	ŷ = (k = findfirst(x -> x == 2, ŷ)) == nothing ? ŷ : ŷ[1:k-1] # Token `2` is the <EOS> token
	@info(ŷ)

	minLength = min(length(ŷ), length(y))
	@info("Number of mismatched tokens: $(
		reduce(+, ŷ[1:minLength] .!= y[1:minLength]) + abs(length(ŷ) - length(y))
	)")
end

function evaluationCallback(samples, model::Union{simpleEncoderDecoderCell})
	# Go to the CPU, since we are doing a lot of scalar operations here.
	# `[model(sample)...] == (length)` array vs. `model(sample) == (length, 1)` array
	results = [model(sample) for sample in eachcol(samples[1])] |> cpu
	targets = [sample for sample in eachcol(samples[2])] |> cpu
	_evaluationCallback(results, targets)
end
function evaluationCallback(samples, model::Union{recursiveAttentionCell})
	# Go to the CPU, since we are doing a lot of scalar operations here.
	results = [flatten(model(sample)) for sample in samples[1]] |> cpu
	targets = [flatten(sample) for sample in samples[2]] |> cpu
	_evaluationCallback(results, targets)
end

function _evaluationCallback(results, targets)
	protoLoss(ŷs, ys, loss) =
		sum([sum(loss(ŷ, y)) for (ŷ, y) in zip(ŷs, ys)]) / length(ys) # length(ys) == batchsize

	@info("Evaluation results:\n")
	namedLosses = [
		["Binary Cross-Entropy", (ŷ, y) -> binarycrossentropy.(ŷ, y)  / (length(y) / length(vocabulary))],
		["Kullback-Leibler Divergence", (ŷ, y) -> kldivergenceC(ŷ, y)  / (length(y) / length(vocabulary))],
		# ["L¹ Loss", mae],
		["Misclassification Rate", misclassificationRate],
		["Fraction of correct Results", expressionIsEqual]
	]
	for namedLoss in namedLosses
		@info(@sprintf("  %-30s %.5f", namedLoss[1], protoLoss(results, targets, namedLoss[2])))
	end
	@info("----------------------------------------")

	# TODO store results
end

# I define my own custom kldiv. Why? because in the current version of Flux it is not stable if y = 0
function kldivergenceC(ŷ, y)
	entropy = sum(xlogx.(y)) / size(y, 2)
	return entropy + crossentropy(ŷ, y)
end

function xlogx(x)
	result = x*log(x)
	iszero(x) ? zero(result) : result
end
@cufunc function xlogx(x)
  result = x * log(x)
  ifelse(iszero(x), zero(result), result)
end

function extractSample(samples, index)
	return samples[1][:,index], samples[2][:,index]
end

; # End of file

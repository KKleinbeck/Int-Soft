using Flux
using CuArrays
using Flux: binarycrossentropy, kldivergence, mae, throttle, reset!, @epochs
using LinearAlgebra # norm
using Printf

include("DataPreparation.jl")
using .DataPreparation

include("ModelZoo.jl")

include("Utils.jl")
using .Utils


# --------------------------------------------------------------------------------------------------
# Auxilliary functions and structures

device = "gpu"
tpu(x) = device == "gpu" ? gpu(x) : cpu(x)

function misclassificationRate(ŷ, y)
	ŷ = unflatten(ŷ)
	y = unflatten(y)
	seqLength = length(y)
	ŷ = (k = findfirst(x -> x == 2, ŷ)) == nothing ? ŷ : ŷ[1:k-1] # Token `2` is the <EOS> token
	y = (k = findfirst(x -> x == 2, y)) == nothing ? y : y[1:k-1]

	minLength = min(length(ŷ), length(y))
	(reduce(+, ŷ[1:minLength] .!= y[1:minLength]) + abs(length(ŷ) - length(y))) / seqLength
end

function expressionIsEqual(ŷ, y)
	ŷ = unflatten(ŷ)
	y = unflatten(y)
	ŷ = (k = findfirst(x -> x == 2, ŷ)) == nothing ? ŷ : ŷ[1:k-1] # Token `2` is the <EOS> token
	y = (k = findfirst(x -> x == 2, y)) == nothing ? y : y[1:k-1]

	length(ŷ) != length(y) && return false
	all(ŷ .== y)
end

function evaluate(x, y, model)
	# assume flattened samples here.
	@info("Target:")
	y = unflatten(y)
	y = (k = findfirst(x -> x == 2, y)) == nothing ? y : y[1:k-1]
	@info(y)

	@info("The model results in:")
	ŷ = unflatten(model(x))
	ŷ = (k = findfirst(x -> x == 2, ŷ)) == nothing ? ŷ : ŷ[1:k-1] # Token `2` is the <EOS> token
	@info(ŷ)

	minLength = min(length(ŷ), length(y))
	@info("Number of mismatched tokens: $(
		reduce(+, ŷ[1:minLength] .!= y[1:minLength]) + abs(length(ŷ) - length(y))
	)")
end


# --------------------------------------------------------------------------------------------------
# Loss function and training procedure

function evaluationCallback(samples, model)
	# Go to the CPU, since we are doing a lot of scalar operations here.
	results = [model(sample) for sample in eachcol(samples[1])] |> cpu
	targets = [sample for sample in eachcol(samples[2])] |> cpu

	protoLoss(ŷs, ys, loss) =
		sum([sum(loss(ŷ, y)) for (ŷ, y) in zip(ŷs, ys)]) / length(ys)

	@info("Evaluation results:\n")
	namedLosses = [
		["Binary Cross-Entropy", (ŷ, y) -> binarycrossentropy.(ŷ, y)  / size(y, 1)],
		["Kullback-Leibler Divergence", (ŷ, y) -> kldivergenceC(ŷ, y)  / size(y, 1)],
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

function lossTarget(x, y, model, params, args::Args)
	ŷ = model(x)

	# TODO BCE does not like matrix input: flatten output and stuff first
	# loss = sum(binarycrossentropy.(ŷ, y)) / (args.nOutputLength * args.batchsize)
	loss = sum(mae.(ŷ, y))
	return loss #+ args.wPenalty * sum(norm, params)
end
# TODO, the weigth decay results in NaN results, if present in the FIRST EPOCH (only if). Why?

function trainEpoch(model, trainingSamples, evaluationSamples, args::Args)
	# align each sample along first dimension and create batches.
	@info("Preparing Data, Optimizer and callbacks")
	trainingSamples = Flux.Data.DataLoader(trainingSamples[1], trainingSamples[2],
		batchsize = args.batchsize, shuffle = args.shuffle
	)

  # Initialize Hyperparameters
	hyperParams = params(model)

  loss(x, y) = lossTarget(x, y, model, hyperParams, args)

	evalcb = () -> evaluationCallback(evaluationSamples, model)

  opt = ADAM(args.η)
	# opt = RMSProp(args.η)
  @info("Training...")
	# @epochs args.nEpochs Flux.train!(loss, hyperParams, trainingSamples, opt, cb = throttle(evalcb, args.throttle))
	@epochs args.nEpochs Flux.train!(loss, hyperParams, trainingSamples, opt, cb = () -> @info("Another one bites the dust"))
  return model
end
# TODO: custom training function; add to progressbar (`@progress` infront of `for`) for juno etc.
# TODO: save snapshots of the evaluation with lowest misclassification rate


# --------------------------------------------------------------------------------------------------
# Main loop

parameterSpecification = Args(
	nVocab = length(vocabulary),
	nInputLength   = 10,
	nOutputLength  = 10,
	nContextLength = 256,

	η = 1e-3,
	wPenalty = 0.0,
	# wPenalty = 1e-3,

	nEpochs = 1,
	batchsize = 1
)

# trainingSamples, evaluationSamples = loadSamples("Samples\\backwards_n=6_samples=500000Samples.dat",
# trainingSamples, evaluationSamples = loadSamples("Samples\\backwards_n=4_samples=1000Samples.dat",
trainingSamples, evaluationSamples = loadSamples("Samples\\forward_n=6_samples=10000Samples.dat",
	evaluationFraction = 0.025,
	maxInputLength     = parameterSpecification.nInputLength,
	maxOutputLength    = parameterSpecification.nOutputLength,
	# dissallowedTokens = r"Csc|Sec|Sech|Csch"
	# flattenTo          = (parameterSpecification.nInputLength, parameterSpecification.nOutputLength),
	expandToMaxLength  = (false, true),
	type               = Float32
) #|> tpu

# model = simpleEncoderDecoder(parameterSpecification) |> tpu
model = recursiveAttentionModel(parameterSpecification) #|> tpu
# TODO make gpu's a possibility again

trainEpoch(model, trainingSamples, evaluationSamples, parameterSpecification)

# This code finds the NaN gradients
for (x, y) in Flux.Data.DataLoader(trainingSamples[1], trainingSamples[2], batchsize = 10, shuffle = false)
  tc = 0; ps = params(model)
  gs = gradient(ps) do
    tl = lossTarget(x, y, model, ps, parameterSpecification)
    tc = tl
    return tl
		# return sum(norm.(model(x)))
  end # gradient evaluation
  println("-----------------------------")
  # println(any(x -> x < 0 || x > 1 || isnan(x) || isinf(x), model(x)))
  println(tc)
  # println(gs[model[1].W][1:10])
  # println(model[1].W[1:10])
  opt = ADAM(0.001)
  Flux.update!(opt, ps, gs)
  # println(get!(opt.state, model[1].W[1], (0.0f0, 0.0f0, opt.beta)))
  # if isnan(gs[model[1].W][1])
    # break
  # end
end

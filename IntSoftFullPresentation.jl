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

include("Auxilliary.jl")

# --------------------------------------------------------------------------------------------------
# Loss function and training procedure

function evaluationCallback(samples, model)
	@error("Make me model independant fist")
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

"""
	lossTarget(x, y, model, params, tp)

	Two methods are given: one for flattened training samples (here batches are matrices) and
	unflattened training samples (batches are Arrays of Arrays).
"""
function lossTarget(x, y, model::Union{simpleEncoderDecoderCell}, params, tP::TrainingParameters)
	return _lossTarget(x, y, model, params, tp) / tp.batchsize
end
function lossTarget(xs, ys, model::Union{recursiveAttentionCell}, params, tP::TrainingParameters)
	return sum([_lossTarget(x, y, model, params, tp) for (x, y) in zip(xs, ys)]) / tp.batchsize
end

function _lossTarget(x, y, model, params, tP::TrainingParameters)
	ŷ = model(x)

	loss = sum(binarycrossentropy.(ŷ, y)) / (tP.nOutputLength)
	# loss = sum(mae.(ŷ, y))
	return loss #+ args.wPenalty * sum(norm, params)
end
# TODO, the weigth decay results in NaN results, if present in the FIRST EPOCH (only if). Why?

function trainEpoch(model, trainingSamples, evaluationSamples, args::TrainingParameters)
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

maxInputLength  = 10
maxOutputLength = 10

trainingParameters = TrainingParameters(
	η = 1e-3,
	wPenalty = 0.0, # wPenalty = 1e-3,

	nEpochs = 1,
	batchsize = 1
)

# trainingSamples, evaluationSamples = loadSamples("Samples\\backwards_n=6_samples=500000Samples.dat",
# trainingSamples, evaluationSamples = loadSamples("Samples\\backwards_n=4_samples=1000Samples.dat",
trainingSamples, evaluationSamples = loadSamples("Samples\\forward_n=6_samples=10000Samples.dat",
	evaluationFraction = 0.025,
	maxInputLength     = maxInputLength,
	maxOutputLength    = maxOutputLength,
	# dissallowedTokens = r"Csc|Sec|Sech|Csch"
	# flattenTo          = (maxInputLength, maxOutputLength),
	expandToMaxLength  = (false, true),
	type               = Float32
) #|> tpu

# model = simpleEncoderDecoder(parameterSpecification) |> tpu
model = recursiveAttentionModel(length(vocabulary), maxInputLength, maxOutputLength, [256];
	nEncoderIterations = 1, nDecoderIterations = 1,
	encoderInterFFDimension = 128, decoderInterFFDimension = 512
) #|> tpu
# TODO make gpu's a possibility again

trainEpoch(model, trainingSamples, evaluationSamples, parameterSpecification)

# This code finds the NaN gradients
for (x, y) in Flux.Data.DataLoader(trainingSamples[1], trainingSamples[2], batchsize = 10, shuffle = false)
  tc = 0; ps = params(model)
  gs = gradient(ps) do
    tl = lossTargetA(x, y, model, ps, parameterSpecification)
    tc = tl
    return tl
		# return sum(norm.(model(x)))
  end # gradient evaluation
  println("-----------------------------")
  # println(any(x -> x < 0 || x > 1 || isnan(x) || isinf(x), model(x)))
  println(tc)
  # println(gs[model[1].W][1:10])
  # println(model[1].W[1:10])
  opt = ADAM(0.00001)
  Flux.update!(opt, ps, gs)
  # println(get!(opt.state, model[1].W[1], (0.0f0, 0.0f0, opt.beta)))
  # if isnan(gs[model[1].W][1])
    # break
  # end
end

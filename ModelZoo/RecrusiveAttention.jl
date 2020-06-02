# --------------------------------------------------------------------------------------------------
# Recursive Attention Model:
#
# Recursive application of the same Attention mechanism

# Idea: Implent Attention like in the Google paper. However use only one Matrix for the
# attention and one for the feed worward layer and reuse them in multiple times!

mutable struct recursiveAttentionCell{A, B}
	# Prediction layers
 	encoderFeedForward::A
 	decoderFeedForward::B

	# Iteration Information
	nEncodingIterations::Int
	encoderInterFFDimension::Int
	nDecodingIterations::Int
	decoderInterFFDimension::Int

	# Metadata
	vocabSize::Int
  outputSize::Int # this can be changed to be free.. But this needs an estimator
end

# TODO args is now depricated
function recursiveAttentionModel(args; nEncoderIterations = 6, nDecoderIterations = 6,
		encoderInterFFDimension = 512, decoderInterFFDimension = 512)
	@error("args is now depricated! Fix this first and then remove this error message!")
	@info("Creating a recursive attention model")
	@info("- Provide unflattened samples")

	encoderFeedForward = Chain(
		Dense(args.nVocab, encoderInterFFDimension, relu),
		Dense(encoderInterFFDimension, args.nVocab)
	)

	decoderFeedForward = Chain(
		Dense(args.nVocab, decoderInterFFDimension, relu),
		Dense(decoderInterFFDimension, args.nVocab)
	)

  return recursiveAttentionCell(encoderFeedForward, decoderFeedForward,
		nEncoderIterations, encoderInterFFDimension, nDecoderIterations, decoderInterFFDimension,
		args.nVocab, args.nOutputLength
	)
end

# parameters
Flux.trainable(model::recursiveAttentionCell) =
	(model.encoderFeedForward, model.decoderFeedForward)

# function gpu(model::recursiveAttentionCell)
# 	return recursiveAttentionCell(
# 		Flux.gpu(model.encoderFeedForward), Flux.gpu(model.decoderFeedForward),
# 		model.encoderIterations, model.decoderIterations, model.vocabSize,
# 		model.inputSize, model.outputSize
# 	)
# end

function _attention(QKV) # Query, Kev Value
	# Softmax along first dimension, so that multiplaction happens with a L¹-normalized vector
	QKV * softmax(transpose(QKV) * QKV / size(QKV, 1), dims=1)
	# Still needs to be tested, whether this is faster then an implementation using CUBLAS.syrk.('U', 'N', α, x)
	# Probably another way is to use `mul(Y, QVK, transpose(QVK))` with a preallocated Y
end

function _attention(QK, V)
	QK * softmax(transpose(QK) * V / size(QK, 1), dims=1)
end

# forward pass
function (model::recursiveAttentionCell)(x)
	for i in 1:model.nEncodingIterations
		x = x .+ _attention.(x)
   	# x = mapslices.(v -> v ./ norm(v), x, dims = 1) # Implement a proper layer norm function (-> 0 mean, variance 1)
		x = x .+ model.encoderFeedForward.(x)
   	# x = mapslices.(v -> v ./ norm(v), x, dims = 1)
	end

	# put the initialization into custom routine
	ŷ = [repeat(one(typeof(x[1][1])) * Flux.onehot("<SOS>", vocabulary), outer=(1, model.outputSize) )
		for _ in 1:length(x)]

	for i in 1:model.nDecodingIterations
		ŷ = ŷ .+ _attention.(ŷ)
   	# ŷ = mapslices.(v -> v ./ norm(v), ŷ, dims = 1)
		ŷ = ŷ .+ _attention.(x, ŷ)
   	# ŷ = mapslices.(v -> v ./ norm(v), ŷ, dims = 1)
		ŷ = ŷ .+ model.decoderFeedForward.(ŷ)
   	# ŷ = mapslices.(v -> v ./ norm(v), ŷ, dims = 1)
	end
	return softmax.(ŷ, dims=1)
end

# pretty printing
function Base.show(io::IO, model::recursiveAttentionCell)
	print(io, "recursiveAttentionModel(", model.nEncodingIterations, "x(")
	print(io, model.vocabSize, ", ", model.encoderInterFFDimension, ", ", model.vocabSize, ") - ")
	print(io, model.nDecodingIterations, "x(")
	print(io, model.vocabSize, ", ", model.decoderInterFFDimension, ", ", model.vocabSize, ")")
end

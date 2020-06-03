# --------------------------------------------------------------------------------------------------
# Recursive Attention Model:
#
# Recursive application of the same Attention mechanism

# Idea: Implent Attention like in the Google paper. However use only one Matrix for the
# attention and one for the feed worward layer and reuse them in multiple times!

mutable struct recursiveAttentionCell{A, B, C, D, E, F}
	# Prediction layers
 	encoderLinearAttentionUnit::A
	encoderTokenFeedForward::B

	estimator::C

	decoderLinearAttentionUnit::D
	decoderExpressionFeedForward::E
	decoderPredictor::F

	# Iteration Information
	nEncodingIterations::Int
	encoderInterFFDimension::Int
	nDecodingIterations::Int
	decoderInterFFDimension::Int

	# Metadata
	vocabSize::Int
  inputLength::Int  # Input size for the _estimator_; needed for the embedding
  outputLength::Int # this can be changed to be free.. But this then needs to be known beforehand
end
# TODO flag whether estimator is trainable

"""
	recursiveAttentionModel(args, estimatorTopology; ...)

	Creates the model with a fresh estimator with the given topology.
	Notation: `estimatorTopology = [512]`: Entire sequence gets reduced
	to a 512 sized context vector, which predicts entire output sequence.
"""
function recursiveAttentionModel(vocabSize::Int, inputLength::Int, outputLength::Int,
		estimatorTopology::Array{Int, 1};
		nEncoderIterations = 6, nDecoderIterations = 6,
		encoderInterFFDimension = 128, decoderInterFFDimension = 512)
	@info("Creating Estimator\n")
	estimatorLayers = [Dense(vocabSize * inputLength, estimatorTopology[1])]
	for i in 1:length(estimatorTopology) - 1
		push!(estimatorLayers, Dense(estimatorTopology[i], estimatorTopology[i+1]) )
	end
	push!(estimatorLayers, Dense(estimatorTopology[end], vocabSize * outputLength))

	estimator = Chain(estimatorLayers...)

	return recursiveAttentionModel(vocabSize, inputLength, outputLength;
		nEncoderIterations = nEncoderIterations, nDecoderIterations = nDecoderIterations,
		encoderInterFFDimension = encoderInterFFDimension, decoderInterFFDimension = decoderInterFFDimension,
		estimator = estimator)
end

function recursiveAttentionModel(vocabSize::Int, inputLength::Int, outputLength::Int;
		nEncoderIterations = 6, nDecoderIterations = 6,
		encoderInterFFDimension = 128, decoderInterFFDimension = 512, estimator = nothing)
	@assert !isnothing(estimator)
	@info("Creating a recursive attention model")
	@info("- Provide unflattened samples")

	encoderLinearAttentionUnit = Dense(vocabSize, vocabSize)

	encoderTokenFeedForward = Chain(
		Dense(vocabSize, encoderInterFFDimension, relu),
		Dense(encoderInterFFDimension, vocabSize)
	)

	decoderLinearAttentionUnit = Dense(vocabSize, vocabSize)

	decoderExpressionFeedForward = Chain(
		Dense(vocabSize * outputLength, decoderInterFFDimension, relu),
		Dense(decoderInterFFDimension, vocabSize * outputLength)
	)

	decoderPredictor = Dense(vocabSize * outputLength, vocabSize * outputLength)

  return recursiveAttentionCell(encoderLinearAttentionUnit, encoderTokenFeedForward, estimator,
		decoderLinearAttentionUnit, decoderExpressionFeedForward, decoderPredictor,
		nEncoderIterations, encoderInterFFDimension, nDecoderIterations, decoderInterFFDimension,
		vocabSize, inputLength, outputLength
	)
end

# parameters
Flux.trainable(model::recursiveAttentionCell) =
	(model.encoderLinearAttentionUnit, model.encoderFeedForward,
	 model.decoderLinearAttentionUnit, model.decoderFeedForward,
	 model.decoderPredictor, model.estimator)

# function gpu(model::recursiveAttentionCell)
# 	return recursiveAttentionCell(
# 		Flux.gpu(model.encoderFeedForward), Flux.gpu(model.decoderFeedForward),
# 		model.encoderIterations, model.decoderIterations, model.vocabSize,
# 		model.inputLength, model.outputLength
# 	)
# end

function _attention(Q, KV) # Query, Kev Value
	# Softmax along first dimension, so that multiplaction happens with a L¹-normalized vector
	KV * softmax(transpose(Q) * KV / size(KV, 2))
	# Still needs to be tested, whether this is faster then an implementation using CUBLAS.syrk.('U', 'N', α, x)
	# Probably another way is to use `mul(Y, QVK, transpose(QVK))` with a preallocated Y
end

function _attention(Q, K, V)
	V * softmax(transpose(Q) * K / size(K, 2))
end

function _encode(model::recursiveAttentionCell, x)
	for i in 1:model.nEncodingIterations
		x = x + _attention(model.encoderLinearAttentionUnit(x), x)
		x = softmax(x) # x = mapslices.(v -> v ./ norm(v), x, dims = 1)
		# Implement a proper layer norm function (-> 0 mean, variance 1)
		x = x + model.encoderTokenFeedForward(x)
		x = softmax(x) # x = mapslices.(v -> v ./ norm(v), x, dims = 1)
	end
	return x
end

function _estimate(model, x)
	x = DataPreparation.flatten(x, model.inputLength)

	return model.estimator(x)
end

function _decode(model::recursiveAttentionCell, context, ŷ)
	ŷ = unflatten(ŷ, model.outputLength)	# == (batchsize, vocabSize, outputLength)

	for i in 1:model.nDecodingIterations
		ŷ = ŷ .+ _attention(model.decoderLinearAttentionUnit(context), ŷ, context)
		ŷ = softmax(ŷ) # ŷ = mapslices.(v -> v ./ norm(v), ŷ, dims = 1)

		ŷ = DataPreparation.flatten(ŷ) # == (batchsize, vocabSize * outputLength)
		ŷ = ŷ + model.decoderExpressionFeedForward(ŷ)
		ŷ = unflatten(ŷ, model.outputLength)	# == (batchsize, vocabSize, outputLength)
		ŷ = softmax(ŷ) # ŷ = mapslices.(v -> v ./ norm(v), ŷ, dims = 1)
	end

	ŷ = model.decoderPredictor(DataPreparation.flatten(ŷ))
	return softmax(unflatten(ŷ, model.outputLength))
end

# forward pass
function (model::recursiveAttentionCell)(x)
	context = _encode(model, x)

	ŷ = _estimate(model, x)

	return _decode(model, context, ŷ)
end

# pretty printing
function Base.show(io::IO, model::recursiveAttentionCell)
	print(io, "recursiveAttentionModel(", model.nEncodingIterations, "x Encoder Layer, ")
	print(io, model.nDecodingIterations, "x Decoder Layer, ")
	print(io, "estimator = ", model.estimator, "; ")
	print(io, "(", model.vocabSize, ", ", model.inputLength, ", ", model.outputLength, ") )")
end

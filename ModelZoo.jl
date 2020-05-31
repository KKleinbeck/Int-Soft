include("ModelZoo/SimpleModel.jl")


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

function recursiveAttentionModel(args; nEncoderIterations = 6, nDecoderIterations = 6,
		encoderInterFFDimension = 512, decoderInterFFDimension = 512)
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
	return ŷ
end

# pretty printing
function Base.show(io::IO, model::recursiveAttentionCell)
	print(io, "recursiveAttentionModel(", model.nEncodingIterations, "x(")
	print(io, model.vocabSize, ", ", model.encoderInterFFDimension, ", ", model.vocabSize, ") - ")
	print(io, model.nDecodingIterations, "x(")
	print(io, model.vocabSize, ", ", model.decoderInterFFDimension, ", ", model.vocabSize, ")")
end


# --------------------------------------------------------------------------------------------------
# Pseudo Model:
#
# Use this code as a snippet

# mutable struct modelCell{A, B, C, D, E, F, G, X, Y, Z}
#     # Prediction layers
#     embedding::A
#     output::B
#     lstm::C
#     # Attention layers
#     attention_linear1::D
#     attention_linear2::E
#     attention_conv::F
#     attention_features_size::G
#     # Metadata
#     filternum::X
#     poollength::Y
#     hiddensize::Z
# end
#
# function modelName(in::Integer, hiddensize::Integer, poollength::Integer, layers=1,
# 	 			filternum=32, filtersize=1; initW = Flux.glorot_uniform, initb = Flux.zeros,
# 				init = Flux.glorot_uniform)
#
# 	embedding = Dense(in, hiddensize, Flux.relu; initW = initW, initb = initb)
#     output = Dense(hiddensize, 1; initW = initW, initb = initb)  # 1 could be replaced by output_horizon
#     lstm = StackedLSTM(hiddensize, hiddensize, hiddensize, layers; init = init)
# 	# ...
#
#     return modelZell(embedding, output, lstm, attention_linear1, attention_linear2,
#                     attention_conv, attention_feature_size, filternum, poollength,
#                     hiddensize)
# end
#
# # Define parameters and resetting the LSTM layer
# Flux.trainable(m::modelCell) = (m.embedding, m.output, m.lstm, m.attention_linear1,
#  				m.attention_linear2, m.attention_conv)
# Flux.reset!(m::modelCell) = Flux.reset!(m.lstm.chain)
#
#
#
# # Model output
# function (m::TPALSTMCell)(x)
#     inp = dropdims(x, dims=3)
# 	#...
#     return m.output(ht)
# end
#
# # Pretty printing
# function Base.show(io::IO, l::TPALSTMCell)
# 	print(io, "modelName(", size(l.embedding.W,2))
# 	print(io, ", ", l.hiddensize)
# 	print(io, ", ", l.poollength)
# 	length(l.lstm.chain) == 1 || print(io, ", ", length(l.lstm.chain))
# 	(l.filternum == 32 && size(l.attention_conv.weight,1) == 1) || print(io, ", ", l.filternum)
# 	(l.filternum == 32 && size(l.attention_conv.weight,1) == 1) || print(io, ", ", size(l.attention_conv.weight,1))
# 	print(io, ")")
# end

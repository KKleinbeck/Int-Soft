# --------------------------------------------------------------------------------------------------
# Simple Model:
#
# One Encoder Layer - One Decoder Layer

mutable struct simpleEncoderDecoderCell{A, W, X, Y, Z}
	# Prediction layers
 	encoderDecoderChain::A
	# Metadata
	vocabSize::W
	inputSize::X
  contextSize::Y
  outputSize::Z
end

function simpleEncoderDecoder(args)
	@info("Creating a simple encoder decoder model")
	@info("- Provide flattened samples")

	encoderDecoder = Chain(
		Dense(args.nInputLength*args.nVocab, args.nContextLength, σ),
		Dense(args.nContextLength, args.nOutputLength*args.nVocab),
		softmax
	)

  return simpleEncoderDecoderCell(encoderDecoder, args.nVocab,
		args.nInputLength, args.nContextLength, args.nOutputLength
	)
end

# Parameters
Flux.trainable(model::simpleEncoderDecoderCell) = model.encoderDecoderChain

# function gpu(model::simpleEncoderDecoderCell)
# 	return simpleEncoderDecoderCell(Flux.gpu(model.encoderDecoderChain), model.vocabSize,
# 		model.inputSize, model.contextSize, model.outputSize
# 	)
# end

# Forward pass
function (model::simpleEncoderDecoderCell)(x, _)
	return model.encoderDecoderChain(x)
end

# Pretty printing
function Base.show(io::IO, model::simpleEncoderDecoderCell)
	print(io, "simpleEncoderDecoder(", model.inputSize)
	print(io, ", ", model.contextSize)
	print(io, ", ", model.outputSize)
	print(io, ")")
end

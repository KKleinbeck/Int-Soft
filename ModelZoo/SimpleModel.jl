using Flux: unstack

# --------------------------------------------------------------------------------------------------
# Simple Model:
#
# One Encoder Layer - One Decoder Layer

mutable struct simpleEncoderDecoderCell{A, W, X, Y, Z}
	# Prediction layers
 	encoderDecoderChain::A
	# Metadata
	vocabSize::W
	inputLength::X
  contextSize::Y
  outputLength::Z
end

function simpleEncoderDecoder(vocabSize, inputLength, outputLength; contextSize = 512)
	@info("Creating a simple encoder decoder model")
	@info("- Provide flattened samples")

	encoderDecoder = Chain(
		Dense(inputLength * vocabSize, contextSize, σ),
		Dense(contextSize, outputLength * vocabSize)
	)

  return simpleEncoderDecoderCell(encoderDecoder, vocabSize, inputLength,
		contextSize, outputLength
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
function (model::simpleEncoderDecoderCell)(x)
	x = model.encoderDecoderChain(x)
	x = unflatten.(unstack(x, 2))
	return hcat(flatten.(softmax.(x))...)
end

# Pretty printing
function Base.show(io::IO, model::simpleEncoderDecoderCell)
	print(io, "simpleEncoderDecoder(", model.inputLength)
	print(io, ", ", model.contextSize)
	print(io, ", ", model.outputLength)
	print(io, ")")
end

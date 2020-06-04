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
		Dense(contextSize, outputLength * vocabSize, σ)
	)

  return simpleEncoderDecoderCell(encoderDecoder, vocabSize, inputLength,
		contextSize, outputLength
	)
end

# Parameters
Flux.trainable(model::simpleEncoderDecoderCell) = model.encoderDecoderChain

function gpu(model::simpleEncoderDecoderCell)
	return simpleEncoderDecoderCell(Flux.gpu(model.encoderDecoderChain), model.vocabSize,
		model.inputLength, model.contextSize, model.outputLength
	)
end

# Forward pass
function (model::simpleEncoderDecoderCell)(x)
	return model.encoderDecoderChain(x)
end

# Pretty printing
function Base.show(io::IO, model::simpleEncoderDecoderCell)
	print(io, "simpleEncoderDecoder(", model.inputLength)
	print(io, ", ", model.contextSize)
	print(io, ", ", model.outputLength)
	print(io, ")")
end

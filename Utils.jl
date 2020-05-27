module Utils

export Args, kldivergenceC, extractSample

using Parameters: @with_kw
using Flux: crossentropy
using CuArrays: @cufunc

@with_kw mutable struct Args
	# Model Hyperparameters
	nVocab::Int         = 0       # size of input layer, will be assigned as length(alphabet)
	nInputLength::Int   = 10      # length of the input sequence
	nOutputLength::Int  = 10
	nContextLength::Int = 128     # size of the context vector

	# Optimizer and loss function Hyperparameters
	η::Float32        = 1e-3    # learning rate
	wPenalty::Float32 = 1e-3

	# Data and Training related Hyperparameters
	batchsize::Int = 1
	shuffle::Bool  = true
	throttle::Int  = 30
	nEpochs::Int   = 10
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

end  # module Utils

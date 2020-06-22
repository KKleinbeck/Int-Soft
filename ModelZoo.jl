import Flux: gpu, cpu

include("ModelZoo/SimpleModel.jl")
include("ModelZoo/RecrusiveAttention.jl")

macro test(x)
	structName = x.args[2]
	structContent = x.args[3].args[2:2:end]
	gpuFlags = [:gpu ∈ line.args for line in structContent]
	gpuElements = []
	trainFlags = [:trainable ∈ line.args for line in structContent]
	trainElements = []

	function extractFlaggedElements!(flaggedElements, flags, content)
		for i in 1:length(content)
			if flags[i]
				push!(flaggedElements, content[i].args[2].args[1])
			end
		end
	end

	extractFlaggedElements!(gpuElements,   gpuFlags,   structContent)
	extractFlaggedElements!(trainElements, trainFlags, structContent)

	function decorateWith!(elements, decoration)
		for i in 1:length(elements)
			elements[i] = Symbol(decoration, ".", elements[i])
		end
	end

	decorateWith!(gpuElements,   :x) # Later needed in the funtion definitions
	decorateWith!(trainElements, :x) # Later needed in the funtion definitions

	function removeTokens!(content, indexFlags)
		for i in 1:length(indexFlags)
			!indexFlags[i] && continue
			content[i] = content[i].args[2]
		end
	end

	removeTokens!(structContent, gpuFlags .| trainFlags)

	x.args[3].args[2:2:end] .= structContent

	println("-----------------------------------------")
	println(gpuElements)
	println("-----------------------------------------")
	return quote
		$x

		function $(esc(:gpu))($(esc(:x))::$(esc(structName)) )
			return tuple($(esc.(gpuElements)...))
			return $(esc(gpuElements[1])), $(esc(gpuElements[2]))
			return x.a, x.b # <- this is needed, at least in form
			return $(esc(Symbol("xa")))
		end
	end
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

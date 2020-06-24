macro FluxModel(x)
	structName     = x.args[1] ? x.args[2].args[1] : x.args[2] # For mutable struct ignore the types, e.g. `Model{A, B}`
	structContent  = x.args[3].args[2:2:end]
	gpuFlags = [:gpu ∈ line.args for line in structContent]
	trainFlags = [:trainable ∈ line.args for line in structContent]
	trainElements = []

	# -----------------------------------------------
	# First extract all marked elements if needed
	function extractFlaggedElements!(flaggedElements, flags, content)
		for i in 1:length(content)
			if flags[i]
				push!(flaggedElements, content[i].args[2].args[1])
			end
		end
	end

	extractFlaggedElements!(trainElements, trainFlags, structContent)


	# -----------------------------------------------
	# Now remove all symboles used for marking
	function removeTokens!(content, indexFlags)
		for i in 1:length(indexFlags)
			!indexFlags[i] && continue
			content[i] = content[i].args[2]
		end
	end

	removeTokens!(structContent, gpuFlags .| trainFlags)


	# -----------------------------------------------
	# Decorate all extracted elements (and once all elements) with an `x.`.
	# This is later needed in the functions.
	function decorateWith(elements, decoration)
		result = Expr[]
		for i in 1:length(elements)
			push!(result, :($decoration.$(elements[i])) )
		end
		return result
	end

	structElements = [line.args[1] for line in structContent]
	structElements = decorateWith(structElements, :x)
	trainElements  = decorateWith(trainElements,  :x)


	# -----------------------------------------------
	# Lastly, wrap all gpu-marked elements in the according functions
	function encapsulate(elements, condition, capsule)
		result = []
		for i in 1:length(elements)
			if condition[i]
				push!(result, :($(capsule)($(elements[i])) ) )
			else
				push!(result, elements[i])
			end
		end
		return result
	end

	gpuElements = encapsulate(structElements, gpuFlags, :gpu)
	cpuElements = encapsulate(structElements, gpuFlags, :cpu)

	x.args[3].args[2:2:end] .= structContent

	return quote
		$x

		function $(esc(:gpu))($(esc(:x))::$(esc(structName)) )
			return $(esc(structName))($(esc.(gpuElements)...))
		end

		function $(esc(:cpu))($(esc(:x))::$(esc(structName)) )
			return $(esc(structName))($(esc.(cpuElements)...))
		end

		function $(esc(:(Flux.trainable)))($(esc(:x))::$(esc(structName)) )
			return tuple($(esc.(trainElements)...))
		end
	end
end

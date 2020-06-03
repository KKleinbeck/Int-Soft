module DataPreparation

export vocabulary, loadSamples, flatten, unflatten, expandTo, tokenize

using Flux: onehotbatch, onehot, onecold, rpad, chunk, unstack

vocabulary = ["<SOS>", "<EOS>", "x", "c", "p", "n", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
	"Exp", "Log", "Sin", "Cos", "Tan", "Cot", "Sec", "Csc",
	"Sinh", "Cosh", "Tanh", "Coth", "Sech", "Csch", "Sqrt",
	"+", "-", "*", "/", "^"
]

function getData(filename::String)
	expressions = split.(readlines(filename), "\t") # Expression is an Array with elements `[function, antiDerivative]`
	expressions = [replace.(expression, r"[(,)]" => "")        for expression in expressions] # strip visual aids
	expressions = [replace.(expression, r"([0-9])" => s" \1 ") for expression in expressions] # add whitespace around numbers
	expressions = [split.(expression)                          for expression in expressions] # split expressions into tokes

	return ([expression[1] for expression in expressions], [expression[2] for expression in expressions])
end

function filterExpressions(data, labels; maxInputLength = Inf, maxOutputLength = Inf,
		dissallowedTokens = r"NON") # give dissallowedTokens as regex, e.g. `r"Tan|Exp|^"`
	maliciousIndices = []

	for index in 1:length(data)
		if  length(data[index])   > maxInputLength  || any(occursin.(dissallowedTokens, data[index]))   ||
				length(labels[index]) > maxOutputLength || any(occursin.(dissallowedTokens, labels[index])) ||
				!all(token -> occursin.(token, vocabulary) |> any, [data[index]..., labels[index]...])
			push!(maliciousIndices, index)
		end
	end

	# maliciousIndices = [index for index in 1:length(data) if ... ] TODO: test whether this workds(it should)

	deleteat!(data, maliciousIndices)
	deleteat!(labels, maliciousIndices)

	return data, labels
end

function isValidExpression(expression)
	return !any([!any(occursin.(token, vocabulary)) for token in expression])
end

function encode(expressions)
	expressions = [onehotbatch(expression, vocabulary) for expression in expressions
		if isValidExpression(expression)]
	return expressions
end

function loadSamples(filename::String; evaluationFraction = 0.1, type = Bool,
		maxInputLength = Inf, maxOutputLength = Inf, flattenTo = nothing, expandToMaxLength = nothing,
		dissallowedTokens = r"NON")
	@assert 0 <= evaluationFraction <= 1
	@info("Loading Samples. Stand by...")
	data, labels  = getData(filename)

	@info("Loaded $(length(data)) samples. Filtering...")
	data, labels   = filterExpressions(data, labels; dissallowedTokens = dissallowedTokens,
		maxInputLength = maxInputLength, maxOutputLength = maxOutputLength)


	nEvaluationSamples = floor(Int, length(data) * evaluationFraction)
	@info("Reduced to $(length(data)) samples,\n" *
		"\t\t$(length(data) - nEvaluationSamples) are used for training,\n" *
		"\t\t$(floor(Int, length(data)*(evaluationFraction) + 1)) are used for evaluation.\n" *
		"\tEncoding...")
	data   = encode(data)   * one(type)
	labels = encode(labels) * one(type)

	if !isnothing(flattenTo)
		@info("Flattening...")
		@assert flattenTo[1] >= maxInputLength && flattenTo[2] >= maxOutputLength
		@assert isnothing(expandToMaxLength)
		data   = hcat(flatten.(data,   flattenTo[1])...)
		labels = hcat(flatten.(labels, flattenTo[2])...)

		return [data[:,1:end - nEvaluationSamples - 1], labels[:,1:end - nEvaluationSamples - 1]],
			[data[:,end - nEvaluationSamples:end], labels[:,end - nEvaluationSamples:end]]
	end

	if !isnothing(expandToMaxLength)
		@info("Expanding to maximum length...")
		@assert isa(expandToMaxLength, Tuple)

		if expandToMaxLength[1]
			data = expandTo.(data, maxInputLength)
		end
		if expandToMaxLength[2]
			labels = expandTo.(data, maxOutputLength)
		end
	end

	# Note: the slices data[x:y] are now arrays of arrays instead of multidimensional arrays. Maybe I need this
	return [data[1:end - nEvaluationSamples - 1], labels[1:end - nEvaluationSamples - 1]],
	 [data[end - nEvaluationSamples:end], labels[end - nEvaluationSamples:end]]
end


# --------------------------------------------------------------------------------------------------
# Auxilliary functions

"""
	flatten(expression[, length])

	Converts expression from format `(vocabSize, expressionLength)` to `(vocabSize * expressionLength)`.
	Optional length parameter can be given, which pads `<EOS>` tokens
"""
function flatten(expression)
	# TODO equivalent to reshape(expression, :). Which is better?
	vcat(unstack(expression, 2)...)
end
function flatten(expression, length::Int)
	vcat(rpad(unstack(expression, 2), length, onehot("<EOS>", vocabulary))...)
end

"""
	unflatten(expression[, expressionLength])

	Converts expression from format `(vocabSize * expressionLength)` to `(vocabSize, expressionLength)`.
	`expressionLength` is implicitly calculated by assuming the above vocabulary; it can be given as
	a parameter as well
"""
function unflatten(expression)
	# TODO equivalent to Flux.batch(chunk(expression, length(expression) ÷ length(vocabulary))). Whats better?
	hcat(chunk(expression, length(expression) ÷ length(vocabulary))...)
end
function unflatten(expression, length::Int)
	hcat(chunk(expression, length)...)
end

"""
	expandTo(expression, length)

	Converts expression from format `(vocabSize, expressionLength)` to `(vocabSize, length)`, by
	padding `<EOS>` tokens to the right.
"""
function expandTo(expression, length)
	return hcat(expression, repeat(onehot("<EOS>", vocabulary), outer = [1, length - size(expression, 2)]))
end

"""
	tokenize(expression)

	Converts (flattened or unflattened) expression into an array of tokens.
"""
function tokenize(expression)
	if length(size(expression)) == 2 # flattened expression
		expression = unflatten(expression)
	end

	return onecold(expression)
end

end # module

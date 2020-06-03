using Test
using Flux: onehotbatch

include("../DataPreparation.jl")
using .DataPreparation

@testset "Expression Conversions" begin
	expression = onehotbatch(["<SOS>", "<EOS>", "x"], vocabulary)
	vocabSize = length(vocabulary)
	@info("Testing encoding and flattening ...")
	@test expression[1,1] && !any(expression[2:end,1]) 			 # proper vocabulary
	@test size(expression) == (vocabSize, 3)					 			 # proper encoding
	@test length(flatten(expression))     == vocabSize * 3 	 # flattening #1
	@test length(flatten(expression, 10)) == vocabSize * 10	 # flattening #2

	flatExpression1 = flatten(expression)
	@info("Testing unflattening ...")
	@test size(unflatten(flatExpression1))    == (vocabSize, 3)				# proper size #1
	@test size(unflatten(flatExpression1, 3)) == (vocabSize, 3)				# proper size #2
	@test unflatten(flatExpression1) == unflatten(flatExpression1, 3) # correct results either way

	flatExpression2 = flatten(expression, 10)
	@info("Testing expanding of unflattened expressions...")
	@test size(expandTo(expression, 10))  == (vocabSize, 10)		 # expanding unflattened expression
	@test expandTo(expression, 10) == unflatten(flatExpression2) # correct results either way
end;

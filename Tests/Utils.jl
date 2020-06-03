using Test

include("../DataPreparation.jl")

include("../Utils.jl")

@testset "Figures of merit" begin
	yUnflattened      = onehotbatch(["Sin", "x",     "<EOS>"], vocabulary)
	ŷUnflattened      = onehotbatch(["Cos", "x",     "<EOS>"], vocabulary)
	ŷUnflattenedShort = onehotbatch(["Sin", "<EOS>", "x"    ], vocabulary)

	yFlattened      = flatten(yUnflattened)
	ŷFlattened      = flatten(ŷUnflattened)
	ŷFlattenedShort = flatten(ŷUnflattenedShort)

	@info("Testing misclassification rates...")
	@test misclassificationRate(ŷUnflattened, yUnflattened) ≈ 1/3
	@test misclassificationRate(ŷUnflattenedShort, yUnflattened) ≈ 1/3
	@test misclassificationRate(ŷFlattened, yFlattened) ≈ 1/3
	@test misclassificationRate(ŷFlattenedShort, yFlattened) ≈ 1/3

	@info("Testing expression equality")
	@test !expressionIsEqual(ŷUnflattened, yUnflattened)
	@test  expressionIsEqual(ŷUnflattened, ŷUnflattened)
	@test  expressionIsEqual(yUnflattened, yUnflattened)
	@test !expressionIsEqual(ŷUnflattenedShort, yUnflattened)

	@test !expressionIsEqual(ŷFlattened, yFlattened)
	@test  expressionIsEqual(ŷFlattened, ŷFlattened)
	@test  expressionIsEqual(yFlattened, yFlattened)
	@test !expressionIsEqual(ŷFlattenedShort, yFlattened)
end;

using Test

include("../DataPreparation")
using .DataPreparation

include("../Utils.jl")
using .Utils

@testset "Figures of merit" begin
	yUnflattened      = Flux.onehotbatch(["Sin", "x",     "<EOS>"], vocabulary)
	ŷUnflattened      = Flux.onehotbatch(["Cos", "x",     "<EOS>"], vocabulary)
	ŷUnflattenedShort = Flux.onehotbatch(["Sin", "<EOS>", "x"    ], vocabulary)

	yFlattened      = DataPreparation.flatten(yUnflattened)
	ŷFlattened      = DataPreparation.flatten(ŷUnflattened)
	ŷFlattenedShort = DataPreparation.flatten(ŷUnflattenedShort)

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

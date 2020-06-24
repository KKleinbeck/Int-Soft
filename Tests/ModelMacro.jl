using Test, Flux, CuArrays
import Flux: gpu, cpu

include("../ModelZoo/AutoStructures.jl")

# Must be defined outside of the `testset` to work
@FluxModel mutable struct TestStruct{A, B}
	array::A :gpu :trainable
	b::Int :trainable
	c::B
end

@testset "Correct Conversions" begin
	cpuStruct  = TestStruct([1.0, 2.0, 3.0], 1, 1.0)
	gpuStruct  = gpu(cpuStruct)
	cpuStruct2 = cpu(gpuStruct)

	@info("Testing correct types...")
	@test isa(cpuStruct.array,  Array)
	@test isa(gpuStruct.array,  CuArray)
	@test isa(cpuStruct2.array, Array)

	@info("Testing (return) values...")
	@test cpuStruct.array == cpuStruct2.array == [1.0, 2.0, 3.0]
	@test cpuStruct.b == cpuStruct2.b == gpuStruct.b == 1
	@test cpuStruct.c == cpuStruct2.c == gpuStruct.c == 1.0
	@test Flux.trainable(cpuStruct)[1] == Flux.trainable(cpuStruct2)[1] == [1.0, 2.0, 3.0]
	@test Flux.trainable(cpuStruct)[2] == Flux.trainable(cpuStruct2)[2] ==
				Flux.trainable(gpuStruct)[2] == 1
end

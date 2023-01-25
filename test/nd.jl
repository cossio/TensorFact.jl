using Test: @test, @testset, @inferred
using TensorFactorizations: masked
using Statistics: cor
using Random: bitrand

@testset "nd" begin
    X = randn(10, 10)
    mask = bitrand(10, 10)
    @test all((masked(X, mask) .== X) .|| (!).(mask))
end

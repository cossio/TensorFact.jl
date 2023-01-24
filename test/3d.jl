using Test: @test, @testset, @inferred
using TensorFact: als
using Statistics: cor

@testset "3-dimensional case" begin
    n, m, k = (7,5,9)
    r = 1

    A, B, C = randn(r, n), randn(r, m), randn(r, k)

    X = reshape(sum(reshape(A, r, n, 1, 1) .* reshape(B, r, 1, m, 1) .* reshape(C, r, 1, 1, k); dims=1), n, m, k) .+ randn(n, m, k) / 100

    (_A, _B, _C), errors = @inferred als(X, r)

    @test abs(cor(vec(A), vec(_A))) > 0.99
    @test abs(cor(vec(B), vec(_B))) > 0.99
    @test abs(cor(vec(C), vec(_C))) > 0.99

end

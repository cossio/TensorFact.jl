using Test: @test, @testset, @inferred
using TensorFactorizations: als
using Statistics: cor
using Random: bitrand
using LinearAlgebra: qr
using Tullio: @tullio

@testset "2-dimensional case" begin
    N, M = (17, 25)
    rank = 1

    A, B = randn(rank, N), randn(rank, M)

    @tullio X[i,j] := A[r,i] * B[r,j]
    X .+= randn(N, M) / 100

    mask = ones(N,M)
    (_A, _B), errors = @inferred als(X, mask; rank)

    @test abs(cor(vec(A), vec(_A))) > 0.99
    @test abs(cor(vec(B), vec(_B))) > 0.99

    N, M = (42, 31)
    A, B = randn(rank, N), randn(rank, M)
    @tullio X[i,j] := A[r,i] * B[r,j]
    X .+= randn(N, M) / 1000
    mask = (rand(N,M) .< 0.9)
    X = ifelse.(mask, X, NaN)
    (_A, _B), errors = @inferred als(X, mask; rank)
    @test abs(cor(vec(A), vec(_A))) > 0.9
    @test abs(cor(vec(B), vec(_B))) > 0.9
end

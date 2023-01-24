# 3-dimensional case

"""
    als(X, rank = 1; γ = 0, mask = 1, niter = 100)

Tensor factorization by alternating least squares.
"""
function als(X::AbstractArray{<:Real,3}, rank::Int = 1; γ::Real=0, mask=1, niter::Int = 100)
    n, m, k = size(X)

    A = randn(rank, n)
    B = randn(rank, m)
    C = randn(rank, k)

    _X = mask .* X

    errors = zeros(niter)

    for iter in 1:niter
        G = (B * B') .* (C * C')
        BC = reshape(B, rank, 1, m, 1) .* reshape(C, rank, 1, 1, k)
        Y = reshape(BC, rank, m * k) * reshape(permutedims(_X, (2,3,1)), m * k, n)
        A .= (G + γ * I) \ Y

        G = (A * A') .* (C * C')
        AC = reshape(A, rank, n, 1, 1) .* reshape(C, rank, 1, 1, k)
        Y = reshape(AC, rank, n * k) * reshape(permutedims(_X, (1,3,2)), n * k, m)
        B .= (G + γ * I) \ Y

        G = (A * A') .* (B * B')
        AB = reshape(A, rank, n, 1, 1) .* reshape(B, rank, 1, m, 1)
        Y = reshape(AB, rank, n * m) * reshape(permutedims(_X, (1,2,3)), n * m, k)
        C .= (G + γ * I) \ Y

        errors[iter] = error(X, A, B, C; mask)
    end

    return (A, B, C), errors
end

function error(X::AbstractArray{<:Real,3}, A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix; mask=trues(size(X)))
    @assert size(A, 1) == size(B, 1) == size(C, 1) # rank
    @assert size(X) == (size(A, 2), size(B, 2), size(C, 2))

    r = size(A, 1) # rank
    n, m, k = (size(A, 2), size(B, 2), size(C, 2))

    _X = reshape(sum(reshape(A, r, n, 1, 1) .* reshape(B, r, 1, m, 1) .* reshape(C, r, 1, 1, k); dims=1), n, m, k)
    return sum(abs2, mask .* (X - _X))
end

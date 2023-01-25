# 3-dimensional case

"""
    als(X, [mask]; rank = 1, γ = 0, niter = 100)

Tensor factorization by alternating least squares. Optionally give a binary `mask` to select
observed data-points.
"""
function als(X::AbstractArray{<:Real,3}; rank::Int=1, γ::Real=0, niter::Int=100)
    n, m, k = size(X)

    A = randn(rank, n)
    B = randn(rank, m)
    C = randn(rank, k)

    errors = zeros(niter)

    for iter in 1:niter
        Ga = (B * B') .* (C * C')
        BC = reshape(B, rank, 1, m, 1) .* reshape(C, rank, 1, 1, k)
        Ya = reshape(BC, rank, m * k) * reshape(permutedims(X, (2,3,1)), m * k, n)
        A .= (Ga + γ * I) \ Ya

        Gb = (A * A') .* (C * C')
        AC = reshape(A, rank, n, 1, 1) .* reshape(C, rank, 1, 1, k)
        Yb = reshape(AC, rank, n * k) * reshape(permutedims(X, (1,3,2)), n * k, m)
        B .= (Gb + γ * I) \ Yb

        Gc = (A * A') .* (B * B')
        AB = reshape(A, rank, n, 1, 1) .* reshape(B, rank, 1, m, 1)
        Yc = reshape(AB, rank, n * m) * reshape(permutedims(X, (1,2,3)), n * m, k)
        C .= (Gc + γ * I) \ Yc

        errors[iter] = error(X, A, B, C)
    end

    return (A, B, C), errors
end

function als(X::AbstractArray{<:Real,3}, mask::AbstractArray{<:Real,3}; rank::Int=1, γ::Real=0, niter::Int=100)
    @assert size(X) == size(mask)

    N, M, K = size(X)

    A = randn(rank, N)
    B = randn(rank, M)
    C = randn(rank, K)

    X_mask = masked(X, mask)

    errors = zeros(niter)

    for iter in 1:niter
        @tullio Ga[r,p,i] := mask[i,j,k] * B[r,j] * B[p,j] * C[r,k] * C[p,k]
        Ga .+= γ * reshape(I(rank), rank, rank, 1)
        @tullio Ya[p,i] := X_mask[i,j,k] * B[p,j] * C[p,k]
        for i in 1:N
            A[:,i] .= Ga[:,:,i] \ Ya[:,i]
        end

        @tullio Gb[r,p,j] := mask[i,j,k] * A[r,i] * A[p,i] * C[r,k] * C[p,k]
        Gb .+= γ * reshape(I(rank), rank, rank, 1)
        @tullio Yb[p,j] := X_mask[i,j,k] * A[p,i] * C[p,k]
        for j in 1:M
            B[:,j] .= Gb[:,:,j] \ Yb[:,j]
        end

        @tullio Gc[r,p,k] := mask[i,j,k] * A[r,i] * A[p,i] * B[r,j] * B[p,j]
        Gc .+= γ * reshape(I(rank), rank, rank, 1)
        @tullio Yc[p,k] := X_mask[i,j,k] * A[p,i] * B[p,j]
        for k in 1:K
            C[:,k] .= Gc[:,:,k] \ Yc[:,k]
        end

        errors[iter] = error(X, mask, A, B, C)
    end

    return (A, B, C), errors
end

function construct(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix)
    @assert size(A, 1) == size(B, 1) == size(C, 1) # rank
    @tullio X[i,j,k] := A[r,i] * B[r,j] * C[r,k]
    return X
end

# 2-dimensional case

function als(X::AbstractArray{<:Any,2}, mask::AbstractArray{<:Any,2}; rank::Int=1, γ::Real=0, niter::Int=100)
    @assert size(X) == size(mask)

    N, M = size(X)

    A = randn(rank, N)
    B = randn(rank, M)

    X_mask = masked(X, mask)

    errors = zeros(niter)

    for iter in 1:niter
        @tullio Ga[r,p,i] := mask[i,j] * B[r,j] * B[p,j]
        Ga .+= γ * reshape(I(rank), rank, rank, 1)
        @tullio Ya[p,i] := X_mask[i,j] * B[p,j]
        for i in 1:N
            A[:,i] .= Ga[:,:,i] \ Ya[:,i]
        end

        @tullio Gb[r,p,j] := mask[i,j] * A[r,i] * A[p,i]
        Gb .+= γ * reshape(I(rank), rank, rank, 1)
        @tullio Yb[p,j] := X_mask[i,j] * A[p,i]
        for j in 1:M
            B[:,j] .= Gb[:,:,j] \ Yb[:,j]
        end

        errors[iter] = error(X, mask, A, B)
    end

    return (A, B), errors
end

function construct(A::AbstractMatrix, B::AbstractMatrix)
    @assert size(A, 1) == size(B, 1) # rank
    @tullio X[i,j] := A[r,i] * B[r,j]
    return X
end

# Generic N-dimensional case
# This is not working yet!

function als(X::AbstractArray, rank::Int = 1; γ::Real=0, mask::AbstractArray=trues(size(X)), niter::Int = 100)
    error("This code is not working yet!")

    F = ntuple(d -> randn(rank, size(X, d)), ndims(X))
    C = map(f -> f * f', F)
    _X = mask .* X

    # reshaped factors
    T = ntuple(d -> reshape(F[d], rank, replace(map(one, size(X)), d, size(X, d))), ndims(X))

    for iter in niter
        for d in 1:ndims(X)
            G = hadamard(tuple_remove(C, d)...)
            Y = reshape(hadamard(tuple_remove(T, d)...), rank, :) * reshape(permutedims(X, (ntuple(l -> l < d ? l : l + 1, ndims(X) - 1)..., d)), :, size(X, d))
            F[d] .= (G + γ * I) \ Y
            C[d] .= F[d] * F[d]'
        end
    end
    return F
end

"""
    matrix_tensor_contract(T::AbstractArray, M::AbstractMatrix...)

Forms the contractive products M[1] * ... * T etc
"""
function matrix_tensor_contract(T::AbstractArray, M::AbstractMatrix...)
    reshape(T, size(T, 1), :)
    _T = last(M) * reshape(T, 1, size(T, 1), :)
    reshape(_T, )
end

matrix_tensor_contract(T::AbstractArray) = T

""""
    tuple_remove(tuple, i)

Removes i'th element from tuple.
"""
tuple_remove(t::Tuple, i::Int) = ntuple(length(t) - 1) do j
    j < i ? t[j] : t[j + 1]
end

tuple_replace(t::Tuple, i::Int, x) = ntuple(length(t)) do j
    i == j ? x : t[j]
end

hadamard(A::AbstractArray, Bs::AbstractArray...) = A .* hadamard(Bs...)
hadamard(A::AbstractArray) = A

function error(X::AbstractArray, F::AbstractArray...; mask=trues(size(X)))
    @assert length(F) == ndims(X)
    T = ntuple(d -> reshape(F[d], rank, replace(map(one, size(X)), d, size(X, d))), ndims(X))
    return sum(abs2, mask .* (X .- sum(hadamard(T)); dims=1))
end

function masked(X::AbstractArray, mask::AbstractArray)
    @assert size(X) == size(mask)
    return @. ifelse(Bool(mask), X, zero(X))
end

function error(X::AbstractArray{<:Real,N}, F::Vararg{AbstractMatrix,N}) where {N}
    @assert size(X) == map(A -> size(A, 2), F)
    _X = construct(F...)
    return sum(abs2, X - _X)
end

function error(X::AbstractArray{<:Real,N}, mask::AbstractArray, F::Vararg{AbstractMatrix,N}) where {N}
    @assert size(X) == map(A -> size(A, 2), F)
    _X = construct(F...)
    X_mask = masked(X, mask) # `X` can have masked `missing` without contaminating the result
    return sum(abs2, mask .* (X_mask - _X))
end

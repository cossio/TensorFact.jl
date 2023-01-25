function center(X::AbstractArray)
    Y = X .- mean(X)
    for d in 1:ndims(X)
        Y .= Y .- mean(Y; dims=d)
    end
    return Y
end

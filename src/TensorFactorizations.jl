module TensorFactorizations

using LinearAlgebra: I
using Statistics: mean
using Tullio: @tullio

const Tensor{N,T} = AbstractArray{T,N}

include("2d.jl")
include("3d.jl")
include("nd.jl")

end

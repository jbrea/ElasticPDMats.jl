module ElasticPDMats
import Base: size, getindex, setindex!, append!, show, view, deleteat!
import LinearAlgebra: mul!, ldiv!
using LinearAlgebra
using PDMats
import PDMats: dim, Matrix, diag, pdadd!, *, \, inv, logdet, eigmax, eigmin, whiten!, unwhiten!, quad, quad!, invquad, invquad!, X_A_Xt, Xt_A_X, X_invA_Xt, Xt_invA_X
export ElasticPDMat, ElasticSymmetricMatrix, ElasticCholesky, setcapacity!, setstepsize!

mutable struct ElasticSymmetricMatrix{T} <: AbstractArray{T, 2}
    N::Int64
    capacity::Int64
    stepsize::Int64
    data::Array{T, 2}
end
function ElasticSymmetricMatrix(m::AbstractArray{T, 2}; N = size(m, 1), capacity = 10^3, stepsize = 10^3) where {T}
    !issymmetric(m) && error("Data is not symmetric.")
    data = zeros(T, capacity, capacity)
    ind = CartesianIndices((1:N, 1:N))
    copyto!(data, ind, m, ind)
    ElasticSymmetricMatrix(N, capacity, stepsize, data)
end
ElasticSymmetricMatrix(; capacity = 10^3, stepsize = 10^3) = ElasticSymmetricMatrix(0, capacity, stepsize, zeros(capacity, capacity))

size(m::ElasticSymmetricMatrix) = (m.N, m.N)
getindex(m::ElasticSymmetricMatrix, i::Integer, j::Integer) = m.data[i, j]
function setindex!(m::ElasticSymmetricMatrix{T}, x::T, i::Integer, j::Integer) where T
    setindex!(m.data, x, i, j)
    i != j && setindex!(m.data, x, j, i)
    m
end
view(m::ElasticSymmetricMatrix, i, j) = view(m.data, i, j)
view(m::ElasticSymmetricMatrix) = view(m, 1:m.N, 1:m.N)

mul!(Y::AbstractArray{T, 1}, M::ElasticSymmetricMatrix{T}, V::AbstractArray{T, 1}) where {T} = mul!(Y, view(M), V)
mul!(Y::AbstractArray{T, 2}, M::ElasticSymmetricMatrix{T}, V::AbstractArray{T, 2}) where {T} = mul!(Y, view(M), V)
mul!(Y::AbstractArray{T, 2}, M1::ElasticSymmetricMatrix{T}, M2::ElasticSymmetricMatrix{T}) where {T} = mul!(Y, view(M1), view(M2))

setnewdata!(obj::ElasticSymmetricMatrix, data) = obj.data = data
resize!(obj::ElasticSymmetricMatrix) = resize!(obj, obj.data)
function resize!(obj, data::AbstractArray{T, 2}) where {T}
    tmp = zeros(T, obj.capacity, obj.capacity)
    ind = CartesianIndices((1:obj.N, 1:obj.N))
    copyto!(tmp, ind, data, ind)
    setnewdata!(obj, tmp)
    obj.capacity
end

append!(g::ElasticSymmetricMatrix{T}, data::Vector{T}) where {T} = append!(g, reshape(data, :, 1))
function append!(g::ElasticSymmetricMatrix{T}, data::AbstractArray{T, 2}) where {T}
    n, m = size(data)
    g.N + m > g.capacity && grow!(g)
    for j in 1:m
        for i in 1:min(g.N + j, n)
            @inbounds g.data[i, j + g.N] = data[i, j]
            @inbounds g.data[j + g.N, i] = data[i, j]
        end
    end
    g.N += m
    g
end
function deleteat!(g::ElasticSymmetricMatrix, i::Int)
    copyto!(g.data, CartesianIndices((1:g.N, i:g.N - 1)), 
            g.data, CartesianIndices((1:g.N, i+1:g.N)))
    copyto!(g.data, CartesianIndices((i:g.N-1, 1:g.N)), 
            g.data, CartesianIndices((i+1:g.N, 1:g.N)))
    g.N -= 1
    g
end

mutable struct ElasticCholesky{T, A} <: Factorization{T}
    N::Int64
    capacity::Int64
    stepsize::Int64
    c::Cholesky{T, A}
end
function ElasticCholesky(c::Cholesky{T, A}; capacity = 10^3, stepsize = 10^3) where {T, A}
    N = size(c, 1)
    data = zeros(T, capacity, capacity)
    ind = CartesianIndices((1:N, 1:N))
    copyto!(data, ind, c.factors, ind)
    ElasticCholesky(N, capacity, stepsize, Cholesky(data, 'U', LinearAlgebra.BlasInt(0)))
end
ElasticCholesky(; capacity = 10^3, stepsize = 10^3) = ElasticCholesky(0, capacity, stepsize, Cholesky(zeros(capacity, capacity), 'U', LinearAlgebra.BlasInt(0)))


function setcapacity!(x::Union{ElasticSymmetricMatrix, ElasticCholesky}, c::Int)
    x.capacity = c
    resize!(x)
end
setstepsize!(x::Union{ElasticSymmetricMatrix, ElasticCholesky}, c::Int) = x.stepsize = c

view(c::ElasticCholesky, i, j) = Cholesky(view(c.c.factors, i, j), c.c.uplo, c.c.info)
view(c::ElasticCholesky) = view(c, 1:c.N, 1:c.N)
show(io::IO, m::MIME{Symbol("text/plain")}, c::ElasticCholesky) = show(io, m, view(c))
size(c::ElasticCholesky) = (c.N, c.N)
size(c::ElasticCholesky, i::Int) = c.N

ldiv!(c::ElasticCholesky, x) = ldiv!(view(c), x)

function grow!(c::Union{ElasticCholesky, ElasticSymmetricMatrix})
    c.capacity += c.stepsize
    resize!(c)
end
resize!(c::ElasticCholesky) = resize!(c, c.c.factors)
setnewdata!(c::ElasticCholesky, data) = c.c = Cholesky(data, 'U', LinearAlgebra.BlasInt(0))

append!(c::ElasticCholesky{T, A}, data::Vector{T}) where {T, A} = append!(c, reshape(data, :, 1))
function append!(c::ElasticCholesky{T,A}, data::A) where {T, A}
    n, m = size(data)
    c.N + m > c.capacity && grow!(c)
    s = data[1:c.N, 1:m]
    LAPACK.trtrs!('U', 'C', 'N', view(c).factors, s)
    colrange = c.N + 1:c.N + m
    copyto!(c.c.factors, CartesianIndices((1:c.N, colrange)), s, CartesianIndices((1:c.N, 1:m)))
    copyto!(c.c.factors, CartesianIndices((colrange, colrange)), view(data, colrange,1:m), CartesianIndices((1:m, 1:m)))
    BLAS.syrk!('U', 'T', -1., s, 1., view(c.c.factors, colrange, colrange))
    LinearAlgebra._chol!(view(c.c.factors, colrange, colrange), UpperTriangular)
    c.N += m
    c
end
# TODO: Check if this can be optimized.
function deleteat!(c::ElasticCholesky, i::Int)
    R = view(c.c.factors, i, i+1:c.N) * view(c.c.factors, i, i+1:c.N)' 
    R += view(c, i+1:c.N, i+1:c.N).U' * view(c, i+1:c.N, i+1:c.N).U
    cholesky!(R)
    copyto!(c.c.factors, CartesianIndices((1:i-1, i:c.N-1)),
            c.c.factors, CartesianIndices((1:i-1, i+1:c.N)))
    copyto!(c.c.factors, CartesianIndices((i:c.N-1, i:c.N-1)),
            R, CartesianIndices((1:c.N-i, 1:c.N-i)))
    c.N -= 1
    c
end

struct ElasticPDMat{T, A} <: AbstractPDMat{T}
    mat::ElasticSymmetricMatrix{T}
    chol::ElasticCholesky{T, A}
end
"""
    ElasticPDMat([m [, chol]]; capacity = 10^3, stepsize = 10^3)

Creates an elastic positive definite matrix with initial `capacity = 10^3` and 
`stepsize = 10^3`. The optional argument `m` is a positive definite, symmetric 
matrix and `chol` its cholesky decomposition. Use `append!` and `deleteat!` to
change an ElasticPDMat.
"""
ElasticPDMat(; kwargs...) = ElasticPDMat(ElasticSymmetricMatrix(; kwargs...), ElasticCholesky(kwargs...))
ElasticPDMat(m; kwargs...) = ElasticPDMat(m, cholesky(m); kwargs...)
function ElasticPDMat(m, chol; kwargs...)
    ElasticPDMat(ElasticSymmetricMatrix(m; kwargs...),
                 ElasticCholesky(chol; kwargs...))
end

function setcapacity!(x::ElasticPDMat, c::Int)
    setcapacity!(x.mat, c)
    setcapacity!(x.chol, c)
end
function setstepsize!(x::ElasticPDMat, c::Int)
    setstepsize!(x.mat, c)
    setstepsize!(x.chol, c)
end

function append!(a::ElasticPDMat, data)
    append!(a.mat, data)
    append!(a.chol, data)
    a
end
function deleteat!(a::ElasticPDMat, i::Int)
    deleteat!(a.mat, i)
    deleteat!(a.chol, i)
    a
end
# TODO: more efficient blockwise deletion
function deleteat!(g::Union{ElasticSymmetricMatrix, ElasticCholesky, ElasticPDMat}, idxs::AbstractArray{Int, 1})
    map(i -> deleteat!(g, i), sort(idxs, rev = true))
end

dim(a::ElasticPDMat) = a.mat.N
Base.Matrix(a::ElasticPDMat) = Matrix(view(a.mat))
LinearAlgebra.diag(a::ElasticPDMat) = diag(view(a.mat))
function pdadd!(r::Matrix, a::Matrix, gb::ElasticPDMat, c::Real)
    b = view(gb.mat)
    PDMats.@check_argdims size(r) == size(a) == size(b)
    # PDMats._addscal!(r, m, view(a.mat), c) doesn't work because _addscal! does
    # not accept views. Below is copy-paste of PDMats
    if c == one(c)
        for i = 1:length(b)
            @inbounds r[i] = a[i] + b[i]
        end
    else
        for i = 1:length(b)
            @inbounds r[i] = a[i] + b[i] * c
        end
    end
    return r
end

*(a::ElasticPDMat, c::Real) = ElasticPDMat(c * Matrix(a), capacity = a.mat.capacity, stepsize = a.mat.stepsize) 
*(a::ElasticPDMat, x::AbstractArray) = a.mat * x 
\(a::ElasticPDMat, x::AbstractArray) = a.chol \ x

inv(a::ElasticPDMat) = ElasticPDMat(inv(a.chol), capacity = a.mat.capacity, stepsize = a.mat.stepsize) 
logdet(a::ElasticPDMat) = logdet(view(a.chol)) 
eigmax(a::ElasticPDMat) = eigmax(view(a.mat))
eigmin(a::ElasticPDMat) = eigmin(view(a.mat))


function whiten!(r::DenseVecOrMat, a::ElasticPDMat, x::DenseVecOrMat)  
    cf = view(a.chol).UL
    v = PDMats._rcopy!(r, x)
    istriu(cf) ? ldiv!(transpose(cf), v) : ldiv!(cf, v)
end

function unwhiten!(r::DenseVecOrMat, a::ElasticPDMat, x::DenseVecOrMat)  
    cf = view(a.chol).UL
    v = PDMats._rcopy!(r, x)
    istriu(cf) ? lmul!(transpose(cf), v) : lmul!(cf, v)
end

quad(a::ElasticPDMat{T, A}, x::AbstractArray{T, 1}) where {T, A} = dot(x, a * x)
quad!(r::AbstractArray, a::ElasticPDMat, x::DenseMatrix) = PDMats.colwise_dot!(r, x, a.mat * x) 
invquad(a::ElasticPDMat{T, A}, x::AbstractArray{T, 1}) where {T, A} = dot(x, a \ x) 
invquad!(r::AbstractArray, a::ElasticPDMat, x::DenseMatrix) = PDMats.colwise_dot!(r, x, a.mat \ x)
                                                 

function X_A_Xt(a::ElasticPDMat, x::DenseMatrix)        
    z = copy(x)
    cf = view(a.chol).UL
    rmul!(z, istriu(cf) ? transpose(cf) : cf)
    z * transpose(z)
end

function Xt_A_X(a::ElasticPDMat, x::DenseMatrix)        
    cf = view(a.chol).UL
    z = lmul!(istriu(cf) ? cf : transpose(cf), copy(x))
    transpose(z) * z
end

function X_invA_Xt(a::ElasticPDMat, x::DenseMatrix)     
    cf = view(a.chol).UL
    z = rdiv!(copy(x), istriu(cf) ? cf : transpose(cf))
    z * transpose(z)
end

function Xt_invA_X(a::ElasticPDMat, x::DenseMatrix)     
    cf = view(a.chol).UL
    z = ldiv!(istriu(cf) ? transpose(cf) : cf, copy(x))
    transpose(z) * z
end
end # module

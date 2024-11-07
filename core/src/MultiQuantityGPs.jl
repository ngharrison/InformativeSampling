"""
This module contains everything to do with what is inferred about values in the
environment. In practical terms: means, variances, and correlations. This is all
built on Gaussian Processes.

Main public types and functions:
$(EXPORTS)
"""
module MultiQuantityGPs

using LinearAlgebra: diag, PosDefException, norm, Diagonal, I
using AbstractGPs: GP, posterior, mean_and_var, mean_and_cov,
                   logpdf, cov, var, diag_Xt_invA_X, cholesky,
                   _symmetric, logdet, _sqmahal, Xt_invA_X
using IrrationalConstants: log2π
using Optim: optimize, Options, NelderMead, LBFGS
using ParameterHandling: value_flatten, fixed, value
using DocStringExtensions: TYPEDSIGNATURES, TYPEDEF, EXPORTS

using ..Samples: SampleInput
using ..Kernels: multiMean, singleKernel, multiKernel, fullyConnectedCovNum,
                 slfmKernel, fullyConnectedCovMat, manyToOneCovNum, manyToOneCovMat,
                 initHyperparams, mtoKernel
using ..Maps: Bounds

export MQGP, quantityCovMat, quantityCorMat, meanDerivAndVar, fullCov

include("LinearModels.jl")

"""
$(TYPEDEF)

Belief model struct and function for multiple quantities with 2D inputs.

Designed on top of a Multi-Quantity Gaussian Process, but can still be used with
a single quantity.

Its interface:
`X -> μ, σ` (SampleInputs -> means, standard deviations)
"""
struct MQGP{T}
    "posterior Gaussian Process used to do inference"
    gp
    "number of quantities of the GP"
    N
    "kernel function used in the GP"
    kernel::T
    "hyperparameters of the GP kernel function"
    θ
end

function Base.show(io::IO, bm::MQGP)
    print(io, "MQGP")
end
function Base.show(io::IO, ::MIME"text/plain", bm::MQGP)
    print(io, "MQGP:\n\tθ = $(bm.θ)")
end

"""
$(TYPEDSIGNATURES)

Creates and returns a MQGP with hyperparameters trained and conditioned on the
samples given. Lower and upper bounds are used to initialize one of the
hyperparameters.

A noise standard deviation can optionally be passed in either as a single scalar
value for all samples or a vector of values, one for each sample.

# Examples
```julia
# create a MQGP
beliefModel = MQGP([M.prior_samples; samples], bounds)
```
"""
function MQGP(samples, bounds::Bounds; N=maximum(s->s.x[2], samples),
                     kernel=multiKernel, means=(use=true, learned=true),
                     noise=(value=0.0, learned=false),
                     use_cond_pdf=false)
    # set up training data
    X, Y_vals, Y_errs = extractSampleVals(samples)

    # choose noise and mean
    σn = (noise.learned ? noise.value : fixed(noise.value))
    μ = (means.use && means.learned ? calcMeans(X, Y_vals, N) :
         fixed(means.use ? calcMeans(X, Y_vals, N) : zeros(N)))
    θ0 = initHyperparams(X, Y_vals, bounds, N, kernel; μ, σn)

    @debug "calculated means:" calcMeans(X, Y_vals, N)

    # optimize hyperparameters (train)
    θ = optimizeLoss(createLossFunc(X, Y_vals, Y_errs, kernel, use_cond_pdf), θ0)

    @debug "learned means:" θ.μ

    # produce optimized gp belief model
    fx = buildPriorGP(X, Y_errs, kernel, θ)
    f_post = posterior(fx, Y_vals) # gp conditioned on training samples

    return MQGP(f_post, N, kernel, θ)
end

# Produce a belief model with pre-chosen hyperparams
function MQGP(samples, θ; N=maximum(s->s.x[2], samples),
                     kernel=multiKernel)
    X, Y_vals, Y_errs = extractSampleVals(samples)

    fx = buildPriorGP(X, Y_errs, kernel, θ)
    f_post = posterior(fx, Y_vals) # gp conditioned on training samples

    return MQGP(f_post, N, kernel, θ)
end

function extractSampleVals(samples)
    X = getfield.(samples, :x)
    Y = getfield.(samples, :y)

    # split measurements if needed
    if Y isa AbstractArray{<:NTuple{2, <:Real}}
        Y_vals, Y_errs = first.(Y), last.(Y)
    else
        Y_vals, Y_errs = Y, 0.0
    end

    return X, Y_vals, Y_errs
end

function calcMeans(X, Y, N)
    mean_vals = zeros(N)
    nums = zeros(Int, N)
    for ((_, q), y) in zip(X, Y)
        mean_vals[q] += y
        nums[q] += 1
    end
    return mean_vals ./ nums
end

function buildPriorGP(X, Y_errs, kernel, θ, ϵ=0.0)
    f = GP(multiMean(θ), kernel(θ))
    σn = θ.σn isa AbstractArray ? (θ.σn[x[2]] for x in X) : θ.σn
    f(X, Y_errs.^2 .+ σn.^2 .+ √eps() .+ ϵ) # eps to prevent numerical issues
end

"""
Inputs:
- `X`: a single sample input or an array of multiple
- `full_cov`: (optional) if this is true, returns the full covariance matrix
  in place of the vector of standard deviations

Outputs:
- `μ, σ`: a pair of expected value(s) and uncertainty(s) for the given point(s)

# Examples
```julia
X = [([.1, .2], 1),
     ([.2, .1], 2)]
μ, σ = beliefModel(X) # result: [μ1, μ2], [σ1, σ2]
```
"""
function (bm::MQGP)(x::SampleInput; kwargs...)
    return only.(bm([x]); kwargs...)
end

function (bm::MQGP)(X::AbstractArray{SampleInput})
    μ, σ² = reshape.(mean_and_var(bm.gp, vec(X)), Ref(size(X)))
    return μ, .√clamp!(σ², 0.0, Inf) # avoid negative variances
end

"""
$(TYPEDSIGNATURES)

Returns the full covariance matrix for the belief model.
"""
function fullCov(bm::MQGP, X::AbstractArray{SampleInput})
    return cov(bm.gp, X) + I*√eps() # avoid negative eigenvalues
end

"""
$(TYPEDSIGNATURES)

Routine to optimize the lossFunc and return the optimal parameters `θ`.

Can pass in a different solver. NelderMead is picked as default for better speed
with about the same performance as LFBGS.
"""
function optimizeLoss(lossFunc, θ0; solver=NelderMead, iterations=5_000)
    options = Options(; iterations)

    θ0_flat, unflatten = value_flatten(θ0)
    loss_flat = lossFunc ∘ unflatten

    opt = optimize(loss_flat, θ0_flat, solver(), options)
    @debug "model optimizer:" opt

    return unflatten(opt.minimizer)
end

"""
$(TYPEDSIGNATURES)

This function creates the loss function for training the GP. The negative log
marginal likelihood is used.
"""
function createLossFunc(X, Y_vals, Y_errs, kernel, use_cond_pdf)
    # returns a function for the negative log marginal likelihood
    θ -> begin
        try
            fx = buildPriorGP(X, Y_errs, kernel, θ)
            logpdf_func = (use_cond_pdf ? condlogpdf : logpdf)
            return -logpdf_func(fx, Y_vals)
        catch e
            # for PosDefException
            # this seems to happen when θ.σ is extremely large and θ.ℓ is
            # much bigger than the search region dimensions
            @error e θ X Y_vals

            fx = buildPriorGP(X, Y_errs, kernel, θ, 1e-1*maximum(θ.σ))
            logpdf_func = (use_cond_pdf ? condlogpdf : logpdf)
            return -logpdf_func(fx, Y_vals)
        end
    end
end

# the logpdf of a conditional distribution
function condlogpdf(fx, Y_vals)
    # GP prior distribution for the observations
    m, C_mat = mean_and_cov(fx)
    X = fx.x

    # partition into primary and secondary
    i1 = filter(i->X[i][2]==1, eachindex(X))
    i2 = filter(i->X[i][2]!=1, eachindex(X))

    y1 = Y_vals[i1]
    y2 = Y_vals[i2]
    m1 = m[i1]
    m2 = m[i2]
    Σ11 = C_mat[i1,i1]
    Σ22 = cholesky(_symmetric(C_mat[i2,i2]))
    Σ21 = C_mat[i2,i1]

    # condition primary on secondary
    m_post = m1 + Σ21' * (Σ22 \ (y2 - m2))
    C_post = Σ11 - Xt_invA_X(Σ22, Σ21)

    return _logpdf(m_post, C_post, y1)
end

function _logpdf(m, C_mat, Y::AbstractVecOrMat{<:Real})
    C = cholesky(_symmetric(C_mat))
    T = promote_type(eltype(m), eltype(C), eltype(Y))
    return -((size(Y, 1) * T(log2π) + logdet(C)) .+ _sqmahal(m, C, Y)) ./ 2
end


"""
$(TYPEDSIGNATURES)

Returns the normed gradient of the mean of the belief model and its variance.
"""
function meanDerivAndVar(bm::MQGP, x::SampleInput)
    return only.(meanDerivAndVar(bm, [x]))
end

function meanDerivAndVar(bm::MQGP, X::AbstractArray{SampleInput})
    Xv = vec(X)
    dims = size(X)
    f = bm.gp
    C_xcond_x = cov(f.prior, f.data.x, Xv)
    m_deriv_norm = reshape(meanDerivNorm(Xv, f.data.x, C_xcond_x, bm.θ.ℓ, f.data.α), dims)
    C_post_diag = reshape(var(f.prior, Xv) - diag_Xt_invA_X(f.data.C, C_xcond_x), dims)
    return m_deriv_norm, .√clamp!(C_post_diag, 0.0, Inf)
end

function meanDerivNorm(X, Xm, C_xcond_x, ℓ, α)
    m_deriv_dims = map(eachindex(X[1][1])) do i
        dif_mat = [xi[1][i] - xj[1][i] for xi in X, xj in Xm]
        k_prime = -dif_mat ./ ℓ^2
        (C_xcond_x' .* k_prime) * α
    end
    return sqrt.(sum(arr.^2 for arr in m_deriv_dims))
end

"""
$(TYPEDSIGNATURES)

Gives the covariance matrix between all quantities from the hyperparameters.
"""
function quantityCovMat(bm::MQGP{typeof(multiKernel)})
    σn = bm.θ.σn isa AbstractArray ? bm.θ.σn : fill(bm.θ.σn, bm.N)
    return fullyConnectedCovMat(bm.θ.σ) .+ Diagonal(σn.^2)
end

function quantityCovMat(bm::MQGP{typeof(mtoKernel)})
    σn = bm.θ.σn isa AbstractArray ? bm.θ.σn : fill(bm.θ.σn, bm.N)
    return manyToOneCovMat(bm.θ.σ) .+ Diagonal(σn.^2)
end

"""
```julia
quantityCorMat(beliefModel::MQGP)
```

Gives the correlation matrix between all quantities from the hyperparameters.
"""
function quantityCorMat(bm::MQGP)
    cov_mat = quantityCovMat(bm)
    vars = diag(cov_mat)
    R = @. cov_mat / √(vars * vars') # broadcast shorthand
    idxs = isnan.(bm.θ.μ)
    R[idxs,:] .= NaN
    R[:,idxs] .= NaN
    return R
end

function LinearModel(mqgp::MQGP, Y, X)
    LinearModel(mqgp.θ.μ, quantityCovMat(mqgp), Y, X)
end

end

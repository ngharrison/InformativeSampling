module BeliefModels

using LinearAlgebra: I
using AbstractGPs: GP, posterior, marginals, logpdf, with_lengthscale,
                   SqExponentialKernel, IntrinsicCoregionMOKernel
using Statistics: mean, std
using Optim: optimize, Options, NelderMead
using ParameterHandling: value_flatten
using DocStringExtensions: SIGNATURES

using Environment: Index

abstract type BeliefModel end

"""
Belief model struct and function for multiple outputs with 2D inputs.

Designed on top of a Multi-output Gaussian Process. Can still be used with a
single output.
"""
struct BeliefModelSimple <: BeliefModel
    gp
    θ
end

"""
A combination of two BeliefModelSimples. One is trained on current samples and
the other is trained on current and previous samples. The main purpose for this
is to use the current for mean estimates and the combined for variance
estimates.
"""
struct BeliefModelSplit <: BeliefModel
    current
    combined
end

"""
Inputs:

    - X: a single sample index or an array of multiple

Outputs:

    - μ, σ: a pair of expected value(s) and uncertainty(s) for the given point(s)
"""
function (beliefModel::BeliefModelSimple)(x::Index)
    pred = only(marginals(beliefModel.gp([x])))
    return pred.μ, pred.σ
end

function (beliefModel::BeliefModelSimple)(X::Vector{Index})
    pred = marginals(beliefModel.gp(X))
    μ = getfield.(pred, :μ)
    σ = getfield.(pred, :σ)
    return μ, σ
end

"""
Inputs:

    - X: a single sample index or an array of multiple

Outputs:

    - μ, σ: a pair of expected value(s) and uncertainty(s) for the given point(s)
"""
function (beliefModel::BeliefModelSplit)(X::Union{Index, Vector{Index}})
    μ, _ = beliefModel.combined(X)
    _, σ = beliefModel.current(X)
    return μ, σ
end

"""
$SIGNATURES

Creates and returns a new BeliefModel. A BeliefModelSimple is returned if there
are no prior_samples, and it is trained and conditioned on the given samples.
Otherwise a BeliefModelSplit is returned, trained and conditioned on both the
samples and prior_samples. Lower and upper bounds are used to initialize one of
the hyperparameters.
"""
function generateBeliefModel(samples, prior_samples, lb, ub)
    # simple
    current = generateBeliefModel(samples, lb, ub)
    isempty(prior_samples) && return current
    # split
    combined = generateBeliefModel([prior_samples; samples], lb, ub)
    return BeliefModelSplit(current, combined)
end

"""
$SIGNATURES

Creates and returns a BeliefModelSimple with hyperparameters trained and
conditioned on the the samples given.
"""
function generateBeliefModel(samples, lb, ub; kernel=multiKernel)
    # set up training data
    X = getfield.(samples, :x)
    Y = getfield.(samples, :y)

    θ0 = initHyperparams(X, Y, lb, ub)

    # optimize hyperparameters (train)
    θ, opt = optimizeLoss(createLossFunc(X, Y, kernel), θ0)

    # produce optimized gp belief model
    f = GP(kernel(θ)) # prior gp
    fx = f(X, θ.σn^2+√eps())
    f_post = posterior(fx, Y) # gp conditioned on training samples

    return BeliefModelSimple(f_post, θ)
end

"""
$SIGNATURES

Creates the structure of hyperparameters and gives them initial values.
"""
function initHyperparams(X, Y, lb, ub)
    T = maximum(last, X) # number of outputs
    n = fullyConnectedCovNum(T)
    # TODO may change to all just 0.5
    σ = (length(Y)>1 ? std(Y) : 0.5)/sqrt(2) * ones(n)
    a = mean(mean([lb, ub]))
    ℓ = length(X)==1 ? a : a/length(X) + mean(std(first.(X)))*(1-1/length(X))
    σn = 0.001
    return (; σ, ℓ, σn)
end

"""
$SIGNATURES

Routine to optimize the lossFunc.

Can pass in a different solver. NelderMead is picked as default for better speed
with about the same performance as LFBGS.
"""
function optimizeLoss(lossFunc, θ0; solver=NelderMead, iterations=1_000)
    options = Options(; iterations)

    θ0_flat, unflatten = value_flatten(θ0)
    loss_flat = lossFunc ∘ unflatten

    opt = optimize(loss_flat, θ0_flat, solver(), options)

    return unflatten(opt.minimizer), opt
end

"""
$SIGNATURES

This function creates the loss function for training the GP. The negative log
marginal likelihood is used.
"""
function createLossFunc(X, Y, kernel)
    # returns a function for the negative log marginal likelihood
    θ -> begin
        try
            f = GP(kernel(θ))
            fx = f(X, θ.σn^2+√eps()) # eps to prevent numerical issues
            return -logpdf(fx, Y)
        catch e
            # for PosDefException
            # this seems to happen when θ.σ is extremely large and θ.ℓ is
            # much bigger than the search region dimensions
            println(); println("Error: $e"); @show(θ, X, Y); println()

            f = GP(kernel(θ))
            fx = f(X, θ.σn^2+√eps()+1e-1*θ.σ) # fix by making diagonal a little bigger
            return -logpdf(fx, Y)
        end
    end
end


## Kernel stuff

"""
$SIGNATURES

A simple squared exponential kernel for the GP with parameters θ.
"""
singleKernel(θ) = θ.σ^2 * with_lengthscale(SqExponentialKernel(), θ.ℓ^2)

"""
$SIGNATURES

A multi-task GP kernel, a variety of multi-output GP kernel based on the
Intrinsic Coregionalization Model with a Squared Exponential base kernel and an
output matrix formed from a lower triangular matrix.
"""
multiKernel(θ) = IntrinsicCoregionMOKernel(kernel=with_lengthscale(SqExponentialKernel(), θ.ℓ^2),
                                           B=fullyConnectedCovMat(θ.σ))

fullyConnectedCovNum(num_outputs) = (num_outputs+1)*num_outputs÷2

"""
$SIGNATURES

Creates an output covariance matrix from an array of parameters by filling a lower
triangular matrix.

Inputs:

    - a: parameter vector, must hold (T+1)*T/2 parameters, where T = number of
      outputs
"""
function fullyConnectedCovMat(a)

    T = floor(Int, sqrt(length(a)*2)) # (T+1)*T/2 in matrix

    # cholesky factorization technique to create a free-form covariance matrix
    # that is positive semidefinite
    L = [(v<=u ? a[u*(u-1)÷2 + v] : 0.0) for u in 1:T, v in 1:T]
    A = L*L' # lower triangular times upper

    return A + √eps()*I
end

manyToOneCovNum(num_outputs) = 2*num_outputs - 1

"""
$SIGNATURES

Creates an output covariance matrix from an array of parameters by filling the
first column and diagonal of a lower triangular matrix.

Inputs:

    - a: parameter vector, must hold 2T-1 parameters, where T = number of
      outputs
"""
function manyToOneCovMat(a)

    # cholesky factorization technique to create a free-form covariance matrix
    # that is positive semidefinite
    T = (length(a)+1)÷2 # T on column + T-1 more on diagonal = 2T-1
    L = zeros(T,T) # will be lower triangular
    L[1,1] = a[1]
    for t=2:T
        L[t,1] = a[1 + t-1]
        L[t,t] = a[1 + 2*(t-1)]
    end
    A = L'*L # upper triangular times lower

    return A + √eps()*I
end

"""
$SIGNATURES

Gives the correlations between the first output and any outputs.
"""
function correlations(beliefModel::BeliefModelSimple)
    cov_mat = fullyConnectedCovMat(beliefModel.θ.σ)
    return [cov_mat[i,1]/√(cov_mat[1,1]*cov_mat[i,i]) for i in 2:size(cov_mat, 1)]
end

function correlations(beliefModel::BeliefModelSplit)
    return correlations(beliefModel.combined)
end

end
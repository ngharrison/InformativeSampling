
using LinearAlgebra: tr, diag

"""
A multivariate linear model of ``y`` dependent on ``x`` with parameters ``a``
and ``b`` of the form
```math
Y = a + b^T X
```
X and Y are matrices containing the points as columns.
"""
struct LinearModel
    a
    b
end

"""
Returns a linear model of set of variables Y conditioned on set X.
Requires full mean vector and covariance matrix of the joint normal
distribution.
"""
function LinearModel(μ, Σ, Y, X)
    b = Σ[X,X]\Σ[X,Y] # slope
    a = μ[Y] - b'*μ[X] # intercept
    return LinearModel(a,b)
end

(lm::LinearModel)(X) = lm.a .+ lm.b'*X

rSquared(Σ, Y, X) = tr(Σ[X,Y]'*(Σ[X,X]\Σ[X,Y]))/tr(Σ[Y,Y])

# # three ways:
# vars = vars = diag(Σ)
# R = @. Σ / √(vars * vars')
# c = R[X,Y]
# c'*(R[X,X]\c)
#
# Σ_cond = Σ[Y,Y] - Σ[X,Y]'*(Σ[X,X]\Σ[X,Y])
# 1 - tr(Σ_cond)/tr(Σ[Y,Y])
#
# tr(Σ[X,Y]'*(Σ[X,X]\Σ[X,Y]))/tr(Σ[Y,Y])

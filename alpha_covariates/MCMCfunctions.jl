module MCMCfit
using Distributions: pdf,cdf,quantile,Uniform,Gamma,Normal,MvNormal
using Distances: pairwise, Euclidean
using SpecialFunctions: gamma, loggamma, digamma, trigamma
using LinearAlgebra: Diagonal
using CSV,DataFrames,Dates
using JSON: parsefile
using Plots

export hyperparameter,parameter,mcmc,distmatrix,reparameterize,deparameterize,getα,initvalsλ,ΓΓ_MCMC,readjson

struct parameter
    α::Vector{Float64}
    β₁::Float64
    β₂::Float64
    ρ::Float64
end

struct hyperparameter
    seed::Int64
    nsites::Int64
    ntimes::Int64
    κ₁::Int64
    κ₂::Int64
    τθ::Float64
    τλ::Float64
    niters::Int64
    thin::Int64
    nadapt::Int64
    burnin1::Int64
    burnin2::Int64
    targtaccpt::Vector{Float64}
    boundtargtaccpt::Matrix{Float64}
    w::Vector{Float64}
end

struct mcmc 
    n_it::Int64
    Y::Matrix{Float64}
    covars::Matrix{Float64}
    distm::Matrix{Float64}
    u::Matrix{Float64}
    θ::parameter
    λ::Matrix{Float64}
    hypers::hyperparameter
    savepath::String
end

"""
Get a distance matrix from the sites' locations

# Arguments
 - loc: x,y position matrix of the sites considered
"""
function distmatrix(loc::Matrix{Float64})
    distm = pairwise(Euclidean(),loc,dims=1)
    return distm
end

"""
Get a correlation matrix from a distance matrix

# Arguments
 - distm: Distance matrix of sites considered
 - ρ: Model correlation parameter
"""
function getΣ(distm::Matrix{Float64},ρ::Float64)
    return(exp.(-(distm)/ρ))
end

"""
Get a vector of values of θ from a parameter object for θ

# Arguments
 - θ: a parameter object
"""
function getθvec(θ::parameter)
    return vcat(θ.α,θ.β₁,θ.β₂,θ.ρ)
end

"""
Get a parameter object for θ from a vector of values of θ

# Arguments
 - vec: Model parameters in vector form
 - θ: a parameter object to conform to in the conversion
"""
function getθobj(vec::Matrix{Float64},θ::parameter)
    n = size(θ.α)[1]
    return parameter(vec[1:n],vec[n+1],vec[n+2],vec[n+3])
end

"""
Reparameterize θ to the tildeθ space

# Arguments
 - θ: Model parameters
"""
function reparameterize(θ::parameter)::parameter
    α₀= θ.α[1] + log(θ.β₁) - log(θ.β₂)
    αₖ = θ.α[2:end]
    repar = parameter(vcat(α₀,αₖ),α₀+log(θ.β₁),-log(θ.β₂),log(θ.ρ))
    return repar
end

"""
Convert the reparameterized parameters to their original space

# Arguments
 - tildeθ: Reparameterized version of θ
"""
function deparameterize(tildeθ::parameter)
    α₀ = 2*tildeθ.α[1]-tildeθ.β₁-tildeθ.β₂
    αₖ = tildeθ.α[2:end]
    β₁ = exp(tildeθ.β₁-tildeθ.α[1])
    depar = parameter(vcat(α₀,αₖ),β₁,exp(-tildeθ.β₂),exp(tildeθ.ρ))
    return depar
end

"""
Compute the penalized complexity (PC) prior for β₁

# Arguments
 - β₁: Model parameter
 - κ₁: Penalty rate hyperparameter
"""
function PCpriorβ₁(β₁::Float64,κ₁::Int64)
    KLD(β₁::Float64) = (β₁-1)*digamma(β₁)-loggamma(β₁)
    dKLD(β₁::Float64) = (β₁-1)*trigamma(β₁)
    dβ₁(β₁::Float64) = sqrt(2*KLD(β₁))
    ddβ₁(β₁::Float64) = dKLD(β₁)/sqrt(2*KLD(β₁))
    if β₁==0
        return(0)
    elseif β₁ == 1
        likelih = 0.5*κ₁*(abs(ddβ₁(1-10^(-6)))+abs(ddβ₁(1+10^(-6))))/2
        return likelih
    else 
        likelih = 0.5*κ₁*exp(-κ₁*dβ₁(β₁))*abs(ddβ₁(β₁))
        return likelih
    end
end

"""
Compute the penalized complexity (PC) prior for β₂

# Arguments
 - β₂: Model parameter
 - κ₂: Penalty rate hyperparameter
"""
function PCpriorβ₂(β₂::Float64,κ₂::Int64)
    if β₂>1
        likelih = sqrt(2)*κ₂*exp.(-sqrt(2)*κ₂*(β₂*(β₂-1))^(-1/2))*(β₂-1/2)*(β₂*(β₂-1))^(-3/2)
        return(likelih)
    else
        return 0
    end
end

"""
Convert [α₀,α₁,...,αₖ] to α

# Arguments
 - α_vec: Parameters [α₀,α₁,...,αₖ]
 - covars: Covariates for the model
 - nsites: number of sites considered
 - ntimes: number of times considered for each location
"""
function getα(α_vec::Vector{Float64},covars::Matrix{Float64},nsites,ntimes)
    n = nsites*ntimes
    if size(covars)[1] == n
        α = α_vec[1].*ones(size(covars)[1]) .+ covars*α_vec[2:end]
    else
        α = α_vec[1].*ones(n) .+ repeat(covars*α_vec[2:end],outer=ntimes)
    end
    return transpose(exp.(reshape(α,nsites,ntimes)))
end

"""
Compute the log density of the data evaluated at λ, θ

# Arguments
 - Y: Observed data (n x d matrix)
 - λ: log of latent parameters λ
 - covars: Covariates for the model
 - θ: Reparameterized θ
 - distm: distance matrix between sites
 - indcens: indices of censored locations and times
 - indnocens: indices of non-censored locations and times
 - u: censoring threshold matrix
 - hypers: hyperparameters of the model
"""
function logpost(Y::Matrix{Float64},λ::Matrix{Float64},covars::Matrix{Float64},θ::parameter,distm::Matrix{Float64},indcens::Vector{CartesianIndex{2}},indnocens::Vector{CartesianIndex{2}},u::Matrix{Float64},hypers::hyperparameter)
    ntimes = hypers.ntimes; nsites = hypers.nsites; n = ntimes*nsites
    α_vec= θ.α; β₁ = θ.β₁; β₂ = θ.β₂; ρ = θ.ρ
    Σ = getΣ(distm,ρ)

    α = getα(α_vec,covars,nsites,ntimes)
    logpriorα = sum([log(pdf(Normal(0,10),α_vec[i])) for i in 1:size(α_vec)[1]])
    logpriorβ₁= log(PCpriorβ₁(β₁,hypers.κ₁)); logpriorβ₂= log(PCpriorβ₂(β₂,hypers.κ₂))
    logpriorρ = log(pdf(Gamma(0.01,100.),ρ))
    logdens = logpriorα+logpriorβ₁+logpriorβ₂+logpriorρ
    for i in indnocens logdens += log(pdf(Gamma(β₁,(1.)/λ[i]),Y[i])) end
    for i in indcens logdens += log(cdf(Gamma(β₁,1),u[i]*λ[i])) end
    Zᵢⱼ = [quantile(Normal(0,1),cdf(Gamma(β₂,1/α[i,j]),λ[i,j])) for i in 1:ntimes, j in 1:nsites]
    for i in 1:ntimes logdens += log(pdf(MvNormal(zeros(nsites),Σ),Zᵢⱼ[i,:])) end
    for i in 1:n logdens += log(pdf(Gamma(β₂,1/α[i]),λ[i])) - log(pdf(Normal(0,1),Zᵢⱼ[i])) end
    return logdens
end 

"""
Compute the log density of the data evaluated at tildeλ, tildeθ

# Arguments
 - Y: Observed data (n x d matrix)
 - tildeλ: log of latent parameters λ
 - covars: Covariates for the model
 - tildeθ: Reparameterized θ
 - distm: distance matrix between sites
 - indcens: indices of censored locations and times
 - indnocens: indices of non-censored locations and times
 - u: censoring threshold matrix
 - hypers: hyperparameters of the model
"""
function tildelogpost(Y::Matrix{Float64},logλ::Matrix{Float64},covars::Matrix{Float64},tildeθ::parameter,distm::Matrix{Float64},indcens::Vector{CartesianIndex{2}},indnocens::Vector{CartesianIndex{2}},u::Matrix{Float64},hypers::hyperparameter)
    θ = deparameterize(tildeθ)
    tlogpost = logpost(Y,exp.(logλ),covars,θ,distm,indcens,indnocens,u,hypers)
    tlogpost += sum(logλ) -tildeθ.α[1] + tildeθ.β₁ - tildeθ.β₂ + tildeθ.ρ
    return tlogpost
end

"""
Compute the numerical gradient of the log density of λ evaluated at λ, θ

# Arguments
 - Y: Observed data (n x d matrix)
 - λ: log of latent parameters λ
 - covars: Covariates for the model
 - θ: Reparameterized θ
 - distm: distance matrix between sites
 - indcens: indices of censored locations and times
 - indnocens: indices of non-censored locations and times
 - u: censoring threshold matrix
 - hypers: hyperparameters of the model
"""
function gradlogpost_n(Y::Matrix{Float64},λ::Matrix{Float64},covars::Matrix{Float64},θ::parameter,distm::Matrix{Float64},indcens::Vector{CartesianIndex{2}},indnocens::Vector{CartesianIndex{2}},u::Matrix{Float64},τ::Vector{Float64},hypers::hyperparameter)
    gradlp = copy(λ)
    for i in 1:hypers.ntimes,j in 1:hypers.nsites
        λp = copy(λ); λm = copy(λ)
        λp[i,j] += hypers.δgrad; λm[i,j] -= hypers.δgrad
        gradlp[i,j] = (logpost(Y,λp,covars,θ,distm,indcens,indnocens,u,hypers)-logpost(Y,λm,covars,θ,distm,indcens,indnocens,u,hypers))/(2*hypers.δgrad)
    end
    return gradlp
end

"""
Compute the theoretical gradient of the log density of λ evaluated at λ, θ (loop version)

# Arguments
 - Y: Observed data (n x d matrix)
 - λ: log of latent parameters λ
 - covars: Covariates for the model
 - θ: Reparameterized θ
 - distm: distance matrix between sites
 - indcens: indices of censored locations and times
 - indnocens: indices of non-censored locations and times
 - u: censoring threshold matrix
 - hypers: hyperparameters of the model
"""
function gradlogpost_t(Y::Matrix{Float64},λ::Matrix{Float64},covars::Matrix{Float64},θ::parameter,distm::Matrix{Float64},indcens::Vector{CartesianIndex{2}},indnocens::Vector{CartesianIndex{2}},u::Matrix{Float64},hypers::hyperparameter)
    α = getα(θ.α,covars,hypers.nsites,hypers.ntimes)
    β₁ = θ.β₁; β₂ = θ.β₂; ρ = θ.ρ
    Σ = getΣ(distm,ρ)
    loggrad = Array{Float64}(undef,hypers.ntimes,hypers.nsites)
    
    # Term 1 in gradient computation (Yadav et al. 2021 supplemental material)
    ∇fnocens(i) = β₁/λ[i]-Y[i]
    ∇fcens(i) = u[i]*pdf(Gamma(β₁,1),u[i]*λ[i])/(cdf(Gamma(β₁,1),u[i]*λ[i]))
    # Term 2
    Z(i,j) = quantile(Normal(),cdf(Gamma(β₂,1/α[i,j]),λ[i,j]))
    Zᵢⱼ = [ Z(i,j) for i in 1:hypers.ntimes, j in 1:hypers.nsites]

    D(i,j) = pdf(Gamma(β₂,1/α[i,j]),λ[i,j])/pdf(Normal(),Zᵢⱼ[i,j])
    Dᵢⱼ = [ D(i,j) for i in 1:hypers.ntimes, j in 1:hypers.nsites]

    ∇term2 = Dᵢⱼ.*transpose((inv(Σ)*transpose(Zᵢⱼ)))
    # Term 3
    ∇term3(i) = (β₂-1)/λ[i]-α[i]
    # Term 4 = Zᵢⱼ*Dᵢⱼ
    for i in indnocens loggrad[i] = ∇fnocens(i)-∇term2[i]+∇term3(i)+Zᵢⱼ[i]*Dᵢⱼ[i] end
    for i in indcens loggrad[i] = ∇fcens(i)-∇term2[i]+∇term3(i)+Zᵢⱼ[i]*Dᵢⱼ[i] end
    return loggrad
end

"""
Compute the theoretical gradient of the log density of λ evaluated at λ, θ (Vectorized version)

# Arguments
 - Y: Observed data (n x d matrix)
 - λ: log of latent parameters λ
 - covars: Covariates for the model
 - θ: Reparameterized θ
 - distm: distance matrix between sites
 - indcens: indices of censored locations and times
 - indnocens: indices of non-censored locations and times
 - u: censoring threshold matrix
 - hypers: hyperparameters of the model
"""
function gradlogpost_t1(Y::Matrix{Float64},λ::Matrix{Float64},covars::Matrix{Float64},θ::parameter,distm::Matrix{Float64},indcens::Vector{CartesianIndex{2}},indnocens::Vector{CartesianIndex{2}},u::Matrix{Float64},hypers::hyperparameter)
    α = getα(θ.α,covars,hypers.nsites,hypers.ntimes)
    β₁ = θ.β₁; β₂ = θ.β₂; ρ = θ.ρ
    Σ = getΣ(distm,ρ)
    loggrad = Array{Float64}(undef,hypers.ntimes,hypers.nsites)
    
    # Term 1 in gradient computation (Yadav et al. 2021 supplemental material)
    @. loggrad[indnocens] = β₁/λ[indnocens]-Y[indnocens]
    @. loggrad[indcens] = u[indcens]*pdf(Gamma(β₁,1),u[indcens]*λ[indcens])/(cdf(Gamma(β₁,1),u[indcens]*λ[indcens]))
    # Term 2
    Zᵢⱼ = quantile.(Normal(),cdf.(Gamma.(β₂,(1.)./α),λ))
    Dᵢⱼ = pdf.(Gamma.(β₂,(1.)./α),λ)./pdf.(Normal(),Zᵢⱼ)
    ∇term2 = Dᵢⱼ.*transpose((inv(Σ)*transpose(Zᵢⱼ)))
    # Term 3
    ∇term3 = (β₂-1.)./λ.-α
    # Term 4 
    Pᵢⱼ = Zᵢⱼ.*Dᵢⱼ
    @. loggrad = loggrad - ∇term2 + ∇term3 + Pᵢⱼ
    return loggrad
end

"""
Compute the gradient of the log density of λ evaluated at tildeλ, tildeθ

# Arguments
 - Y: Observed data (n x d matrix)
 - logλ: log of latent parameters λ
 - covars: Covariates for the model
 - tildeθ: Reparameterized θ
 - distm: distance matrix between sites
 - indcens: indices of censored locations and times
 - indnocens: indices of non-censored locations and times
 - u: censoring threshold matrix
 - hypers: hyperparameters of the model
"""
function tildegradlogpost(Y::Matrix{Float64},logλ::Matrix{Float64},covars::Matrix{Float64},tildeθ::parameter,distm::Matrix{Float64},indcens::Vector{CartesianIndex{2}},indnocens::Vector{CartesianIndex{2}},u::Matrix{Float64},hypers::hyperparameter)
    θ = deparameterize(tildeθ)
    λ = exp.(logλ)
    return(gradlogpost_t1(Y,λ,covars,θ,distm,indcens,indnocens,u,hypers).*λ.+1)
end

"""
Obtain new candidate/proposal for tildeθ

# Arguments
 - tildeθ: Reparameterized θ
 - hypers: hyperparameters of the model
"""
function θproposal(tildeθ::parameter,τ::Vector{Float64})
    μ = getθvec(tildeθ)
    Σ = Diagonal(τ[1] * ones(size(μ)[1]))
    proptilde = rand(MvNormal(μ,Σ),1)
    return getθobj(proptilde,tildeθ)
end

"""
Obtain new candidate/proposal for tildeλ

# Arguments
 - Y: Observed data (n x d matrix)
 - logλ: log of latent parameters λ
 - ∇logλ: gradient of the log-density evaluated at logλ, ∇logλ = tildegradlogpost(...)
 - covars: Covariates for the model
 - tildeθ: Reparameterized θ
 - distm: distance matrix between sites
 - indcens: indices of censored locations and times
 - indnocens: indices of non-censored locations and times
 - u: censoring threshold matrix
 - hypers: hyperparameters of the model
"""
function λproposal(logλ::Matrix{Float64},∇logλ::Matrix{Float64},τ::Vector{Float64},hypers::hyperparameter)
    μ = vec(logλ .+ τ[2]*∇logλ/2)
    Σ = Diagonal(τ[2] .* ones(size(μ)[1]))
    newλ = reshape(rand(MvNormal(μ,Σ),1),hypers.ntimes,hypers.nsites)
    return newλ
end

"""
Compute density of the λ proposal distribution at logλprop

# Arguments
 - Y: Observed data (n x d matrix)
 - logλ: log of latent parameters λ
 - ∇logλ: gradient of the log-density evaluated at logλ, ∇logλ = tildegradlogpost(...)
 - logλprop: New logλ candidate
 - covars: Covariates for the model
 - tildeθ: Reparameterized θ
 - distm: distance matrix between sites
 - indcens: indices of censored locations and times
 - indnocens: indices of non-censored locations and times
 - u: censoring threshold matrix
 - hypers: hyperparameters of the model
"""
function logdensλprop(Y::Matrix{Float64},logλ::Matrix{Float64},∇logλ::Matrix{Float64},logλprop::Matrix{Float64},covars::Matrix{Float64},tildeθ::parameter,distm::Matrix{Float64},indcens::Vector{CartesianIndex{2}},indnocens::Vector{CartesianIndex{2}},u::Matrix{Float64},τ::Vector{Float64},hypers::hyperparameter)
    μ = logλ .+ τ[2]*∇logλ/2
    logdensλ = 0
    for i in 1:hypers.nsites*hypers.ntimes logdensλ += log(pdf(Normal(μ[i],sqrt(τ[2])),logλprop[i])) end
    return logdensλ
end

""" Vectorized version of above function (Not faster) """
function logdensλprop1(Y::Matrix{Float64},logλ::Matrix{Float64},∇logλ::Matrix{Float64},logλprop::Matrix{Float64},covars::Matrix{Float64},tildeθ::parameter,distm::Matrix{Float64},indcens::Vector{CartesianIndex{2}},indnocens::Vector{CartesianIndex{2}},u::Matrix{Float64},τ::Vector{Float64},hypers::hyperparameter)
    μ = logλ .+ τ[2]*∇logλ/2
    logdensλ = sum(log.(pdf.(Normal.(μ,sqrt(τ[2])),logλprop)))
    return logdensλ
end

"""
Adapt the τθ and τλ hyperparameters

# Arguments
 - τ: Vector of [τθ, τλ]
 - it: MCMC iteration number
 - hypers: hyperparameters
 - accptcount: Proposal acceptation count vector for τθ, τλ
# Outputs
 - τ: updated τ vector
 - accptcount: same accptcount or [0,0] if τ was updated
 - boolean value indicating if τ was updated
"""
function adaptτ(τ::Array{Float64,1},it::Int64,hypers::hyperparameter,accptcount::Array{Int64,1})
    if mod(it,hypers.nadapt) == 0
        if it < hypers.burnin1 +1
            τ = exp.((accptcount/hypers.nadapt .- hypers.targtaccpt)./hypers.w).*τ
        elseif it < hypers.burnin2 +1
            if accptcount[1]/hypers.nadapt < hypers.boundtargtaccpt[1,1] || accptcount[1]/hypers.nadapt > hypers.boundtargtaccpt[2,1]
                τ[1] = exp((accptcount[1]/hypers.nadapt-hypers.targtaccpt[1])/hypers.w[1])*τ[1]
            end 
            if accptcount[2]/hypers.nadapt < hypers.boundtargtaccpt[1,2] || accptcount[2]/hypers.nadapt > hypers.boundtargtaccpt[2,2]
                τ[2] = exp((accptcount[2]/hypers.nadapt-hypers.targtaccpt[2])/hypers.w[2])*τ[2]
            end 
        else
            println(string("Iteration: ",it,", Prob accept [θ,λ]: ", accptcount/hypers.nadapt, ", τ = [τθ,τλ]: ", τ))
            return τ,[0,0],1
        end
        println(string("Iteration: ",it,", Prob accept [θ,λ]: ", accptcount/hypers.nadapt, ", τ = [τθ,τλ]: ", τ))
        return τ,[0,0],1
    else
        return τ,accptcount,0
    end
end

"""
Increment the acceptation counter if proposal is accepted

# Arguments
 - accptcount: Proposal acceptation count vector for τθ, τλ
 - θλ: Indicator of which counter to increment, 1 for θ, 2 for λ
"""
function incrementaccptcount(accptcount::Array{Int64,1},θλ::Int64)
    accptcount[θλ] += 1
    return accptcount
end

"""
Obtain sensible initial values for λ

# Arguments
 - θ : parameter object of initial θ values
 - covars: Covariates for the model
 - hypers: Hyperparameters of the model
"""
function initvalsλ(θ::parameter,covars::Matrix{Float64},hypers::hyperparameter)
    α = getα(θ.α,covars,hypers.nsites,hypers.ntimes)
    initλ = quantile.(Gamma.(θ.β₂,(1.)./α),0.5)
    return initλ
end 

"""
Perform MCMC sampling to obtain postetior chains for θ

# Arguments
 - init : mcmc object containing all relevent parameters to run the model

# Output
 - Posterior chains for the model
"""
function ΓΓ_MCMC(init::mcmc)
    Y=init.Y; covars=init.covars; distm = init.distm; u = init.u
    hypers = init.hypers; τ = [hypers.τθ, hypers.τλ]
    θ = init.θ; θvec = getθvec(θ); λ = init.λ
    tildeθ = reparameterize(θ); tildeλ = log.(λ)

    # Censoring indices
    indcens = findall(x->x==1,Y.<u)
    indnocens = findall(x->x==1,Y.>u)

    # Save parameter and latent parameter chains
    chains = Array{Float64}(undef,max(convert(Int64,floor(init.n_it/hypers.thin)),2),hypers.nsites*hypers.ntimes+size(θvec)[1])
    chains[1,:] = vcat(getθvec(tildeθ),vec(tildeλ))
    
    # Save τ values
    τmatrix = Array{Float64}(undef,max(convert(Int64,floor(init.n_it/hypers.nadapt)),2),2); τmatrix[1,:] = τ 

    thincount = 1 # Count the number of thinned parameter and latent params saved
    accptcount = [0,0] #[count accept θ, count accept λ]
    τmatrixcount = 1 # Counter for number of τ updates

    logliktildeθ = 0
    changeθind = 0 # 1 if there was a change for θ in previous iteration
    ∇logtildeλ = tildegradlogpost(Y,tildeλ,covars,tildeθ,distm,indcens,indnocens,u,hypers)

    for i in 1:(init.n_it-1)
        if mod(i,10000) == 0
            display(Plots.plot(chains[:,1:size(θvec)[1]]))
        end

        # Update τ after 'nadapt' iterations and save resulting values
        τ,accptcount,updtind = adaptτ(τ,i,hypers,accptcount)
        if updtind == 1
            τmatrixcount += 1
            τmatrix[τmatrixcount,:] = τ
        end

        # Proposal of new tildeθ
        proptildeθ = θproposal(tildeθ,τ)

        # loglikelihood of tildeθ and proptildeθ
        if i == 1
            logliktildeθ = tildelogpost(Y,tildeλ,covars,tildeθ,distm,indcens,indnocens,u,hypers)
        else
            logliktildeθ = logliktildeλ # Same value and already computed in last iteration
        end
        loglikproptildeθ = tildelogpost(Y,tildeλ,covars,proptildeθ,distm,indcens,indnocens,u,hypers)

        if log(rand(Uniform(),1)[1]) < loglikproptildeθ - logliktildeθ
            logliktildeθ = loglikproptildeθ # So there is no need to recompute logliktildeλ after
            tildeθ = proptildeθ
            accptcount[1] += 1
            changeθind = 1
        end
        
        # Proposal of new tildeλ
        if changeθind == 1
            ∇logtildeλ = tildegradlogpost(Y,tildeλ,covars,tildeθ,distm,indcens,indnocens,u,hypers)
            changeθind = 0
        end
        proptildeλ = λproposal(tildeλ,∇logtildeλ,τ,hypers)

        # loglikelihood of tildeλ, proptildeλ and proposal
        logliktildeλ = logliktildeθ # Same value and already computed in this iteration
        loglikproptildeλ = tildelogpost(Y,proptildeλ,covars,tildeθ,distm,indcens,indnocens,u,hypers)

        logdenscurtoprop = logdensλprop(Y,tildeλ,∇logtildeλ,proptildeλ,covars,tildeθ,distm,indcens,indnocens,u,τ,hypers)
        ∇logproptildeλ = tildegradlogpost(Y,proptildeλ,covars,tildeθ,distm,indcens,indnocens,u,hypers)
        logdensproptocur = logdensλprop(Y,proptildeλ,∇logproptildeλ,tildeλ,covars,tildeθ,distm,indcens,indnocens,u,τ,hypers)

        logratio = loglikproptildeλ + logdensproptocur - logliktildeλ - logdenscurtoprop
        if log(rand(Uniform(),1)[1]) < logratio
            logliktildeλ = loglikproptildeλ
            tildeλ = proptildeλ
            ∇logtildeλ = ∇logproptildeλ
            accptcount[2] += 1
        end

        # Save the new parameters and latent parameters
        if mod(i,hypers.thin) == 0
            thincount += 1
            chains[thincount,:] = vcat(getθvec(tildeθ),vec(tildeλ))
        end
    end
    dt = replace(replace(string(Dates.now()),"."=>"-"),":"=>"-")
    fpath = string(init.savepath,dt,".csv")
    CSV.write(fpath,DataFrame(chains,:auto))
    return chains,τmatrix
end

function readjson(pathjson::String)
    json = parsefile(pathjson)

    hyps = get(json,"hyperparameters",0)
    sim_or_real = get(json,"sim_or_real",0)
    sim = get(json,"simulation",0)
    realdata = get(json,"real_data",0)
    hypers = hyperparameter(get(hyps,"seed",0),
                            get(hyps,"nsites",0),
                            get(hyps,"ntimes",0),
                            get(hyps,"kappa1",0),
                            get(hyps,"kappa2",0),
                            get(hyps,"tau_theta",0),
                            get(hyps,"tau_lambda",0),
                            get(hyps,"n_iter",0),
                            get(hyps,"n_thin",0),
                            get(hyps,"nadapt",0),
                            get(hyps,"burnin1",0),
                            get(hyps,"burnin2",0),
                            get(hyps,"targt_accept_tau",0),
                            hcat(get(hyps,"bound_targt_accpt_tau_theta",0),get(hyps,"bound_targt_accpt_tau_lambda",0)),
                            get(hyps,"w",0))

    if get(sim,"yesno",false) & get(realdata,"yesno",false)
        println("Only one of 'simulation' or 'real_data' must be true.")
        return -1
    end

    if get(sim,"yesno",false) # Perform model fitting on simulated data
        trueθdict = get(sim,"true_theta",0)
        initθdict = get(sim_or_real,"init_theta",0)
        trueθ = parameter(get(trueθdict,"alpha",0),
                        get(trueθdict,"beta1",0),
                        get(trueθdict,"beta2",0),
                        get(trueθdict,"rho",0))
        initθ = parameter(get(initθdict,"alpha",0),
                        get(initθdict,"beta1",0),
                        get(initθdict,"beta2",0),
                        get(initθdict,"rho",0))
        return true,hypers,sim_or_real,initθ,trueθ
    elseif get(realdata,"yesno",false) #  Perform model fitting on real data
        print("Not done yet")
        return false,hypers,sim_or_real,initθ,realdata
    else
        println("One of simulation yesno or real_data yesno must be true.")
    end

end

end

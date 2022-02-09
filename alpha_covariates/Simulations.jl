module simulations
include("MCMCfunctions.jl")
using .MCMCfit: parameter, getα
using Distributions: pdf,cdf,quantile,FDist,Gamma,Normal,MvNormal,Uniform
using Plots: plot, abline!
using PlotlyJS, DataFrames, CSV

export locmatrix, simulation, testYmargins, boxplot

"""
Generate artifical sites' locations

# Arguments
 - nsites : number of artificial sites
"""
function locmatrix(nsites::Int64)
    locs = rand(Uniform(),2*nsites)
    return reshape(locs,nsites,2)
end

"""
Generate artifical site locations

# Arguments
 - nsites : number of artificial sites
 - ntimes : number of artificial temporal observations
 - distm : distance matrix for the artificial sites
 - trueθ : True model parameters to simulate from (and hopefully recover with the model)
 - covars: Covariates for the model
"""
function simulation(nsites::Int64,ntimes::Int64,distm::Matrix{Float64},trueθ,covars)
    n = nsites*ntimes
    if size(covars)[1] == n
        α = trueθ.α[1].*ones(size(covars)[1]) .+ covars*trueθ.α[2:end]
    else
        α = trueθ.α[1].*ones(n) .+ repeat(covars*trueθ.α[2:end],outer=ntimes)
    end
    α = exp.(transpose(reshape(α,nsites,ntimes)))
    Σ = exp.(-(distm)./trueθ.ρ)
    X₁ = reshape(rand(Gamma(trueθ.β₁,1),n),ntimes,nsites)
    mvnorm = transpose(rand(MvNormal(zeros(nsites),Σ),ntimes))
    pnorm = cdf.(Normal(0,1),mvnorm)
    X₂ = [quantile(Gamma(trueθ.β₂,(1.)/α[i,j]),pnorm[i,j]) for i in 1:size(α)[1], j in 1:size(α)[2]]
    return X₁./X₂, X₁, X₂
end

"""
Obtain a Q-Q plot of marginal observations

# Arguments
 - Y : Observations
 - α : result of α₀ + [α₁,...,αₖ]*covars
 - θ : fitter parameters
 - column : site to examine marginally
 - qqplot : boolean to display the Q-Q plot
"""
function testYmargins(Y,covars,θ,column,qqplot=true)
    α = getα(θ.α,covars,size(Y)[2],size(Y)[1])
    probs = collect(1:size(Y)[1])/(size(Y)[1]+1)
    term1 = (θ.β₁/θ.β₂)*quantile(FDist(2*θ.β₁,2*θ.β₂),probs)
    theor = [α[i,column]*term1 for i in 1:size(Y)[1]]
    p = plot(sort(Y[:,column]),theor,title=string("Site ", column), 
            seriestype = :scatter,legend=:none,size=(1200,1200))
    abline!(p,1,0)
    if qqplot == true
        display(p)
    end
    return(p)

end

function boxplot(Y)
    Ydf = DataFrame(Y,:auto)
    Ydf = stack(Ydf,1:size(Y)[2])
    display(PlotlyJS.plot(Ydf, x=:variable, y=:value, kind="box"))
end

function simΓΓfitted(postchains::Matrix{Float64},θ)
end

end #module

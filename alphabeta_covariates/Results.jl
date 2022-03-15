module results
    
using Plots: plot, abline!
using Statistics
using Distributions: pdf, FDist
using SpecialFunctions: gamma
include("MCMCfunctions.jl")
using .MCMCfit: parameter, getα
include("Simulations.jl")
using .simulations: testYmargins

export plotθλ, compareQQ, preddens, getQQ

function getθvec(θ)
    return vcat(θ.α,θ.β₁,θ.β₂,θ.ρ)
end

function plotθλ(chains::Matrix{Float64},trueθ,trueλ,λind)
    n = size(trueθ.α)[1] + size(trueθ.β₂)[1] + 2 + size(λind)[1]
    txt = ["α"*Char(0x2080 + i) for i in 0:(size(trueθ.α)[1]-1)]
    txt = vcat(txt,"β₁")
    txt = vcat(txt,["β₂"*Char(0x2080 + i) for i in 0:(size(trueθ.β₂)[1]-1)])
    txt = vcat(txt,"ρ")
    p = fill(plot(),n,1)
    for k in 1:size(txt)[1]
        p[k] = plot(chains[:,k],title=txt[k])
        abline!(0,mean(chains[:,k]),linecolor=[:red])
        abline!(0,quantile(chains[:,k],0.025),linecolor=[:orange])
        abline!(0,quantile(chains[:,k],0.975),linecolor=[:orange])
        abline!(0,vcat(getθvec(trueθ),vec(log.(trueλ)))[k],linecolor=[:black])
    end
    for k in 1:size(λind)[1]
        j = λind[k]
        subscript = join(Char(0x2080 + d) for d in reverse!(digits(j)))
        p[size(txt)[1]+k] = plot(chains[:,j],title="λ"*subscript)
        abline!(0,mean(chains[:,j]),linecolor=[:red])
        abline!(0,quantile(chains[:,j],0.025),linecolor=[:orange])
        abline!(0,quantile(chains[:,j],0.975),linecolor=[:orange])
        abline!(0,vcat(getθvec(trueθ),vec(log.(trueλ)))[j],linecolor=[:black])
    end
    display(plot(p...,layout =(3,4),size=(2400,1800)))
end

function compareQQ(Y,covars,trueθ,fittedθ,hypers)
    p = fill(plot(),size(Y)[2]*2,1)
    for i in 0:(size(Y)[2]-1)
        p[2*i+1] = testYmargins(Y,covars,trueθ,i+1,hypers,false)
        p[2*i+2] = testYmargins(Y,covars,fittedθ,i+1,hypers,false)
    end
    for i in 0:convert(Int64,size(Y)[2]/5)
        display(plot(p[collect((8*i+1):(8*i+8))]...,layout=(4,2)))
    end
end

function getQQ(Y,covars,fittedθ,hypers)
    p = fill(plot(),size(Y)[2],1)
    for i in 1:(size(Y)[2])
        p[i] = testYmargins(Y,covars,fittedθ,i,hypers,false)
    end
    display(plot(p...))
end

function preddens(fittedθ,covars,hypers,ind,range)
    α = getα(fittedθ.α,covars[:,hypers.covarsα],hypers.nsites,hypers.ntimes)
    β₂ = getβ₂(fittedθ.β₂,covars,hypers)
    β₁ = fittedθ.β₁;
    αᵢⱼ = α[ind[1],ind[2]]
    f(x) = αᵢⱼ^(-β₁)*gamma(β₁+β₂)*(1+x/αᵢⱼ)^(-β₁-β₂)*x^(β₁-1)/(gamma(β₁)*gamma(β₂))
    display(plot(f,range[1],range[2]))
end

end
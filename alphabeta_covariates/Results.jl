module results
    
using Plots: plot, abline!
using Statistics
using Distributions: pdf,cdf,FDist,Normal,MvNormal,Gamma
using SpecialFunctions: gamma
include("MCMCfunctions.jl")
using .MCMCfit: parameter,getα
include("Simulations.jl")
using .simulations: testYmargins

export plotθλ, compareQQ, preddens, getQQ, posterior_pred

function getθvec(θ)
    return vcat(θ.α,θ.β₁,θ.β₂,θ.ρ)
end

function getθobj(vec,θ)
    nₐ = size(θ.α)[1]
    nᵦ₂ = size(θ.β₂)[1]
    return parameter(vec[1:nₐ],vec[nₐ+1],vec[nₐ+2:nₐ+nᵦ₂+1],vec[nₐ+nᵦ₂+2])
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

function preddens(fittedθ,covarsα,covarsβ₂,range)
    α = fittedθ.α[1] + sum(fittedθ.α[2:end].*covarsα)
    β₂ = fittedθ.β₂[1] + sum(ifelse(size(fittedθ.β₂)[1]==1,0,fittedθ.β₂[2:end].*covarsβ₂))
    β₁ = fittedθ.β₁;
    f(x) = α^(-β₁)*gamma(β₁+β₂)*(1+x/α)^(-β₁-β₂)*x^(β₁-1)/(gamma(β₁)*gamma(β₂))
    display(plot(f,range[1],range[2]))
end

function posterior_pred(burnin,chains,covarsα,fittedθ,dist_m)
    sizeθ = size(getθvec(fittedθ))[1]
    chains = chains[burnin:end,1:sizeθ]
    d = size(dist_m)[1]
    Ys = Array{Float64}(undef,size(chains)[1],d)
    for i in 1:1 #size(chains)[1]
        #generate MVN vector
        θ = getθobj(chains[i,1:sizeθ],fittedθ)
        α = θ.α[1] .+ covarsα*θ.α[2:end]
        μ = zeros(d)
        Σ = exp.(-dist_m/θ.ρ)
        MvN = rand(MvNormal(μ,Σ),1)
        u_scale = cdf.(Normal(0,1),MvN)
        λ = [quantile.(Gamma(θ.β₂,(1.)/θ.α[i]),u_scale[i]) for i in 1:d]
        Γ₁ = rand(Gamma(trueθ.β₁,1),d)
        Y = Γ₁./λ
    end
end

end
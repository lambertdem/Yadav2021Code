using PlotlyJS, DataFrames, BenchmarkTools, Dates, CSV
using Distributions: pdf,cdf,quantile,Uniform,Gamma,Normal,MvNormal,FDist
using Random: rand,seed!
using Plots: plot, abline!

include("MCMCfunctions.jl")
using .MCMCfit: hyperparameter,parameter,mcmc,distmatrix,reparameterize,deparameterize,initvalsλ,ΓΓ_MCMC,readjson,gridsearchdensity,getβ₂
include("Simulations.jl")
using .simulations: locmatrix, simulation, testYmargins, boxplot
include("Results.jl")
using .results: plotθλ, compareQQ, preddens

jsonfilenm = "RunAlphaBeta1"
jsonpath = string("C:\\Users\\lambe\\Documents\\McGill\\Masters\\Thesis\\Yadav2021code\\Runs\\",jsonfilenm,".json")
sim,hypers,sim_or_real,initθ = readjson(jsonpath)

######################################
# If fitting model on simulated data #
######################################
if sim == true
    seed!(hypers.seed)

    # Create a location matrix for artificial weather sites
    m = locmatrix(hypers.nsites)
    distm= distmatrix(m)

    # Plot site locations
    # display(plot(m[:,1],m[:,2], seriestype = :scatter, title = "Locations"))

    # Create an artifical covariance matrix with 3 predictor variables
    covars = hcat(m[:,1],m[:,2],rand(Normal(0,1),hypers.nsites))

    # Get remaining simulation JSON data
    _,_,_,_,trueθ = readjson(jsonpath)

    # Generate simulated Y data
    Y,X₁,trueλ = simulation(distm,trueθ,covars,hypers)

    # Create censoring threshold
    # u = reshape(repeat([3.],hypers.ntimes*hypers.nsites),hypers.ntimes,hypers.nsites)
    u = reshape(repeat([100.,0.,0.,100.,15.,15.,11.,33.,166.,13.,314.,123.,17.,11.,8.,7.,169.,73.,0.,0.],inner=hypers.ntimes),
                hypers.ntimes,hypers.nsites)

#################################    
# If fitting model on real data #
#################################
else
    println("Not developped yet")

    # Get remaining JSON data
    _,_,_,realdata = readjson(jsonpath)

end

# Boxplot of Y
boxplot(Y)
# Test margins against trueθ
# testYmargins(Y,covars,trueθ,12,hypers,true)

# Get sensible initial λ values based on initθ
λ = initvalsλ(initθ,covars,hypers)

# filenm = "Run3_2022-02-09T12-46-43-964.csv"
# savepath = get(sim_or_real,"save_path",0)
# prev_chains = Matrix{Float64}(CSV.read(string(savepath,filenm),DataFrame))[:,8:end]
# λ = exp.(reshape([mean(prev_chains[:,i]) for i in 1:size(prev_chains)[2]],hypers.ntimes,hypers.nsites))

# Get the censoring indices
indcens = findall(x->x==1,Y.<u)
indnocens = findall(x->x==1,Y.>u)

# Create a mcmc object to run the experiment
mcmc1 = mcmc(hypers.niters, # Number of iterations
            Y,      # Observed data
            covars, # Covariates for the model
            distm,  # Distance matrix
            u,      # Censoring threshold
            initθ,      # Model parameters
            λ,      # Latent variables
            hypers, # hyperparameters
            string(get(sim_or_real,"save_path",0),jsonfilenm,"_")) # Save path

# mcmc2 = mcmc(hypers.niters, # Number of iterations
#             Y,      # Observed data
#             covars, # Covariates for the model
#             distm,  # Distance matrix
#             u,      # Censoring threshold
#             trueθ,      # Model parameters
#             trueλ,      # Latent variables
#             hypers, # hyperparameters
#             string(get(sim_or_real,"save_path",0),jsonfilenm,"_")) # Save path

# param = 2
# loglik,pars=gridsearchdensity(mcmc2,param)
# m = hcat(pars,loglik)
# m = m[findall(!isnan, m[:,2]),:]
# display(plot(m[:,1],m[:,2],legend=:none,seriestype = :scatter))
# reparameterize(trueθ)

@time chains,τs = ΓΓ_MCMC(mcmc1)
chains

# filenm = "Run1_2022-02-04T19-27-47-847.csv"
# filenm = "Run3_2022-02-06T15-27-39-240.csv"
# filenm = "Run3_2022-02-06T18-47-59-678.csv"
# filenm = "Run4_2022-02-07T18-39-00-877.csv"
# savepath = get(sim_or_real,"save_path",0)
# chains = Matrix{Float64}(CSV.read(string(savepath,filenm),DataFrame))

plotθλ(chains,reparameterize(trueθ),trueλ,[111,222,555,777])

fittedtildeθ = parameter([mean(chains[:,i]) for i in 1:size(trueθ.α)[1]],
                    mean(chains[:,size(trueθ.α)[1]+1]),
                    [mean(chains[:,size(trueθ.α)[1]+i+1]) for i in 1:size(trueθ.β₂)[1]],
                    mean(chains[:,size(trueθ.α)[1]+size(trueθ.β₂)[1]+2]))

fittedθ = deparameterize(fittedtildeθ)
trueθ

compareQQ(Y,covars,trueθ,fittedθ)

preddens(fittedθ,covars,hypers,[1,1],[0,300])

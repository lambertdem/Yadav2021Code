using PlotlyJS, DataFrames, BenchmarkTools, Dates, CSV, Statistics
using Distributions: pdf,cdf,quantile,Uniform,Gamma,Normal,MvNormal,FDist
using Random: rand,seed!
using Plots: plot, abline!

include("MCMCfunctions.jl")
using .MCMCfit: hyperparameter,parameter,mcmc,distmatrix,reparameterize,deparameterize,initvalsλ,ΓΓ_MCMC,readjson,plotθ
include("Simulations.jl")
using .simulations: locmatrix, simulation, testYmargins, boxplot
include("Results.jl")
using .results: plotθλ, getQQ, compareQQ, preddens, posterior_pred

jsonfilenm = "RunHQ2_u4" # Do NOT add .json extension
runspath = "C:\\Users\\lambe\\Documents\\McGill\\Masters\\Thesis\\Yadav2021code\\Runs\\"
jsonpath = string(runspath,jsonfilenm,".json")
sim,hypers,sim_or_real,initθ = readjson(jsonpath)

######################################
# If fitting model on simulated data #
######################################
if sim == true
    seed!(hypers.seed)

    # Create a location matrix for artificial weather sites
    loc_m = locmatrix(hypers.nsites)
    distm= distmatrix(loc_m)

    # Plot site locations
    # display(plot(loc_m[:,1],loc_m[:,2], seriestype = :scatter, title = "Locations"))

    # Create an artifical covariance matrix with 3 predictor variables
    covars_path = get(sim_or_real,"covars_path",0)
    covars = Matrix{Float64}(CSV.read(covars_path,DataFrame))
    
    # Get remaining simulation JSON data
    _,_,_,_,trueθ = readjson(jsonpath)
    display(trueθ)
    display(reparameterize(trueθ))

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
    seed!(hypers.seed)

    # Get remaining JSON data
    _,_,_,_,realdata = readjson(jsonpath)

    # Get the location matrix for weather sites
    locm_path = get(sim_or_real,"loc_m_path",0)
    csv_loc_m = CSV.read(locm_path,DataFrame,header=0)[:,2:3]
    loc_m = Matrix{Float64}(csv_loc_m)
    distm= distmatrix(loc_m)

    # Plot site locations
    # display(plot(loc_m[:,1],loc_m[:,2], seriestype = :scatter, title = "Locations"))

    # Get predictive variables matrix
    covars_path = get(sim_or_real,"covars_path",0)
    covars_csv = CSV.read(covars_path,DataFrame)
    covars = Matrix{Float64}(covars_csv)

    # Get Y data
    Y_path = get(realdata,"data_path",0)
    Y = Matrix{Float64}(CSV.read(Y_path,DataFrame))

    # Create censoring threshold
    u_path = get(sim_or_real,"u_path",0)
    u = reshape(repeat(Matrix{Float64}(CSV.read(u_path,DataFrame)),hypers.ntimes),hypers.ntimes,hypers.nsites)
end

# Boxplot of Y
# boxplot(Y)
# Test margins against trueθ
# testYmargins(Y,covars,trueθ,12,hypers,true)

# Get sensible initial λ values based on initθ
λ = initvalsλ(initθ,covars,hypers)

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

# @time chains,τs = ΓΓ_MCMC(mcmc1)
# chains

filenm = "RunHQ2_2022-03-14T21-20-13-905.csv"
# filenm = "RunHQ3_2022-03-18T13-40-04-923.csv"
# filenm = "RunHQ2_2022-03-21T19-52-12-796.csv"
filenm = "RunHQ2a_2022-03-21T21-42-50-842.csv"
filenm = "RunHQ2NoCovs_2022-03-22T14-37-46-249.csv"
filenm = "RunHQ2_3mods_2022-03-22T17-38-04-427.csv"
filenm = "RunHQ2_u4_2022-03-27T15-42-25-899.csv"
filenm = "RunHQ2_u4_2022-03-28T20-13-25-865.csv"
filenm = "RunHQ2_u2_2022-03-28T00-42-31-884.csv"

savepath = get(sim_or_real,"save_path",0)
chains = Matrix{Float64}(CSV.read(string(savepath,filenm),DataFrame))

plotθ(chains,initθ,[6,7,8,9,10])

burn = 1500
fittedtildeθ = parameter([mean(chains[burn:end,i]) for i in 1:size(initθ.α)[1]],
                    mean(chains[burn:end,size(initθ.α)[1]+1]),
                    [mean(chains[burn:end,size(initθ.α)[1]+i+1]) for i in 1:size(initθ.β₂)[1]],
                    mean(chains[burn:end,size(initθ.α)[1]+size(initθ.β₂)[1]+2]))

fittedθ = deparameterize(fittedtildeθ)

vars = [Statistics.var(chains[100:end,i]) for i in 1:5]
vars/minimum(vars)

# α = fittedθ.α[1] + 40*fittedθ.α[2]
# β₂ = fittedθ.β₂[1]
# exp1 = β₂/α
# expY = fittedθ.β₁/exp1


##############
# Simulation #
##############

# plotθλ(chains,reparameterize(trueθ),trueλ,[111,222,555,777])
# compareQQ(Y,covars,trueθ,fittedθ,hypers)

#############
# Real Data #
#############
getQQ(Y,covars,fittedθ,hypers)

covarsα = [20.0,2.0,1.0,1.0,10.0]
# covarsβ₂ = []
# preddens(fittedθ,covarsα,covarsβ₂,[0,100])

simY = posterior_pred(burn,chains,covarsα,fittedθ,distm)
boxplot(simY)

[quantile(simY[:,i],0.90) for i in 1:size(simY)[2]]
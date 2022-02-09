using PlotlyJS, DataFrames, BenchmarkTools, Dates, CSV
using Distributions: pdf,cdf,quantile,Uniform,Gamma,Normal,MvNormal,FDist
using Random: rand,seed!
using Plots: plot, abline!

include("MCMCfunctions.jl")
using .MCMCfit: hyperparameter,parameter,mcmc,distmatrix,reparameterize,deparameterize,initvalsλ,ΓΓ_MCMC,readjson
include("Simulations.jl")
using .simulations: locmatrix, simulation, testYmargins, boxplot
include("Results.jl")
using .results: plotθλ, compareQQ, preddens

jsonfilenm = "Run3"
jsonpath = string("C:\\Users\\lambe\\Documents\\McGill\\Masters\\Thesis\\MScThesisCode\\Runs\\",jsonfilenm,".json")
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
    display(plot(m[:,1],m[:,2], seriestype = :scatter, title = "Locations"))

    # Create an artifical covariance matrix with 3 predictor variables
    covars = hcat(m[:,1],m[:,2],rand(Normal(0,1),hypers.nsites))

    # Get remaining simulation JSON data
    _,_,_,_,trueθ = readjson(jsonpath)

    # Generate simulated Y data
    Y,X₁,trueλ = simulation(hypers.nsites,hypers.ntimes, distm, trueθ,covars)

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
# testYmargins(Y,covars,trueθ,10,true)

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

@time chains,τs = ΓΓ_MCMC(mcmc1)
chains

# filenm = "Run1_2022-02-04T19-27-47-847.csv"
filenm = "Run3_2022-02-06T15-27-39-240.csv"
# filenm = "Run3_2022-02-06T18-47-59-678.csv"
savepath = get(sim_or_real,"save_path",0)
chains = Matrix{Float64}(CSV.read(string(savepath,filenm),DataFrame))

plotθλ(chains,reparameterize(trueθ),trueλ,[111,222,555,777,1000])

fittedtildeθ = parameter([mean(chains[:,i]) for i in 1:size(trueθ.α)[1]],
                    mean(chains[:,size(trueθ.α)[1]+1]),
                    mean(chains[:,size(trueθ.α)[1]+2]),
                    mean(chains[:,size(trueθ.α)[1]+3]))

fittedθ = deparameterize(fittedtildeθ)
trueθ

compareQQ(Y,covars,trueθ,fittedθ)

preddens(fittedθ,covars,hypers,[1,1],[0,300])

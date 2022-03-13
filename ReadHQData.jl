using DataFrames, CSV, Statistics, Plots, GLM

path = "C:\\Users\\lambe\\Documents\\McGill\\Masters\\Thesis\\"
obs = Matrix{Float64}(CSV.read(string(path,"ObservationsHQ.csv"),DataFrame,header=0))
prev_raw = Matrix{Float64}(CSV.read(string(path,"PrevisionsHQ.csv"),DataFrame,header=0)) 

prev = Array{Float64}(undef,10,90,2191,5)
for i in 1:5, j in 1:2191
    strt = (i-1)*10*2191+(j-1)*10+1
    finsh = (i-1)*10*2191+ j*10
    prev[:,:,j,i] = prev_raw[strt:finsh,:]
end

##########################################################
# Plot prevs against obs at site for a certain day ahead #
##########################################################
site = 5
days_ahead = 1
prevs = [Statistics.mean(prev[days_ahead,:,i,j][findall(!isnan,prev[days_ahead,:,i,j])]) for i in 1:2191,j in 1:5]

Plots.plot(obs[:,site],prevs[:,site],seriestype = :scatter)
Plots.abline!(1,0)

Plots.plot(prevs[:,site],obs[:,site]-prevs[:,site],seriestype = :scatter)

Plots.plot(prevs,obs,seriestype= :scatter)
display(Plots.plot(prevs,obs.-prevs,seriestype = :scatter))

# Filter residuals to set negatives to 0
res = obs.-prevs
for i in 1:size(res)[1], j in 1:size(res)[2] res[i,j] = ifelse(res[i,j]>0,res[i,j],0) end
res
sum_res = [sum(res[i,:]) for i in 1:2191]
Plots.plot(sum_res,seriestype= :scatter)

# Select only those times such that the spatial sum of residuals is more than undef
u = 5
ind_extr = findall(x-> x>u,sum_res)

obs_extr = obs[ind_extr,:]
prevs_extr = prevs[ind_extr,:]

res_extr = obs_extr .- prevs_extr
plot(prevs_extr,res_extr,seriestype = :scatter)

plot(res_extr)

############################
# Fixed threshold approach #
############################

obs
prevs_mean = [Statistics.mean(prev[days_ahead,:,i,j][findall(!isnan,prev[days_ahead,:,i,j])]) for i in 1:2191,j in 1:5]
prevs_std = [Statistics.std(prev[days_ahead,:,i,j][findall(!isnan,prev[days_ahead,:,i,j])]) for i in 1:2191,j in 1:5]

plot(obs)
[quantile(obs[:,i],0.9) for i in 1:size(obs)[2]]

u=10
ind_exc = sum([ifelse(obs[i,j]<u,0,1) for i in 1:size(obs)[1], j in 1:size(obs)[2]],dims=2)
ind_exc = [ifelse(size(findall(x->x>u,obs[i,:]),1)==0,0,1) for i in 1:size(obs)[1]]
exprt_obs = obs[findall(x->x==1,ind_exc),:]
dims = size(exprt_obs)
exprt_prevs_mean = reshape(transpose(prevs_mean[findall(x->x==1,ind_exc),:]),dims[1]*dims[2],1)
exprt_prevs_std = reshape(transpose(prevs_std[findall(x->x==1,ind_exc),:]),dims[1]*dims[2],1)
covars = hcat(exprt_prevs_mean,exprt_prevs_std)

# fpath = "C:\\Users\\lambe\\Documents\\McGill\\Masters\\Thesis\\Extr_obs_HQ.csv"
# CSV.write(fpath,DataFrame(exprt_obs,:auto))

# fpath = "C:\\Users\\lambe\\Documents\\McGill\\Masters\\Thesis\\Covars_HQ.csv"
# CSV.write(fpath,DataFrame(covars,:auto))

#######################
# Logistic Regression #
#######################

df = DataFrames.DataFrame(hcat(ind_exc,prevs_mean,prevs_std),:auto)

df[!,"Max_Prevs"] = [maximum(df[i,2:6]) for i in 1:size(df)[1]]
df[!,"Sum_Prevs"] = [sum(df[i,2:6]) for i in 1:size(df)[1]]
df[!,"Sum_Std"] = [sum(df[i,7:11]) for i in 1:size(df)[1]]
df


logit = glm(@formula(x1 ~ Max_Prevs+Sum_Prevs+Sum_Std),df,Binomial(),LogitLink())

vars = reshape([20.0,22.0,0.0],1,3)
test = DataFrames.DataFrame(vars,:auto)
rename!(test,[:Max_Prevs,:Sum_Prevs,:Sum_Std])
predict(logit,test)

sum(df[:,1])
size(df)

preds = []
for i in 1:size(df)[1]
    train = df[1:end .!= i,:]
    test = DataFrames.DataFrame(reshape([df[i,12:end][j] for j in 1:3],1,3),:auto)
    rename!(test,[:Max_Prevs,:Sum_Prevs,:Sum_Std])
    logit = glm(@formula(x1 ~ Max_Prevs+Sum_Prevs+Sum_Std),train,Binomial(),LogitLink())
    pred = predict(logit,test)
end

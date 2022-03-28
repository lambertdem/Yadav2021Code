using DataFrames, CSV, Statistics, Plots, GLM, MLBase, Distributions

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
[quantile(obs[:,i],0.7) for i in 1:size(obs)[2]]

u=2
# ind_exc = sum([ifelse(obs[i,j]<u,0,1) for i in 1:size(obs)[1], j in 1:size(obs)[2]],dims=2)
ind_exc = [ifelse(size(findall(x->x>u,obs[i,:]),1)==0,0,1) for i in 1:size(obs)[1]]
exprt_obs = obs[findall(x->x==1,ind_exc),:]
dims = size(exprt_obs)
exprt_prevs_mean = reshape(transpose(prevs_mean[findall(x->x==1,ind_exc),:]),dims[1]*dims[2],1)
exprt_prevs_std = reshape(transpose(prevs_std[findall(x->x==1,ind_exc),:]),dims[1]*dims[2],1)
covars = hcat(exprt_prevs_mean,exprt_prevs_std)

fpath = "C:\\Users\\lambe\\Documents\\McGill\\Masters\\Thesis\\Extr_obs_HQ_u2.csv"
CSV.write(fpath,DataFrame(exprt_obs,:auto))

fpath = "C:\\Users\\lambe\\Documents\\McGill\\Masters\\Thesis\\Covars_HQ_u2.csv"
CSV.write(fpath,DataFrame(covars,:auto))

prevs_mean1 = [Statistics.mean(prev[days_ahead,1:20,i,j][findall(!isnan,prev[days_ahead,1:20,i,j])]) for i in 1:2191,j in 1:5]
prevs_mean2 = [Statistics.mean(prev[days_ahead,21:40,i,j][findall(!isnan,prev[days_ahead,21:40,i,j])]) for i in 1:2191,j in 1:5]
prevs_mean3 = [Statistics.mean(prev[days_ahead,41:90,i,j][findall(!isnan,prev[days_ahead,41:90,i,j])]) for i in 1:2191,j in 1:5]

u=10
# ind_exc = sum([ifelse(obs[i,j]<u,0,1) for i in 1:size(obs)[1], j in 1:size(obs)[2]],dims=2)
ind_exc = [ifelse(size(findall(x->x>u,obs[i,:]),1)==0,0,1) for i in 1:size(obs)[1]]
exprt_obs = obs[findall(x->x==1,ind_exc),:]
dims = size(exprt_obs)
prevs_mean1 = reshape(transpose(prevs_mean1[findall(x->x==1,ind_exc),:]),dims[1]*dims[2],1)
prevs_mean2 = reshape(transpose(prevs_mean2[findall(x->x==1,ind_exc),:]),dims[1]*dims[2],1)
prevs_mean3 = reshape(transpose(prevs_mean3[findall(x->x==1,ind_exc),:]),dims[1]*dims[2],1)

for i in findall(isnan,prevs_mean2) prevs_mean2[i] = (prevs_mean1[i]+prevs_mean3[i])/2 end
covars = hcat(prevs_mean1,prevs_mean2,prevs_mean3)

fpath = "C:\\Users\\lambe\\Documents\\McGill\\Masters\\Thesis\\Covars_HQ3mods.csv"
CSV.write(fpath,DataFrame(covars,:auto))


#######################
# Logistic Regression #
#######################

df = DataFrames.DataFrame(hcat(ind_exc,prevs_mean,prevs_std),:auto)

df[!,"Max_Prevs"] = [maximum(df[i,2:6]) for i in 1:size(df)[1]]
df[!,"Sum_Prevs"] = [sum(df[i,2:6]) for i in 1:size(df)[1]]
df[!,"Sum_Std"] = [sum(df[i,7:11]) for i in 1:size(df)[1]]
df


logit = glm(@formula(x1 ~ Max_Prevs+Sum_Prevs+Sum_Std+Sum_Prevs*Sum_Std),df,Binomial(),LogitLink())
logit
vars = reshape([20.0,22.0,0.0],1,3)
test = DataFrames.DataFrame(vars,:auto)
rename!(test,[:Max_Prevs,:Sum_Prevs,:Sum_Std])
predict(logit,test)

preds = []
for i in 1:size(df)[1]
    println(i)
    train = df[1:end .!= i,:]
    test = DataFrames.DataFrame(reshape([df[i,12:end][j] for j in 1:3],1,3),:auto)
    rename!(test,[:Max_Prevs,:Sum_Prevs,:Sum_Std])
    logit = glm(@formula(x1 ~ Max_Prevs+Sum_Prevs+Sum_Std+Sum_Prevs*Sum_Std),train,Binomial(),LogitLink())
    pred = predict(logit,test)
    preds = vcat(preds,pred)
end

sorted_df = sort(DataFrame(hcat(preds,df[:,1]),:auto),:x1)

est_ps = []
for i in 1:10
    ind = findall(x-> x>0.0 +0.1*(i-1) && x<= 0.1*i,sorted_df[:,1])
    est_p = sum(sorted_df[ind,2])/size(sorted_df[ind,2])[1]
    est_ps = vcat(est_ps,est_p)
end

plot(preds,df[:,1],seriestype= :scatter)
plot!((1:10)/10,est_ps)
Plots.abline!(1,0)

plot(1:20,est_ps)
sum(sorted_df[:,2])

plot(sorted_df[:,1])

bin_pred = [ifelse(x<0.5,0,1) for x in preds]
confsn_mtrx = MLBase.roc(convert.(Int64,df[:,1]),bin_pred)

precision = confsn_mtrx.tp/(confsn_mtrx.tp+confsn_mtrx.fp)
recall = confsn_mtrx.tp/(confsn_mtrx.tp+confsn_mtrx.fn)


logit0 = glm(@formula(x1 ~ Max_Prevs+Sum_Prevs),df,Binomial(),LogitLink())
logit1 = glm(@formula(x1 ~ Max_Prevs+Sum_Prevs+Sum_Std),df,Binomial(),LogitLink())
logit2 = glm(@formula(x1 ~ Max_Prevs+Sum_Prevs+Sum_Std+Sum_Prevs*Sum_Std),df,Binomial(),LogitLink())
lik_ratio_test = deviance(logit1) - deviance(logit2)

1-Distributions.cdf(Distributions.Chisq(1),lik_ratio_test)

plot(df[:,"Sum_Std"],df[:,1],seriestype=:scatter)
plot(df[:,"Sum_Std"].*df[:,"Sum_Prevs"],df[:,1],seriestype=:scatter)

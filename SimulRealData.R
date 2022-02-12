install.packages('SpatialExtremes')
library(SpatialExtremes)
set.seed(4)
alt <- rep(rgamma(20,1.5,0.01),50)
mod1 <- rgamma(1000,0.6,0.2)
plot(mod1)
mod2 <- rnorm(1000,mod1,3)
mod2 <- ifelse(mod2<0,0,mod2)
plot(mod1,mod2)

err <- rgpd(1000,0,1,0.2)
plot(err)

Y <- (mod1+mod2)/2 + err + 0.01*alt
plot(mod1,Y)
abline(0,1,col="red")
plot(mod2,Y)
abline(0,1,col="red")

Y <- matrix(Y,ncol=20)
plot(c(Y)-mod1)

range <- abs(mod1-mod2)
covars <- cbind(mod1,mod2,alt,range)
colnames(covars) <- c("Pred1","Pred2","Alt","Range")

u <- matrix((mod1+mod2)/2,ncol=20) +2
plot(Y,u)

abline(0,1,col="red")

path = "C:/Users/lambe/Documents/McGill/Masters/Thesis/Yadav2021Code/Runs/"
write.csv(covars,paste0(path,"covars.csv"))

write.csv(Y,paste0(path,"Y.csv"))

write.csv(u,paste0(path,"u.csv"))

cbind(Y[1:20],covars[1:20,1:2])

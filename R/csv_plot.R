args = commandArgs(trailingOnly=TRUE)
if(length(args)==0){
  stop("csv_plot.R [csv file name]");
}
d = read.csv(file=args[1], header= TRUE)
fields <- colnames(d)

png(paste(args[1], "_plot.png", sep=''))
dt <- (log(d + exp(1)) - 1)
l <- lm(dt)
print(l)
r_sq <-summary(l)$r.squared
print(r_sq)
plot(dt[,1] ~ d[,2], xlab=fields[2], ylab=fields[1]) 
print(l)
print(l[0])
print(l[1])
abline(l, col="red")
legend("topright", bty="n", legend=paste("r^2 =", format(r_sq, digits=4)), col="red")
dev.off()
write.csv(r_sq, paste(args[1], "_rsq.txt", sep=''))

png(paste(args[1], "_logplot.png", sep=''))
dt <- (log(d + exp(1)) - 1)
l <- lm(dt)
print(l)
r_sq <-summary(l)$r.squared
print(r_sq)
plot(dt[,1] ~ dt[,2], xlab=fields[2], ylab=fields[1])
print(l)
print(l[0])
print(l[1])
abline(l, col="red")
legend("topright", bty="n", legend=paste("r^2 =", format(r_sq, digits=4)), col="red")
dev.off()
write.csv(r_sq, paste(args[1], "_rsq.txt", sep=''))

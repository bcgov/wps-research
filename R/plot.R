# plot.R
d<-read.csv("tmp.csv")
plot(d$knn_use, log(d$n_seg), type="l")

plot(log(d$n_seg), d$knn_use, type="l")


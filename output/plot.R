# plot.R
d<-read.csv("tmp.csv")
plot(d$knn_use, log(d$n_seg), type="l")

# 20221112 plotting Triangular distribution in R with slider! No R-Studio or webserver req'd
library(tcltk)
library(tkrplot)
library(extraDistr)  # https://search.r-project.org/CRAN/refmans/extraDistr/html/Triangular.html

#==PARAMETERS==================================================================
MIN = 0.  # a = minimum possible value for Random Variable
MAX = 1.  # b = maximum possible value for Random Variable
N = 200. # how many increments to plot!
#==============================================================================

tt = tktoplevel()
C = tclVar(.5)  # initial value for parameter (c)
x <- seq(MIN, MAX, by=(1./N))  # values to plot at

# https://bookdown.org/content/4e34e34f-ca48-4090-90ca-8ae7b1b65e0e/plotting-with-r-base-code.html
plotf = function(...){
   plot(x,
	dtriang(x, MIN, MAX, as.numeric(tclvalue(C)), log = FALSE),
	type="l",
	xlab="x",  # potential value for random variable
	ylab="f(x)",  # probability density value for each value of x
   	main="Demo: plotting triangular distribution (PDF) with slider") 
}

img = tkrplot(tt, plotf)  # tkrplot widget using the above function
densplot = function(...)tkrreplot(img)  # trivial updating function that calls tkrreplot
scl = tkscale(tt,  # create slider (call updating function when value changes)
	      command=densplot,
	      from=MIN,
	      to=MAX,
	      showvalue=TRUE,
              variable=C,
	      resolution=(1./N),
	      orient='horiz')
tkpack(img, side='top')  # pack the widgets
tkpack(scl, side='top')

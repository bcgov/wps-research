# 20230516 ran this at terminal with:
# 	Rscript view.R
# 
# Pattern recognition / stats should probably use the data as-is..
#   ..or perhaps with the simple linear scaling to [0, 1].
# 
# I.e. Pattern recognition shouldn't need the histogram trimming /
#   scaling step... That's just to help see that the fire pattern
#   can / should be segmentable
# 
# Note: Band/channel order got reversed to make the fire look orange.
#   Without reversing the order the fire looks blue. Cool but harder
#   to see!


# install.packages('raster')
library(raster) 
defaultW <- getOption("warn")
options(warn = -1)  # too many warnings for no reason

write_png = function(in_file, data, title_s){
    png(paste(in_file, "_", title_s, ".png", sep=""), res=1200)
    plotRGB(data)
    dev.off()
}

plot_tiff = function(in_file){    
    data = brick(in_file)  # why seperate fxn to read 3 channels, vs. 1? 
    print(data[[1]])
    
    min_values = cellStats(data, stat='min', na.rm=TRUE)
    max_values = cellStats(data, stat='max', na.rm=TRUE)
    
    # linear scaling to [0, 1]
    b1_t <- (data[[1]] - min_values[[1]]) / (max_values[[1]] - min_values[[1]])
    b2_t <- (data[[2]] - min_values[[2]]) / (max_values[[2]] - min_values[[2]])
    b3_t <- (data[[3]] - min_values[[3]]) / (max_values[[3]] - min_values[[3]])
	
    data[[3]] <- 255. * b1_t  # switched from 1,2,3 to 3,2,1 to make fire orange!
    data[[2]] <- 255. * b2_t
    data[[1]] <- 255. * b3_t
		write_png(in_file, data, "scaled")  # plot to PNG

    b1_s = sort(as.vector(b1_t), decreasing=FALSE)
    b2_s = sort(as.vector(b2_t), decreasing=FALSE)
    b3_s = sort(as.vector(b3_t), decreasing=FALSE)
    p = as.integer((1.5 / 100.) * length(b1_s)) # 1.5 percentile
    
		# find values for bottom and top percentiles
    max_values[[1]] = b1_s[length(b1_s) - p]
    max_values[[2]] = b2_s[length(b2_s) - p]
    max_values[[3]] = b3_s[length(b3_s) - p]
    min_values[[1]] = b1_s[p]
    min_values[[2]] = b2_s[p]
    min_values[[3]] = b3_s[p]
    
		# histogram scaling
    b1_t <- (b1_t - min_values[[1]]) / (max_values[[1]] - min_values[[1]])
    b2_t <- (b2_t - min_values[[2]]) / (max_values[[2]] - min_values[[2]])
    b3_t <- (b3_t - min_values[[3]]) / (max_values[[3]] - min_values[[3]])
    
    # clip values outside of [0,1]
    b1_t[b1_t < 0.] <- 0.
    b2_t[b2_t < 0.] <- 0.
    b3_t[b3_t < 0.] <- 0.
    
    b1_t[b1_t > 1.] <- 1.
    b2_t[b2_t > 1.] <- 1.
    b3_t[b3_t > 1.] <- 1.
    
		# scale to [0, 255] so that plotRGB is happy!
    data[[3]] <- 255. * b1_t  # switched 1,2,3 to 3,2,1 to make fire orange!
    data[[2]] <- 255. * b2_t
    data[[1]] <- 255. * b3_t
		write_png(in_file, data, "trimmed")  # plot to PNG
}

plot_tiff("G80223_20230513.tif")
plot_tiff("G90292_20230514.tif")

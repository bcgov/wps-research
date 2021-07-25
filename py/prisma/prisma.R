library(prismaread)
# need to declare in_file as an .he5 file

pr_convert(in_file = infile,
           out_folder = "out", 
           out_format = "ENVI",
           VNIR=TRUE,
           SWIR=TRUE,
           LATLON=FALSE, #TRUE,
           PAN=FALSE,# TRUE,
           CLOUD=FALSE) # TRUE)

gdal_translate 5MCP13_20210715.tif -of ENVI -ot Float32 5MCP13_20210715.bin & 
gdal_translate 5MCP13_20210719.tif -of ENVI -ot Float32 5MCP13_20210719.bin &
gdal_translate 5MCP18_20210718.tif -of ENVI -ot Float32 5MCP18_20210718.bin &
gdal_translate 5MCP19_20210710.tif -of ENVI -ot Float32 5MCP19_20210710.bin &
gdal_translate 5MCP19_20210714.tif -of ENVI -ot Float32 5MCP19_20210714.bin &
gdal_translate 5MCP19_20210722.tif -of ENVI -ot Float32 5MCP19_20210722.bin &
gdal_translate  5MCP7_20210708.tif -of ENVI -ot Float32 5MCP7_20210708.bin &
gdal_translate  5MCP7_20210716.tif -of ENVI -ot Float32 5MCP7_20210716.bin &
gdal_translate  5MCP7_20210720.tif -of ENVI -ot Float32 5MCP7_20210720.bin &
wait

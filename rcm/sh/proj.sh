# the other script rewrite was to get rid of the error: 
# from gdal: # AttributeError: 'NoneType' object has no attribute 'GetProjection'
# now we can project:
project_onto 5MCP13_20210715.tif ../projected/5MCP7_20210708.tif ../projected/5MCP13_20210715.tif &
project_onto 5MCP13_20210719.tif ../projected/5MCP7_20210708.tif ../projected/5MCP13_20210719.tif &
project_onto 5MCP18_20210718.tif ../projected/5MCP7_20210708.tif ../projected/5MCP18_20210718.tif & 
project_onto 5MCP19_20210710.tif ../projected/5MCP7_20210708.tif ../projected/5MCP19_20210710.tif &
project_onto 5MCP19_20210714.tif ../projected/5MCP7_20210708.tif ../projected/5MCP19_20210714.tif &
project_onto 5MCP19_20210722.tif ../projected/5MCP7_20210708.tif ../projected/5MCP19_20210722.tif &
project_onto 5MCP7_20210716.tif  ../projected/5MCP7_20210708.tif ../projected/5MCP7_20210716.tif &
project_onto 5MCP7_20210720.tif  ../projected/5MCP7_20210708.tif ../projected/5MCP7_20210720.tif &
wait

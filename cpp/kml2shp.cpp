/* 20220824 kml to shape:
ogr2ogr -f 'ESRI Shapefile' output.shp input.kml

shape to kml: 
ogr2ogr -f KML output.kml input.shp */
#include"misc.h"
int main(int argc, char ** argv){
  if(argc < 2){
	  err("kml2shp [input .kml filename]");
  }
  str fn(argv[1]);
  str ofn(fn + str(".shp"));

  run(str("ogr2ogr -f 'ESRI Shapefile' ") + ofn + str(" ") + fn);
  return 0;
}

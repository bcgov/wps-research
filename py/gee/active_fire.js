/*20220510 active_fire.js*/

/* PARAMETERS ==============================================================*/
var t1 = ee.Date('2021-8-02T00:00', 'Etc/GMT-8');  //var t1 = ee.Date('2019-7-02T00:00', 'Etc/GMT-8');
var N_DAYS = 14  // width of date-range for data search
var CLOUD_THRES = 22.2 // % cloud cover max
/* END PARAMETERS ==========================================================*/

// function from online example:
function maskS2clouds(image) {
  var qa = image.select('QA60'); // Bits 10 and 11 are clouds and cirrus, respectively.
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;   // Both flags should be set to zero, indicating clear conditions.
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0).and(qa.bitwiseAnd(cirrusBitMask).eq(0));
  return image.updateMask(mask).divide(10000);
}

/* get DEM, LandCover, Sentinel-2 "L2A" (level two atmospherically corrected)
 * data */
var nasa_dem = ee.Image('NASA/NASADEM_HGT/001').select('elevation');  // dem data
var land_cover = ee.ImageCollection("ESA/WorldCover/v100").first(); // values of this layer shown below
var dataset = ee.ImageCollection('COPERNICUS/S2_SR').filterDate(t1, t1.advance(N_DAYS, 'day'))
var dates = dataset.map(function(image){ return ee.Feature(null, {'date': image.date().format('YYYY-MM-dd')})}).distinct('date').aggregate_array('date')
print(dates)

/* apply cloud threshold and mask */
dataset = dataset.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',CLOUD_THRES)).map(maskS2clouds).mean();
var visualization = {min: 0.0, max: 1.0, bands: ['B12', 'B11', 'B9']}; // rgb vis in which fires are "red"

/* apply classification rule */
var r = dataset.expression(
    'R > G && R > B && (LC != 80) && (LC != 50) && (LC != 70) && (DEM < 1500)', {
      'R': dataset.select('B12'),
      'G': dataset.select('B11'),
      'B': dataset.select('B9'),
      'LC': land_cover.select('Map'),
      'DEM': nasa_dem
    });
Map.addLayer(r, {min: 0, max: 1})  // the binary classification
Map.addLayer(dataset, visualization, 'RGB'); // the Short-wave infrared vis that inspired it
Map.setCenter(-119.97, 50.34)

/* Values of ESA/WorldCover/v100:
  
  Value Color   Description
  10      006400        Trees
  20      ffbb22        Shrubland
  30      ffff4c        Grassland
  40      f096ff        Cropland
  50      fa0000        Built-up
  60      b4b4b4        Barren / sparse vegetation
  70      f0f0f0        Snow and ice
  80      0064c8        Open water
  90      0096a0        Herbaceous wetland
  95      00cf75        Mangroves
  100   fae6a0  Moss and lichen*/

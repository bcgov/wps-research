'''Set Band descriptions from https://gis.stackexchange.com/questions/290796/how-to-edit-the-metadata-for-individual-bands-of-a-multiband-raster-preferably
Usage:
    python set_band_desc.py /path/to/file.ext band1 desc1 [band2 desc2] .. [band-n desc-n]
Where:
    band = band number to set (starting from 1)
    desc = band description string (enclose in "double quotes" if it contains spaces)
Example:
    python set_band_desc.py /path/to/dem.tif 1 "Band 1 desc"  2 "Band 2 desc"  3 "Band 3 desc"
'''
import sys
from osgeo import gdal

def set_band_descriptions(filepath, bands):
    """
    filepath: path/virtual path/uri to raster
    bands:    ((band, description), (band, description),...)
    """
    ds = gdal.Open(filepath, gdal.GA_Update)
    for band, desc in bands:
        print("Setting band description of band: " + str(band) + " to: " + desc)
        rb = ds.GetRasterBand(band)
        rb.SetDescription(desc)
    del ds

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('''Usage:
    python set_band_desc.py /path/to/file.ext band desc [band desc...]
Where:
    band = band number to set (starting from 1)
    desc = band description string (enclose in "double quotes" if it contains spaces)
Example:
    python set_band_desc.py /path/to/dem.tif 1 "Band 1 desc"  2 "Band 2 desc"  3 "Band 3 desc"''')
        sys.exit(1)
    filepath = sys.argv[1]
    bands = [int(i) for i in sys.argv[2::2]]
    names = sys.argv[3::2]
    set_band_descriptions(filepath, zip(bands, names))



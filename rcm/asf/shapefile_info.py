# min size in hectares
MIN_SIZE = 3000
MIN_YEAR = 2006
MAX_YEAR = 2011
PROV = 'BC'

import os
import sys
import json
import urllib.parse
from osgeo import gdal # need gdal / python installed!
from osgeo import ogr
from osgeo import gdalconst
from pyproj import CRS
from pyproj import Transformer

if not os.path.exists('data'):
    os.mkdir('data')

crs = CRS.from_wkt('PROJCS["NAD_1983_Lambert_Conformal_Conic",GEOGCS["GCS_North_American_1983",DATUM["D_North_American_1983",SPHEROID["GRS_1980",6378137.0,298.257222101]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Lambert_Conformal_Conic"],PARAMETER["False_Easting",0.0],PARAMETER["False_Northing",0.0],PARAMETER["Central_Meridian",-95.0],PARAMETER["Standard_Parallel_1",49.0],PARAMETER["Standard_Parallel_2",77.0],PARAMETER["Latitude_Of_Origin",49.0],UNIT["Meter",1.0]]')
proj = Transformer.from_crs(crs, crs.geodetic_crs)

print(crs)

def err(m):
    print("Error: " + m); sys.exit(1)
args = sys.argv  # arguments
if len(args) < 2: err("Error: shapefile_info.py [input shapefile .shp]")
InputVector = args[1] # shapefile to rasterize

Shapefile = ogr.Open(InputVector) # open shp
layer = Shapefile.GetLayer()
layerDefinition = layer.GetLayerDefn()
feature_count = layer.GetFeatureCount()

def records(layer):
    # generator
    for i in range(layer.GetFeatureCount()):
        feature = layer.GetFeature(i)
        yield json.loads(feature.ExportToJson())

print("feature_count," + str(feature_count))
print("feature","type", "x","y")
features = records(layer)
feature_names, feature_ids = [], []
for f in features:  
    geom = ""
    #print("keys", f.keys())
    #for k in f.keys():
    #    print('  ', k, ':', f[k])
    try:
        geom = f['geometry']
    except Exception:
        pass

    year = int(f['properties']['YEAR'])
    prov = f['properties']['SRC_AGENCY'].strip()
    
    if year <= MAX_YEAR and year >= MIN_YEAR:
        if prov == PROV:
            ix = f['properties']['FIRE_ID']
            sz = f['properties']['SIZE_HA']
            sz = int(sz)
            if sz > MIN_SIZE:
                # print(ix, year, sz) #             print(f['properties'])
                # print(geom.keys())
                '''feature_id = f['id']
                feature_ids.append(feature_id) # print(f['properties'].keys())
                feature_name = ''
                try:
                    feature_name = f['properties']['Name']
                except Exception:
                    pass # feature name not available
                feature_names.append(feature_name)
                '''
                if geom['type'] == 'Point':
                    stuff = [feature_id,
                             geom['type'],
                             geom['coordinates'][0],
                             geom['coordinates'][1]]
                    print(','.join([str(x) for x in stuff]))
                else: # assume polygon or multipolygon
                    pts = geom['coordinates']
                    if geom['type'] == 'MultiPolygon':
                        p = []
                        for x in pts:
                            p += x
                        pts = p

                    p = []
                    for x in pts:
                        p += x
                    pts = p

                    pts = [[p[0], p[1]] for p in pts]
                    # print(geom['type'], pts)

                    bbx_min, bby_min = p[0][0], p[0][1]
                    bbx_max, bby_max = p[0][0], p[0][1]
                    for p in pts:
                        if p[0] < bbx_min: bbx_min = p[0]
                        if p[0] > bbx_max: bbx_max = p[0]
                        if p[1] < bby_min: bby_min = p[1]
                        if p[1] > bby_max: bby_max = p[1]

                    # convert to lat, lon
                    [bbx_min, bby_min] = proj.transform(bbx_min, bby_min)
                    [bbx_max, bby_max] = proj.transform(bbx_max, bby_max)

                    [bbx_min, bbx_max, bby_min, bby_max] = [round(x, 4) for x in
                                                            [bbx_min, bbx_max, bby_min, bby_max]]
                    '''
                    import matplotlib.pyplot as plt
                    plt.plot([p[0] for p in pts], [p[1] for p in pts], 'o', label='data')
                    plt.plot([bbx_min, bbx_max, bbx_max, bbx_min, bbx_min],
                              [bby_min, bby_min, bby_max, bby_max, bby_min], label='bbx')
                    plt.legend()
                    plt.show()
                   
                    print(geom['type'])
                    for c in pts: # geom['coordinates']:
                        for d in c:
                            print("  " + str(d))
                    '''
                    # intersectsWith=polygon((-119.543 37.925, -118.443 37.7421, -118.682 36.8525, -119.77 37.0352, -119.543 37.925 ))
                    c = 'intersectsWith=POLYGON(('
                    c += (' '.join([str(bbx_min), str(bby_min) + ',',
                                    str(bbx_max), str(bby_min) + ',',
                                    str(bbx_max), str(bby_max) + ',',
                                    str(bbx_min), str(bby_max) + ',', 
                                    str(bbx_min), str(bby_min)]))
                    c += '))'
                    c = c.replace(', ', ',')
                    print(ix, year, sz, c)

                    c = c.replace(' ', '%20')
                    c = c.replace('(', '%28')
                    c = c.replace(')', '%29')
                    url = 'https://api.daac.asf.alaska.edu/services/search/param?' #keyword1=value1&keyword2=value2,value3&keyword3=value4-6'
                    url += 'platform=ALOS'
                    url += '&instrument=PALSAR'
                    url += '&polarization=QUADRATURE,FULL'
                    url += ('&' + c) 
                    # urllib.parse.quote_plus(c))
                    print(url)

                    import urllib. request #pip install concat("urllib", number of current versio
                    my_request = urllib.request.urlopen(url)
                    my_HTML = my_request.read().decode("utf8")
                    print(my_HTML)
                    sys.exit(1)
            # sys.exit(1)

'''

Search API Keywords

Keywords are used to find the desired data. Use as many or as few keywords as needed. Available keywords and descriptions are listed below for each Search API endpoint. Keywords are case sensitive.

Note: Any errors will be returned in JSON format.
Search Endpoint

https://api.daac.asf.alaska.edu/services/search/param
Dataset Parameters

    platform
        See also 'instrument'
        Remote sensing platform that acquired the data. Sentinel-1 and ERS have multiple remote sensing platforms, and you may choose whether to specify a specific platform. You may specify a single value, or a list of values.
        Example:
            platform=ALOS
            platform=SA,SB
            platform=S1
        Values:
            ALOS, A3, AIRSAR, AS, ERS, ERS-1, E1, ERS-2, E2, JERS-1, J1, RADARSAT-1, R1, SEASAT, SS, S1, Sentinel, Sentinel-1, Sentinel-1A, SA, Sentinel-1B, Sentinel-1 Interferogram (BETA), SB, SIR-C, SMAP, SP, UAVSAR, UA

    instrument
        See also 'platform'
        Remote sensing instrument that acquired the data. For some platforms, such as ALOS, there are multiple instruments to choose from.
        Example:
            ALOS: instrument=PALSAR
            ALOS: instrument=AVNIR-2
        Values:
            C-SAR, PALSAR, AVNIR-2

    absoluteOrbit
        For ALOS, ERS-1, ERS-2, JERS-1, RADARSAT-1, Sentinel-1A, and Sentinel-1B this value corresponds to the orbit count within the orbit cycle. For UAVSAR it is the Flight ID. You may specify a single value, range of values, or a list of values.
        Example:
            RADARSAT: absoluteOrbit=25436
            PALSAR: absoluteOrbit=25436-25445,25450
            UAVSAR: absoluteOrbit=12006

    asfframe
        See also 'frame'
        This is primarily an ASF / JAXA frame reference. However, some platforms use other conventions. You may specify a single value, range of values, or a list of values.
        Example:
            asfframe=300 or asfframe=2845-2855 or asfframe=2800,2845-2855
        Values:
            ERS, JERS, RADARSAT: ASF frames 0 to 900
            ALOS PALSAR: JAXA frames 0 to 7200
            SEASAT: ESA-like frames 0208 to 3458 (must use a leading zero for frames 208-999)
            Sentinel-1: In-house values 0 to 1184

    maxBaselinePerp
        For interferometric SAR (InSAR) analysis, Perpendicular Baseline is the spatial distance between the first and second observations measured perpendicular to the satellite look direction and provides an indication of the sensitivity to topographic height.
        Works for ERS-1, ERS-2, JERS, RADARSAT-1, ALOS PALSAR. (Not Sentinel-1)
        Example:
            maxBaselinePerp=1500 or maxBaselinePerp=50.5

    minBaselinePerp
        For interferometric SAR (InSAR) analysis, Perpendicular Baseline is the spatial distance between the first and second observations measured perpendicular to the satellite look direction and provides an indication of the sensitivity to topographic height.
        Works for ERS-1, ERS-2, JERS, RADARSAT-1, ALOS PALSAR. (Not Sentinel-1)
        Example:
            minBaselinePerp=100 or minBaselinePerp=50.5

    beamMode
        The beam mode used to acquire the data. See also beamSwath. You may specify a single value, or a list of values.
        Example:
            beamMode=FBS or beamMode=EW,IW or beamMode=ScanSAR+Wide
        Values:
            AIRSAR: 3FP, ATI, XTI
            ALOS: FBD, FBS, PLR, WB1, WB2, DSN
            ERS-1: Standard, STD
            ERS-2: Standard, STD
            JERS-1: Standard, STD
            RADARSAT-1: Standard, STD, Fine, High, Low, Wide, Narrow, ScanSAR+Wide, ScanSAR+Narrow
            SEASAT: Standard, STD
            SMAP: Standard, STD
            Sentinel-1A: EW, IW, S1, S2, S3, S4, S5, S6, WV
            Sentinel-1B: EW, IW, S1, S2, S3, S4, S5, S6, WV
            UAVSAR: POL, RPI

    beamSwath
        BeamSwath encompasses a look angle and beam mode. You may specify a single value, or a list of values.
        Example:
            beamSwath=0
            beamSwath=FN1, FN2, FN3, FN4, FN5
        Values:
            AIRSAR: 3FP, ATI, XTI
            ALOS: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19, 20
            ERS-1: STD
            ERS-2: STD
            JERS-1: STD
            RADARSAT-1: FN1, FN2, FN3, FN4, FN5, SNA, SNB, ST1, ST2, ST3, ST4, ST5, ST6, ST7, SWA, SWB, WD1, WD2, WD3, EH3, EH4, EH6, EL1
            SEASAT: STD
            Sentinel-1A: EW, IW, S1, S2, S3, S4, S5, S6, WV
            Sentinel-1B: EW, IW, S1, S2, S3, S4, S5, S6, WV
            UAVSAR: POL, RPI

    collectionName
        For UAVSAR and AIRSAR data collections only. Search by the mission/campaign name. You may specify a single value. For a list of available collections, refer to the Mission List Endpoint below.
        Example:
            UAVSAR: collectionName=ABoVE
            AIRSAR: collectionName=collectionName=Akiyoshi,+Japan

    maxDoppler
        Doppler provides an indication of how much the look direction deviates from the ideal perpendicular flight direction acquisition.
        Example:
            maxDoppler=1500 or maxDoppler=1500.5

    minDoppler
        Doppler provides an indication of how much the look direction deviates from the ideal perpendicular flight direction acquisition.
        Example:
            minDoppler=100 or minDoppler=1500.5

    maxFaradayRotation
        Rotation of the polarization plane of the radar signal impacts imagery. HH and HV signals become mixed. One-way rotations exceeding 5° are likely to significantly reduce the accuracy of geophysical parameter recovery, such as forest biomass.
        Example:
            maxFaradayRotation=3.5

    minFaradayRotation
        Rotation of the polarization plane of the radar signal impacts imagery. HH and HV signals become mixed. One-way rotations exceeding 5° are likely to significantly reduce the accuracy of geophysical parameter recovery, such as forest biomass.
        Example:
            minFaradayRotation=2

    flightDirection
        Satellite orbit direction during data acquisition. You may specify a single value.
        Example:
            flightDirection=DESCENDING
        Values:
            A, ASC, ASCENDING, D, DESC, DESCENDING

    flightLine
        Specify a flightline for UAVSAR or AIRSAR. You may specify a single value.
        Example:
            UAVSAR: flightLine=05901
            AIRSAR: flightLine=gilmorecreek045-1.93044

    frame
        See also 'asfframe'
        ESA-referenced frames are offered to give users a universal framing convention. Each ESA frame has a corresponding ASF frame assigned. You may specify a single value, range of values, or a list of values.
        Example:
            frame=300
            frame=300-400
            frame=300,303,305
            frame=300,303,305-315
        Values:
            Any number from 0 to 7200.

    granule_list
        Comma-separated list of specific scenes (granules). Large lists will need to utilize a POST request.
        Note: specifying a granule list will supersede most other keywords.
        Example:
            granule_list=ALPSRP111041130, S1B_IW_GRDH_1SDV_20161124T032008_20161124T032033_003095_005430_9906

    groupid
        Comma-separated list of specific group IDs. For some datasets, the group ID is the same as the scene name. For others, such as Sentinel-1, the group ID is unique for a group of scenes. The group ID value is included in GeoJSON, JSON, and CSV outputs.
        Example:
            groupid=S1A_IWDV_0112_0118_037147_150

    lookDirection
        Left or right direction of data acquisition. You may specify a single value.
        Example:
            lookDirection=L
        Values:
            R, RIGHT, L, LEFT

    maxInsarStackSize
        An InSAR stack is composed of all SAR granules that cover the same geographic region, are from the same platform, and were acquired with the same beam mode, look angle, and bandwidth. To obtain InSAR stacks containing a certain number of SAR granules specify a min, max, or both.
        Works for ERS-1, ERS-2, JERS, RADARSAT-1, ALOS PALSAR. (Not Sentinel-1)
        Example:
            maxInsarStackSize=175

    minInsarStackSize
        An InSAR stack is composed of all SAR granules that cover the same geographic region, are from the same platform, and were acquired with the same beam mode, look angle, and bandwidth. To obtain InSAR stacks containing a certain number of SAR granules specify a min, max, or both.
        Works for ERS-1, ERS-2, JERS, RADARSAT-1, ALOS PALSAR. (Not Sentinel-1)
        Example:
            minInsarStackSize=20

    offNadirAngle
        Off-nadir angles for ALOS PALSAR. You may specify a single value, range of values, or a list of values.
        Example:
            offNadirAngle=21.5
            offNadirAngle=9.7-14
            offNadirAngle=21.5,23.1,20.5-24.2
        Values:
            Most common: 21.5, 23.1, 27.1, 34.3
            Other: 9.7, 9.9, 13.8, 14, 16.2, 17.3, 17.9, 18, 19.2, 20.5, 21.5, 23.1, 24.2, 24.6, 25.2, 25.8, 25.9, 26.2, 27.1, 28.8, 30.8, 34.3, 36.9, 38.8, 41.5, 43.4, 45.2, 46.6, 47.8, 49, 50, 50.8

    polarization
        A property of SAR electromagnetic waves that can be used to extract meaningful information about surface properties of the earth. You may specify a single value, or a list of values.
        Example:
            polarization=VV
            polarization=VV,HH
            polarization=VV+VH
            polarization=Dual+VV
        Values:
            AIRSAR: FULL
            ALOS: QUADRATURE, HH+5SCAN, HH, HH+4SCAN, VV, HH+3SCAN, FULL, HH+HV, VV+VH
            ERS-1: VV
            ERS-2: VV
            JERS-1: HH
            RADARSAT-1: HH
            SEASAT: HH
            Sentinel-1A: VV, VV+VH, Dual VV, VV+VH, Dual HV, HH, HH+HV, VV, Dual VH
            Sentinel-1B: VV, VV+VH, Dual VV, VV+VH, Dual HV, HH, HH+HV, VV, Dual VH
            UAVSAR: FULL, HH

    processingLevel
        Level to which the data has been processed, also type of product, such as browse. You may specify a single value, or a list of values.
        Example:
            processingLevel=L0,L1
        Values:
            AIRSAR: 3FP, ATI, LTIF, PTIF, CTIF, PSTOKES, BROWSE, DEM, CSTOKES, JPG, LSTOKES, THUMBNAIL
            ALOS: L1.0, L1.1, L1.5, RTC_LOW_RES, RTC_HI_RES, BROWSE, THUMBNAIL, METADATA, INTERFEROMETRY
            ERS-1: L0, L1, BROWSE, THUMBNAIL
            ERS-2: L0, L1, BROWSE, THUMBNAIL
            JERS-1: L0, L1, BROWSE, THUMBNAIL
            RADARSAT-1: L0, L1, BROWSE, THUMBNAIL
            SEASAT: L1, BROWSE, THUMBNAIL
            Sentinel-1A: METADATA_GRD, GRD_HS, GRD_HD, GRD_MS, GRD_MD, GRD_FS, GRD_FD, SLC, RAW, OCN, METADATA_RAW, METADATA, METADATA_SLC, THUMBNAIL
            Sentinel-1B: METADATA_GRD, GRD_HS, GRD_HD, GRD_MS, GRD_MD, GRD_FS, GRD_FD, SLC, RAW, OCN, METADATA_RAW, METADATA, METADATA_SLC, THUMBNAIL
            SMAP: L1A_Radar_RO_QA, L1B_S0_LoRes_HDF5, L1B_S0_LoRes_QA, L1B_S0_LoRes_ISO_XML, L1A_Radar_QA, L1A_Radar_RO_ISO_XML, L1C_S0_HiRes_ISO_XML, L1C_S0_HiRes_QA, L1C_S0_HiRes_HDF5, L1A_Radar_HDF5
            UAVSAR: KMZ, PROJECTED, PAULI, PROJECTED_ML5X5, STOKES, AMPLITUDE, BROWSE, COMPLEX, DEM_TIFF, PROJECTED_ML3X3, METADATA, AMPLITUDE_GRD, INTERFEROMETRY, INTERFEROMETRY_GRD, THUMBNAIL

    product_list
        Comma-separated list of specific files (products). Large lists will need to utilize a POST request. You can find the product_list values for any file in the GeoJSON (fileID) or JSON (product_file_id) outputs. It is also available from CMR, in the granuleUR field. It is guaranteed to be a unique indentifier in CMR. You can also find the product_list value in Vertex! See the Cookbook page for this Tip & more.
        Example:
            product_list=ALAV2A276512920, S1A_IW_SLC__1SDV_20210614T154839_20210614T154905_038338_048643_D7E4-SLC

    relativeOrbit
        Path or track of satellite during data acquisition. For UAVSAR it is the Line ID. You may specify a single value, range of values, or a list of values.
        Example:
            relativeOrbit=500,550-580
            UAVSAR: relativeOrbit=05905
        Values:
            ALOS: 1-671
            ERS-1: 0-2410
            ERS-2: 0-500
            JERS-1: 0-658
            RADARSAT-1: 0-342
            SEASAT: 1-243
            UAVSAR: various

Geospatial Parameters

    bbox
        Deprecation Notice: This keyword will be deprecated. Please use 'intersectsWith' instead.
        Bounding boxes define an area using two long/lat points. The Bounding box parameters are 4 comma-separated numbers: lower left longitude,latitude, and upper right longitude,latitude. This is a great choice for very wide search areas.
        Example:
            bbox=-150.2,65.0,-150.1,65.5

    intersectsWith
        Search by polygon, a line segment (“linestring”), or a point defined in 2-D Well-Known Text (WKT). Each polygon must be explicitly closed, i.e. the first vertex and the last vertex of each listed polygon must be identical. Coordinate pairs for each vertex are in decimal degrees: longitude is followed by latitude.
        Notes:
            Does not support multi-polygon, multi-line or multi-point.
            Polygon holes are ignored
            This keyword also accepts a POST request
        Example (Note: The spaces and parentheses below need to be URL encoded first):
            intersectsWith=polygon((-119.543 37.925, -118.443 37.7421, -118.682 36.8525, -119.77 37.0352, -119.543 37.925 ))
            intersectsWith=linestring(-119.543 37.925, -118.443 37.7421)
            intersectsWith=point(-119.543, 37.925)
        Properly URL encoded:
            intersectsWith=point%28-119.543+37.925%29

    polygon
        Deprecation Notice: This keyword will be deprecated. Please use 'intersectsWith' instead.
        Bounding polygon in the digital long/lat format; enter coordinates in counter clockwise direction, repeat the first point at the end to close the polygon: in the format ABCDA
        Example:
            polygon=-155.08,65.82,-153.5,61.91,-149.50,63.07,-149.94,64.55,-153.28,64.47,-155.08,65.82

Temporal Parameters

    processingDate
        Limit results to records that have been processed at ASF since a given date and/or time.
        Example:
            processingDate=2017-01-01T00:00:00UTC

    start
        Date of data acquisition. Can be used in combination with 'end'. You may enter natural language dates, or a date and/or time stamp. All times are in UTC. For more information on accepted date formats, see the Date Parser endpoint below.
        Example:
            start=May+30,+2018
            start=yesterday
            start=2010-10-30T11:59:59Z
            start=1+week+ago&end=now

    end
        Date of data acquisition. Can be used in combination with 'start'. You may enter natural language dates, or a date and/or time stamp. All times are in UTC. For more information on accepted date formats, see the Date Parser endpoint below.
        Example:
            end=May+30,+2018
            end=today
            end=2021-04-30T11:59:59Z
            start=1+week+ago&end=now

    season
        Start and end day of year for desired seasonal range. This keyword may be used in conjunction with start/end to specify a seasonal range within an overall date range. Values are based on the Julian calendar. You must specify both a season start and end date.
        Example:
            season=1,31
            season=45,67
            season=360,10
        Values:
            1 through 365

Results Parameters

    output
        Desired format of the Search API results. If not specified, the default format is metalink. The preferred format is geoJSON.
        Example:
            output=geojson
        Values:
            geojson, csv, json, kml, metalink, count, download
        Description:
            GeoJSON is the preferred output format. If a required field is not included, please contact ASF using the info below or reach the team directly at uaf-asf-discovery@alaska.edu
            KML can be opened in Google Earth, ArcGIS Earth, or a similar program
            Count returns the number of results returned by your query. It does not include any additional information. Using count output can be helpful in determining if your query has returned the correct number of results. There is a time limit on running Search API queries. See the Troubleshooting page for more details.
            Metalink provides download information for the scenes returned by your query. It does not include metadata.
            Download returns a bulk download script that includes the files returned by the search. See the Bulk Download documentation for a full guide on using the bulk download script.
            JSON includes scene metadata and product URLs. If GeoJSON does not meet your needs, JSON is the preferred format for programmatic use.
            CSV also includes scene metadata and product URLs. CSV returns less fields than JSON.

    maxResults
        Maximum number of data records to return from your query.
        Example:
            maxResults=10

Baseline Endpoint

https://api.daac.asf.alaska.edu/services/search/baseline

    reference
        This is the only mandatory keyword. Input the reference scene name for which you wish to see baseline results.
        Example:
            reference=S1B_IW_SLC__1SDV_20210704T135937_20210704T140004_027645_034CB0_4B2C

    processingLevel
        Level to which the data has been processed. Baseline data is only available for certain processing levels.
        Example:
            processingLevel=L1.5
        ProcessingLevel Values Which Contain Baseline Data:
            ALOS: L1.1, L1.5; default is L1.1
            ERS-1 & ERS-2: L0, L1; default is L0
            JERS-1: L0, L1; default is L0
            RADARSAT-1: L0, L1; default is L0
            Sentinel-1A & Sentinel-1B: SLC

    output
        Desired format of the Search API results. If not specified, the default format is metalink. The preferred format is geoJSON.
        Example:
            output=geojson
        Values:
            geojson, csv, json, kml, metalink, count, download
        Description:
            GeoJSON is the preferred output format. If a required field is not included, please contact ASF using the info below or reach the team directly at uaf-asf-discovery@alaska.edu
            KML can be opened in Google Earth, ArcGIS Earth, or a similar program
            Count returns the number of results returned by your query. It does not include any additional information. Using count output can be helpful in determining if your query has returned the correct number of results. There is a time limit on running Search API queries. See the Troubleshooting page for more details.
            Metalink provides download information for the scenes returned by your query. It does not include metadata.
            Download returns a bulk download script that includes the files returned by the search. See the Bulk Download documentation for a full guide on using the bulk download script.
            JSON includes scene metadata and product URLs. If GeoJSON does not meet your needs, JSON is the preferred format for programmatic use.
            CSV also includes scene metadata and product URLs. CSV returns less fields than JSON.

    maxResults
        Maximum number of data records to return from your query.
        Example:
            maxResults=10

WKT Validation Endpoint

https://api.daac.asf.alaska.edu/services/utils/wkt

This endpoint will validate and repair a WKT input. The repaired WKT output is how the Search API will interpret the provided WKT input. If a WKT cannot be repaired, it will return an error stating the reason. All validations and errors are returned in JSON format.

    wkt
        This is the only accepted keyword for this endpoint.
        Example:
            wkt=GEOMETRYCOLLECTION(POLYGON((46 -19,30 26,-3 41,22 39,49 16,46 -19)), POLYGON((27 24,12 4,18 31,27 24)))
            In this example, the JSON return will list the errors that were repaired, and the final wrapped and unwrapped WKT.

GeoSpatial Files to WKT Endpoint

https://api.daac.asf.alaska.edu/services/utils/files_to_wkt

This endpoint will accept a POST request with files attached. It will return the parsed WKT from the file, as well as the repaired wrapped and unwrapped WKT. All outputs are returned in JSON format. The preferred file format is geojson, but the Search API will also support other formats, such as shapefile or kml.

See the Tools page for more details on POST requests.

    Example:
        curl -X POST -F 'files=@/path/to/file' 'https://api.aac.asf.alaska.edu/services/utils/files_to_wkt'

Date Parser Endpoint

https://api.daac.asf.alaska.edu/services/utils/date

This endpoint can be used to check how dates are parsed by the Search API. All parsed dates are returned in JSON format.

    date
        This is the only accepted keyword for this endpoint. You can use natural language, such as "yesterday", dates with or without the time stamp, or days of the week.

Mission List Endpoint

https://api.daac.asf.alaska.edu/services/utils/mission_list

This endpoint lists all missions (also known as campaigns or collections) for all datasets. Any of the missions returned in the list may be used as a value for the collectionName keyword in the Search endpoint. The mission list is returned in JSON format.

    platform
        This keyword is optional. If used, it will restrict the list of missions to the specified platform(s).
        Remote sensing platform that acquired the data. Sentinel-1 and ERS have multiple remote sensing platforms, and you may choose whether to specify a specific platform. You may specify a single value, or a list of values.
        Example:
            platform=ALOS
            platform=SA,SB
            platform=S1
        Values:
            ALOS, A3, AIRSAR, AS, ERS, ERS-1, E1, ERS-2, E2, JERS-1, J1, RADARSAT-1, R1, SEASAT, SS, S1, Sentinel, Sentinel-1, Sentinel-1A, SA, Sentinel-1B, Sentinel-1 Interferogram (BETA), SB, SMAP, SP, UAVSAR, UA

Health Endpoint

https://api.daac.asf.alaska.edu/health

This endpoint is used to check the Search API health. It is returned in JSON format. There are no keywords associated with the health check endpoint.

In addition to Search API health, it also returns Search API configs and CMR health status.
'''

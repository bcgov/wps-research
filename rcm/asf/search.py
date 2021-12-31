'''
https://api.daac.asf.alaska.edu/services/search/param?keyword1=value1&keyword2=value2,value3&keyword3=value4-6
'''


url = 'https://api.daac.asf.alaska.edu/services/search/param?' #keyword1=value1&keyword2=value2,value3&keyword3=value4-6'
url += 'platform=ALOS'
url += '&instrument=PALSAR'
url += '&polarization=QUADRATURE,FULL'





'''
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

'''

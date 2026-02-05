<h1>SENTINEL-2 Data Processing</h1>

Ongoing work: 

+ Cloud masking and imagery enhancement

Past work:

+ Burn mapping


Requirements:

+ For burn-mapping, Nvidia GPU(s) are required (switched from CPU due to slow processing).


Important libraries:

+ Regular tasks:

    - gdal

+ GPU tasks:

    - cuML (rapids)


Mapping Guidelines:

+ W/  polygon file: python3  burn_mapping.py  test_C11659/small3/1009.bin

+ W/o polygon file: python3  burn_mapping.py  test_C11659/small3/1009.bin  test_C11659/small3/polygon_0000.bin


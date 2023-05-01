```
bcws_list_active.py
```

Enter appropriate folder with fpf file.

```
find_sentinel2.py
```

Enter folder for appropriate date.

```
s2u
```

Open .bin file.

```
cut.py
```

```
s2a
gimp result.bin_thres.tif
```

Crop out false positives, export to result.png

```
png2bin result.png
cp result.bin_thres.hdr result.png.hdr (for result.png.bin)

binary_polygonize.py result.png.bin
```

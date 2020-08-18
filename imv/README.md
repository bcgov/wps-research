# imv
minimalist hyperspectral image viewer. band switching by r[number]&lt;return>, g[number]&lt;return>, b[number]&lt;return>

sample test data (ENVI binary format, data type "4" aka IEEE-standard (byte-order 0) Float32, "band sequential" format) red, green, blue, near-infrared

note: doing something specific prioritized over having a readable code (this repo derives from m3ta3::glut2)

## Example: default bands (r,g,b = 1,2,3) natural colour

![sample](rgb-1,2,3.png)

## Example: band selection (r,g,b = 4,3,2) so-called false colour

![sample](rgb-4,3,2.png)

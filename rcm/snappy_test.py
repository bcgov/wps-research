from snappy import ProductIO
p = ProductIO.readProduct('snappy/testdata/MER_FRS_L1B_SUBSET.dim')
list(p.getBandNames())

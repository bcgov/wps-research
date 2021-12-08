#!/usr/bin/env bash
test ! -f band_splice.exe && g++ -w -O4 band_splice.cpp  misc.cpp -o band_splice.exe -lpthread &
test ! -f binary_confusion.exe && g++ -w -O4 binary_confusion.cpp  misc.cpp -o binary_confusion.exe -lpthread &
test ! -f binary_dilate.exe && g++ -w -O4 binary_dilate.cpp  misc.cpp -o binary_dilate.exe -lpthread &
test ! -f binary_invert.exe && g++ -w -O4 binary_invert.cpp  misc.cpp -o binary_invert.exe -lpthread &
test ! -f cat_append.exe && g++ -w -O4 cat_append.cpp  misc.cpp -o cat_append.exe -lpthread &
test ! -f class_count.exe && g++ -w -O4 class_count.cpp  misc.cpp -o class_count.exe -lpthread &
test ! -f class_match_onto.exe && g++ -w -O4 class_match_onto.cpp  misc.cpp -o class_match_onto.exe -lpthread &
test ! -f class_mean.exe && g++ -w -O4 class_mean.cpp  misc.cpp -o class_mean.exe -lpthread &
test ! -f class_mode_filter.exe && g++ -w -O4 class_mode_filter.cpp  misc.cpp -o class_mode_filter.exe -lpthread &
test ! -f class_onehot.exe && g++ -w -O4 class_onehot.cpp  misc.cpp -o class_onehot.exe -lpthread &
test ! -f class_recode.exe && g++ -w -O4 class_recode.cpp  misc.cpp -o class_recode.exe -lpthread &
test ! -f class_replace_nan.exe && g++ -w -O4 class_replace_nan.cpp  misc.cpp -o class_replace_nan.exe -lpthread &
test ! -f class_wheel.exe && g++ -w -O4 class_wheel.cpp  misc.cpp -o class_wheel.exe -lpthread &
test ! -f cluster.exe && g++ -w -O4 cluster.cpp  misc.cpp -o cluster.exe -lpthread &
test ! -f convert_cplx_to_iq.exe && g++ -w -O4 convert_cplx_to_iq.cpp  misc.cpp -o convert_cplx_to_iq.exe -lpthread &
test ! -f convert_iq_to_cplx.exe && g++ -w -O4 convert_iq_to_cplx.cpp  misc.cpp -o convert_iq_to_cplx.exe -lpthread
test ! -f convert_iq_to_s2.exe && g++ -w -O4 convert_iq_to_s2.cpp  misc.cpp -o convert_iq_to_s2.exe -lpthread &
test ! -f csv_spectra_class_raster_dist.exe && g++ -w -O4 csv_spectra_class_raster_dist.cpp  misc.cpp -o csv_spectra_class_raster_dist.exe -lpthread &
test ! -f csv_split.exe && g++ -w -O4 csv_split.cpp  misc.cpp -o csv_split.exe -lpthread &
test ! -f cut.exe && g++ -w -O4 cut.cpp  misc.cpp -o cut.exe -lpthread &
test ! -f cv.exe && g++ -w -O4 cv.cpp  misc.cpp -o cv.exe -lpthread &
test ! -f hclust.exe && g++ -w -O4 hclust.cpp  misc.cpp -o hclust.exe -lpthread &
test ! -f hsv2rgb.exe && g++ -w -O4 hsv2rgb.cpp  misc.cpp -o hsv2rgb.exe -lpthread &
test ! -f htrim2.exe && g++ -w -O4 htrim2.cpp  misc.cpp -o htrim2.exe -lpthread &
test ! -f kmeans.exe && g++ -w -O4 kmeans.cpp  misc.cpp -o kmeans.exe -lpthread &
test ! -f kmeans_iter.exe && g++ -w -O4 kmeans_iter.cpp  misc.cpp -o kmeans_iter.exe -lpthread &
test ! -f kmeans_multi.exe && g++ -w -O4 kmeans_multi.cpp  misc.cpp -o kmeans_multi.exe -lpthread &
test ! -f multilook.exe && g++ -w -O4 multilook.cpp  misc.cpp -o multilook.exe -lpthread &
test ! -f multilook_large.exe && g++ -w -O4 multilook_large.cpp  misc.cpp -o multilook_large.exe -lpthread &
test ! -f multiply.exe && g++ -w -O4 multiply.cpp  misc.cpp -o multiply.exe -lpthread &
test ! -f nodata.exe && g++ -w -O4 nodata.cpp  misc.cpp -o nodata.exe -lpthread &
test ! -f raster_at.exe && g++ -w -O4 raster_at.cpp  misc.cpp -o raster_at.exe -lpthread
test ! -f raster_difference.exe && g++ -w -O4 raster_difference.cpp  misc.cpp -o raster_difference.exe -lpthread &
test ! -f raster_dominant.exe && g++ -w -O4 raster_dominant.cpp  misc.cpp -o raster_dominant.exe -lpthread &
test ! -f raster_increment.exe && g++ -w -O4 raster_increment.cpp  misc.cpp -o raster_increment.exe -lpthread &
test ! -f raster_inequality.exe && g++ -w -O4 raster_inequality.cpp  misc.cpp -o raster_inequality.exe -lpthread &
test ! -f raster_mask.exe && g++ -w -O4 raster_mask.cpp  misc.cpp -o raster_mask.exe -lpthread &
test ! -f raster_mode_filter.exe && g++ -w -O4 raster_mode_filter.cpp  misc.cpp -o raster_mode_filter.exe -lpthread &
test ! -f raster_nan_to_zero.exe && g++ -w -O4 raster_nan_to_zero.cpp  misc.cpp -o raster_nan_to_zero.exe -lpthread &
test ! -f raster_nearest_centre.exe && g++ -w -O4 raster_nearest_centre.cpp  misc.cpp -o raster_nearest_centre.exe -lpthread &
test ! -f raster_negate.exe && g++ -w -O4 raster_negate.cpp  misc.cpp -o raster_negate.exe -lpthread &
test ! -f raster_normalize.exe && g++ -w -O4 raster_normalize.cpp  misc.cpp -o raster_normalize.exe -lpthread &
test ! -f raster_quickstats.exe && g++ -w -O4 raster_quickstats.cpp  misc.cpp -o raster_quickstats.exe -lpthread &
test ! -f raster_relative_difference.exe && g++ -w -O4 raster_relative_difference.cpp  misc.cpp -o raster_relative_difference.exe -lpthread &
test ! -f raster_sub.exe && g++ -w -O4 raster_sub.cpp  misc.cpp -o raster_sub.exe -lpthread &
test ! -f raster_sum.exe && g++ -w -O4 raster_sum.cpp  misc.cpp -o raster_sum.exe -lpthread &
test ! -f raster_zero_to_nan.exe && g++ -w -O4 raster_zero_to_nan.cpp  misc.cpp -o raster_zero_to_nan.exe -lpthread &
test ! -f snip.exe && g++ -w -O4 snip.cpp  misc.cpp -o snip.exe -lpthread
test ! -f squiggle.exe && g++ -w -O4 squiggle.cpp  misc.cpp -o squiggle.exe -lpthread &
test ! -f unstack.exe && g++ -w -O4 unstack.cpp  misc.cpp -o unstack.exe -lpthread &
test ! -f vri_rasterize.exe && g++ -w -O4 vri_rasterize.cpp  misc.cpp -o vri_rasterize.exe -lpthread &
wait


/*
raster_flood_seg_supervised_classifier.cu

Compile:
nvcc -O3 -o raster_flood_seg_supervised_classifier misc.cpp raster_flood_seg_supervised_classifier.cu

Invocation example:
./raster_flood_seg_supervised_classifier stack_nonsnow.bin ftl.bin_class.bin_cc_labels_float.bin ftl.bin_class.bin --mode
*/

#include "misc.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <vector>
#include <unordered_map>
#include <limits>
#include <algorithm>
#include <iostream>
#include <atomic>
#include <chrono>

#include <cuda_runtime.h>

using std::size_t;
using std::chrono::steady_clock;
using std::chrono::duration_cast;
using std::chrono::seconds;

// ---------------------------------------------------
// CUDA kernel: L1 distance for mode computation
// Each thread handles one pixel in one segment
// ---------------------------------------------------
__global__ void l1_distance_kernel(
    const float* img,           // nband x np
    const int* pixel_indices,   // flattened per-segment pixel indices
    const size_t* seg_offsets,  // starting index in pixel_indices for each segment
    const size_t* seg_sizes,    // number of pixels in each segment
    float* pixel_sums,
    size_t nband,
    size_t np
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Find which segment this idx belongs to
    int seg = 0;
    while(seg_offsets[seg+1]<=idx) seg++;

    size_t local_idx = idx - seg_offsets[seg];
    if(local_idx >= seg_sizes[seg]) return;

    int pix_idx = pixel_indices[seg_offsets[seg] + local_idx];
    size_t npix = seg_sizes[seg];

    float sum_d = 0.0f;
    for(size_t k=0;k<npix;k++){
        if(k==local_idx) continue;
        int idx2 = pixel_indices[seg_offsets[seg] + k];
        for(size_t b=0;b<nband;b++){
            sum_d += fabsf(img[b*np + pix_idx] - img[b*np + idx2]);
        }
    }
    pixel_sums[idx] = sum_d;
}

// ---------------------------------------------------
int main(int argc, char** argv) {
    if(argc < 4) {
        err("Usage: raster_flood_seg_supervised_classifier [image.bin] [segment_index.bin] [class_map.bin] [--mode optional]");
    }

    bool use_mode = false;
    if(argc > 4 && strcmp(argv[4], "--mode") == 0)
        use_mode = true;

    str img_fn(argv[1]);
    str seg_fn(argv[2]);
    str cls_fn(argv[3]);

    // ---------------------------------------------------
    // Read headers
    // ---------------------------------------------------
    str h_img(hdr_fn(img_fn));
    str h_seg(hdr_fn(seg_fn));
    str h_cls(hdr_fn(cls_fn));

    size_t nrow, ncol, nband;
    std::vector<std::string> bandNames;
    hread(h_img, nrow, ncol, nband, bandNames);

    size_t nrow_seg, ncol_seg, nband_seg;
    hread(h_seg, nrow_seg, ncol_seg, nband_seg, bandNames);
    size_t nrow_cls, ncol_cls, nband_cls;
    hread(h_cls, nrow_cls, ncol_cls, nband_cls, bandNames);

    if(nrow != nrow_seg || ncol != ncol_seg || nrow != nrow_cls || ncol != ncol_cls)
        err("All inputs must have same dimensions.");
    if(nband_seg!=1 || nband_cls!=1)
        err("Segment and class maps must be single-band.");

    size_t np = nrow*ncol;

    // ---------------------------------------------------
    // Read image, segment, and class maps
    // ---------------------------------------------------
    float* img = bread(img_fn, nrow, ncol, nband);
    float* seg = bread(seg_fn, nrow, ncol, nband_seg);
    float* cls = bread(cls_fn, nrow, ncol, nband_cls);

    if(!img || !seg || !cls)
        err("Failed to read inputs.");

    // ---------------------------------------------------
    // Compute per-segment stats
    // ---------------------------------------------------
    std::unordered_map<int,std::vector<size_t>> segment_pixels;
    for(size_t i=0;i<np;i++){
        if(!std::isnan(seg[i])){
            int sid = (int)seg[i];
            segment_pixels[sid].push_back(i);
        }
    }

    std::unordered_map<int,std::vector<float>> seg_means;

    if(!use_mode){
        // Segment mean (CPU)
        std::vector<float> pix(nband);
        for(auto &kv : segment_pixels){
            int sid = kv.first;
            const auto &pixels = kv.second;
            std::vector<float> mean_vec(nband,0.0f);
            for(size_t idx: pixels){
                for(size_t b=0;b<nband;b++)
                    mean_vec[b] += img[b*np+idx];
            }
            for(size_t b=0;b<nband;b++)
                mean_vec[b]/=pixels.size();
            seg_means[sid] = mean_vec;
        }
    } else {
        // ---------------------------------------------------
        // Mode computation on GPU with progress
        // ---------------------------------------------------
        size_t nseg = segment_pixels.size();
        size_t total_jobs = 0;
        for(auto &kv: segment_pixels) total_jobs += kv.second.size();

        // Flatten pixel indices
        std::vector<int> pixel_indices(total_jobs);
        std::vector<size_t> seg_offsets(nseg+1,0);
        std::vector<size_t> seg_sizes(nseg,0);

        size_t offset = 0;
        size_t seg_idx = 0;
        for(auto &kv: segment_pixels){
            seg_offsets[seg_idx] = offset;
            seg_sizes[seg_idx] = kv.second.size();
            for(size_t i=0;i<kv.second.size();i++){
                pixel_indices[offset+i] = kv.second[i];
            }
            offset += kv.second.size();
            seg_idx++;
        }
        seg_offsets[seg_idx] = offset;

        // Allocate GPU memory
        float* d_img;
        int* d_pixel_indices;
        size_t* d_seg_offsets;
        size_t* d_seg_sizes;
        float* d_pixel_sums;

        cudaMalloc(&d_img, nband*np*sizeof(float));
        cudaMalloc(&d_pixel_indices, total_jobs*sizeof(int));
        cudaMalloc(&d_seg_offsets, (nseg+1)*sizeof(size_t));
        cudaMalloc(&d_seg_sizes, nseg*sizeof(size_t));
        cudaMalloc(&d_pixel_sums, total_jobs*sizeof(float));

        cudaMemcpy(d_img, img, nband*np*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_pixel_indices, pixel_indices.data(), total_jobs*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_seg_offsets, seg_offsets.data(), (nseg+1)*sizeof(size_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_seg_sizes, seg_sizes.data(), nseg*sizeof(size_t), cudaMemcpyHostToDevice);

        // Launch kernel in batches for progress update
        size_t batch_size = 50000; // can tune
        std::vector<float> pixel_sums(total_jobs);
        auto start_time = steady_clock::now();
        size_t done = 0;
        for(size_t batch_start=0; batch_start<total_jobs; batch_start+=batch_size){
            size_t batch_end = std::min(batch_start+batch_size,total_jobs);
            int blocks = (batch_end-batch_start+255)/256;
            l1_distance_kernel<<<blocks,256>>>(
                d_img, d_pixel_indices, d_seg_offsets, d_seg_sizes, d_pixel_sums, nband, np
            );
            cudaDeviceSynchronize();
            // Copy batch back
            cudaMemcpy(pixel_sums.data()+batch_start, d_pixel_sums+batch_start, (batch_end-batch_start)*sizeof(float), cudaMemcpyDeviceToHost);
            // Progress
            done = batch_end;
            double frac = double(done)/total_jobs;
            auto now = steady_clock::now();
            double elapsed = duration_cast<seconds>(now-start_time).count();
            double eta = elapsed/frac - elapsed;
            printf("Mode processing progress: %zu / %zu (%.2f%%), ETA %.0f s\n", done, total_jobs, frac*100.0, eta);
        }

        // Pick minimum per segment
        seg_idx = 0;
        offset = 0;
        for(auto &kv: segment_pixels){
            int sid = kv.first;
            const auto &pixels = kv.second;
            float minval = std::numeric_limits<float>::max();
            size_t best_idx = 0;
            for(size_t i=0;i<pixels.size();i++){
                if(pixel_sums[offset+i]<minval){
                    minval = pixel_sums[offset+i];
                    best_idx = i;
                }
            }
            std::vector<float> mode_vec(nband);
            size_t pix_idx = pixels[best_idx];
            for(size_t b=0;b<nband;b++) mode_vec[b] = img[b*np+pix_idx];
            seg_means[sid] = mode_vec;
            offset += pixels.size();
            seg_idx++;
        }

        cudaFree(d_img);
        cudaFree(d_pixel_indices);
        cudaFree(d_seg_offsets);
        cudaFree(d_seg_sizes);
        cudaFree(d_pixel_sums);
    }

    // ---------------------------------------------------
    // Compute per-class means
    // ---------------------------------------------------
    std::unordered_map<int, std::vector<float>> class_accum;
    std::unordered_map<int,int> class_count;
    for(size_t i=0;i<np;i++){
        if(std::isnan(seg[i]) || std::isnan(cls[i])) continue;
        int sid = (int)seg[i];
        int cid = (int)cls[i];
        auto it = seg_means.find(sid);
        if(it==seg_means.end()) continue;
        if(class_accum.find(cid)==class_accum.end())
            class_accum[cid] = std::vector<float>(nband,0.0f);
        for(size_t b=0;b<nband;b++)
            class_accum[cid][b] += it->second[b];
        class_count[cid]++;
    }

    std::unordered_map<int,std::vector<float>> class_means;
    for(auto &kv: class_accum){
        int cid = kv.first;
        class_means[cid] = kv.second;
        for(size_t b=0;b<nband;b++)
            class_means[cid][b] /= class_count[cid];
    }

    // ---------------------------------------------------
    // Build output images
    // ---------------------------------------------------
    std::vector<float> out_segment(nband*np,NAN);
    for(size_t i=0;i<np;i++){
        if(std::isnan(seg[i])) continue;
        int sid = (int)seg[i];
        auto it = seg_means.find(sid);
        if(it==seg_means.end()) continue;
        for(size_t b=0;b<nband;b++)
            out_segment[b*np+i] = it->second[b];
    }

    str out_seg_fn(img_fn + (use_mode ? "_segment_mode.bin" : "_segment_average.bin"));
    str out_seg_hdr(hdr_fn(out_seg_fn,true));
    bwrite(out_segment.data(), out_seg_fn, nrow, ncol, nband);
    hwrite(out_seg_hdr,nrow,ncol,nband,4,bandNames);

    // Classification map
    std::vector<float> out_class(np,NAN);
    std::vector<float> pix(nband);
    for(size_t i=0;i<np;i++){
        bool skip=false;
        for(size_t b=0;b<nband;b++){
            pix[b]=img[b*np+i];
            if(std::isnan(pix[b])) {skip=true;break;}
        }
        if(skip) continue;
        double best_d = std::numeric_limits<double>::max();
        int best_c = -9999;
        for(auto &kv: class_means){
            float d=0.0f;
            for(size_t b=0;b<nband;b++)
                d += fabsf(pix[b]-kv.second[b]);
            if(d<best_d){best_d=d;best_c=kv.first;}
        }
        out_class[i] = (float)best_c;
    }

    str out_cls_fn(img_fn + "_classification_map.bin");
    str out_cls_hdr(hdr_fn(out_cls_fn,true));
    bwrite(out_class.data(), out_cls_fn, nrow, ncol, 1);
    hwrite(out_cls_hdr,nrow,ncol,1,4);

    free(img);
    free(seg);
    free(cls);

    return 0;
}




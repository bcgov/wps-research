/* 20251020 inverse of class_onehot.cpp, i.e., convert one-hot encoding to class map */

/* convert one-hot encoded ENVI image to numbered classes */
#include "misc.h"
#include <cmath>   // for std::isnan
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <iostream>

using namespace std;

int main(int argc, char ** argv){
    if(argc < 2) err("Usage: onehot_to_class [input binary (one-hot) file name]");

    str fn(argv[1]);            // input file name
    str hfn(hdr_fn(fn));        // header file
    size_t nrow, ncol, nband;
    vector<str> band_names;
    hread(hfn, nrow, ncol, nband, band_names); // read header

    if(nband < 1) err("input must have at least 1 band (one-hot encoding)");

    size_t np = nrow * ncol;
    size_t p1 = np / 100;

    printf("Reading data...\n");
    float * dat = bread(fn, nrow, ncol, nband); // read all bands into flat array

    printf("Converting to class map...\n");
    float * out = (float*)malloc(np * sizeof(float));
    if(!out) err("Memory allocation failed");

    for(size_t i = 0; i < np; i++){
        bool assigned = false;
        for(size_t b = 0; b < nband; b++){
            float val = dat[i + b * np];
            if(val != 0.0f && !std::isnan(val)){
                out[i] = (float)b;  // class index starts at 0
                assigned = true;
                break; // first hot band wins
            }
        }
        if(!assigned){
            out[i] = NAN; // no class assigned
        }

        if(i % p1 == 0) printf("Processing %zu / 100\n", i/p1);
    }

    str ofn(fn + str("_class.bin"));
    str ohfn(fn + str("_class.hdr"));
    hwrite(ohfn, nrow, ncol, 1);  // single-band output

    // Example class names (update these as needed)
    /*vector<string> class_names;
    for (size_t i = 0; i < nband; ++i) {
        class_names.push_back("class_" + to_string(i));
    }
    */
    // Append class names to header
    happend_class_names(ohfn, band_names);

    FILE * f = fopen(ofn.c_str(), "wb");
    if(!f) err("Failed to open output file");
    fwrite(out, sizeof(float), np, f);
    fclose(f);

    free(dat);
    free(out);

    printf("Done: %s\n", ofn.c_str());
    return 0;
}

/* multilook a multispectral image or radar stack, square or rectangular window
   input assumed ENVI type-4 32-bit IEEE standard floating-point format, BSQ interleave
   20230524 band by band processing to support larger file
   20220506 consider all-zero (if nbands > 1) equivalent to NAN / no-data
   20250203 parallel version using parfor
   20250203 per-band arrays, parallel alloc, directory batch mode */

#include"misc.h"
#include<glob.h>
#include<sys/stat.h>

// global state for parallel processing
size_t g_nrow, g_ncol, g_nband, g_np;
size_t g_nrow2, g_ncol2, g_np2;
size_t g_n, g_m;  // multilook factors (row, col)
float **g_dat;    // input data pointers [nband] -> [nrow * ncol]
float **g_dat2;   // output data pointers [nband] -> [nrow2 * ncol2]
str g_fn;

// parallel memory allocation for input bands
void alloc_input_band(size_t k){
    g_dat[k] = falloc(g_np);
    if(!g_dat[k]) fprintf(stderr, "error: failed to allocate input band %zu\n", k);
}

// parallel memory allocation for output bands
void alloc_output_band(size_t k){
    g_dat2[k] = falloc(g_np2);
    if(!g_dat2[k]) fprintf(stderr, "error: failed to allocate output band %zu\n", k);
}

// parallel band read: each worker reads one band
void read_band(size_t k){
    FILE *f = fopen(g_fn.c_str(), "rb");
    if(!f){
        fprintf(stderr, "error: failed to open %s for band %zu\n", g_fn.c_str(), k);
        return;
    }
    fseek(f, k * g_np * sizeof(float), SEEK_SET);
    size_t nr = fread(g_dat[k], sizeof(float), g_np, f);
    if(nr != g_np) fprintf(stderr, "warning: band %zu read %zu of %zu\n", k, nr, g_np);
    fclose(f);
}

// parallel processing: each worker handles one output row (all bands)
void process_output_row(size_t ip){
    if(ip >= g_nrow2) return;  // bounds check
    
    size_t i_start = ip * g_n;
    size_t i_end = i_start + g_n;
    if(i_end > g_nrow) i_end = g_nrow;
    
    for(size_t k = 0; k < g_nband; k++){
        float *dat_k = g_dat[k];
        float *dat2_k = g_dat2[k];
        
        if(!dat_k || !dat2_k) continue;  // safety check
        
        for(size_t jp = 0; jp < g_ncol2; jp++){
            size_t j_start = jp * g_m;
            size_t j_end = j_start + g_m;
            if(j_end > g_ncol) j_end = g_ncol;
            
            float sum = 0.f;
            float count = 0.f;
            
            for(size_t i = i_start; i < i_end; i++){
                for(size_t j = j_start; j < j_end; j++){
                    float d = dat_k[i * g_ncol + j];
                    if(!isnan(d) && !isinf(d)){
                        sum += d;
                        count += 1.f;
                    }
                }
            }
            
            size_t out_idx = ip * g_ncol2 + jp;
            dat2_k[out_idx] = (count > 0.f) ? (sum / count) : NAN;
        }
    }
}

// parallel band write: each worker writes one band
void write_band(size_t k){
    str ofn(g_fn + str("_mlk.bin"));
    FILE *g = fopen(ofn.c_str(), "r+b");
    if(!g){
        fprintf(stderr, "error: failed to open %s for writing band %zu\n", ofn.c_str(), k);
        return;
    }
    fseek(g, k * g_np2 * sizeof(float), SEEK_SET);
    size_t nw = fwrite(g_dat2[k], sizeof(float), g_np2, g);
    if(nw != g_np2) fprintf(stderr, "warning: band %zu wrote %zu of %zu\n", k, nw, g_np2);
    fclose(g);
}

// parallel memory free for input bands
void free_input_band(size_t k){
    if(g_dat[k]){ free(g_dat[k]); g_dat[k] = NULL; }
}

// parallel memory free for output bands
void free_output_band(size_t k){
    if(g_dat2[k]){ free(g_dat2[k]); g_dat2[k] = NULL; }
}

// process a single file (assumes global dims and memory already set up)
void process_file(const str &fn){
    g_fn = fn;
    str hfn(hdr_fn(g_fn));
    str ofn(g_fn + str("_mlk.bin"));
    str ohfn(g_fn + str("_mlk.hdr"));
    
    vector<str> band_names(parse_band_names(hfn));
    
    printf("processing: %s\n", fn.c_str());
    fflush(stdout);

    // parallel read: one worker per band
    printf("  reading...\n"); fflush(stdout);
    parfor(0, g_nband - 1, read_band, g_nband);

    // parallel processing: one worker per output row, max cores
    printf("  processing %zu output rows...\n", g_nrow2); fflush(stdout);
    parfor(0, g_nrow2 - 1, process_output_row);

    // create output file (pre-allocate for parallel writes)
    printf("  creating output file...\n"); fflush(stdout);
    FILE *g = fopen(ofn.c_str(), "wb");
    if(!g){
        fprintf(stderr, "error: failed to create %s\n", ofn.c_str());
        return;
    }
    fseek(g, g_nband * g_np2 * sizeof(float) - 1, SEEK_SET);
    fputc(0, g);
    fclose(g);

    // parallel write: one worker per band
    printf("  writing...\n"); fflush(stdout);
    parfor(0, g_nband - 1, write_band, g_nband);

    // write header
    hwrite(ohfn, g_nrow2, g_ncol2, g_nband, 4, band_names);
    
    printf("  -> %s\n", ofn.c_str());
    fflush(stdout);
}

int main(int argc, char ** argv){
    if(argc < 3)
        err("multilook [input binary file or directory] [vertical or square multilook factor] "
            "# [optional: horiz multilook factor]");

    str input_path(argv[1]);
    g_n = (size_t)atol(argv[2]);
    g_m = (argc > 3) ? (size_t)atol(argv[3]) : g_n;
    
    printf("multilook factor (row): %zu\n", g_n);
    printf("multilook factor (col): %zu\n", g_m);

    // check if input is a directory
    vector<str> files;
    struct stat st;
    stat(input_path.c_str(), &st);
    
    if(S_ISDIR(st.st_mode)){
        // glob for *.bin files in directory
        str pattern = input_path + str("/*.bin");
        glob_t globbuf;
        glob(pattern.c_str(), 0, NULL, &globbuf);
        
        for(size_t i = 0; i < globbuf.gl_pathc; i++)
            files.push_back(str(globbuf.gl_pathv[i]));
        
        globfree(&globbuf);
        
        if(files.size() == 0)
            err("no .bin files found in directory");
        
        printf("found %zu .bin files in directory\n", files.size());
    }
    else{
        files.push_back(input_path);
    }

    // read first header to establish dimensions
    str first_hfn(hdr_fn(files[0]));
    hread(first_hfn, g_nrow, g_ncol, g_nband);
    
    // verify all files have same dimensions
    for(size_t i = 1; i < files.size(); i++){
        size_t nrow_i, ncol_i, nband_i;
        str hfn_i(hdr_fn(files[i]));
        hread(hfn_i, nrow_i, ncol_i, nband_i);
        
        if(nrow_i != g_nrow || ncol_i != g_ncol || nband_i != g_nband)
            err(str("dimension mismatch: ") + files[i] + 
                str(" (") + to_string(nrow_i) + str("x") + to_string(ncol_i) + str("x") + to_string(nband_i) +
                str(") != ") + files[0] +
                str(" (") + to_string(g_nrow) + str("x") + to_string(g_ncol) + str("x") + to_string(g_nband) + str(")"));
    }
    
    g_np = g_nrow * g_ncol;
    g_nrow2 = g_nrow / g_n;
    g_ncol2 = g_ncol / g_m;
    g_np2 = g_nrow2 * g_ncol2;
    
    printf("input dims: %zu rows x %zu cols x %zu bands\n", g_nrow, g_ncol, g_nband);
    printf("output dims: %zu rows x %zu cols x %zu bands\n", g_nrow2, g_ncol2, g_nband);
    
    size_t input_bytes = g_nband * g_np * sizeof(float);
    size_t output_bytes = g_nband * g_np2 * sizeof(float);
    printf("memory required: %.2f GB input + %.2f GB output = %.2f GB total\n",
           input_bytes / 1e9, output_bytes / 1e9, (input_bytes + output_bytes) / 1e9);
    fflush(stdout);

    // allocate band pointer arrays
    g_dat = new float*[g_nband];
    g_dat2 = new float*[g_nband];
    for(size_t k = 0; k < g_nband; k++){
        g_dat[k] = NULL;
        g_dat2[k] = NULL;
    }

    // parallel allocation of input bands
    printf("allocating input memory...\n"); fflush(stdout);
    parfor(0, g_nband - 1, alloc_input_band, g_nband);

    // parallel allocation of output bands
    printf("allocating output memory...\n"); fflush(stdout);
    parfor(0, g_nband - 1, alloc_output_band, g_nband);
    
    // verify allocations
    for(size_t k = 0; k < g_nband; k++){
        if(!g_dat[k]) err(str("failed to allocate input band ") + to_string(k));
        if(!g_dat2[k]) err(str("failed to allocate output band ") + to_string(k));
    }
    printf("memory allocation complete\n"); fflush(stdout);

    // process all files
    for(size_t i = 0; i < files.size(); i++){
        printf("\n[%zu/%zu] ", i + 1, files.size());
        fflush(stdout);
        process_file(files[i]);
    }

    // parallel free of input bands
    printf("\nfreeing input memory...\n"); fflush(stdout);
    parfor(0, g_nband - 1, free_input_band, g_nband);

    // parallel free of output bands
    printf("freeing output memory...\n"); fflush(stdout);
    parfor(0, g_nband - 1, free_output_band, g_nband);

    delete[] g_dat;
    delete[] g_dat2;
    
    printf("done.\n");
    return 0;
}

/* multilook a multispectral image or radar stack, square or rectangular window
   input assumed ENVI type-4 32-bit IEEE standard floating-point format, BSQ interleave
   20230524 band by band processing to support larger file
   20220506 consider all-zero (if nbands > 1) equivalent to NAN / no-data
   20250203 parallel version using parfor */

#include"misc.h"

// global state for parallel processing
size_t g_nrow, g_ncol, g_nband, g_np;
size_t g_nrow2, g_ncol2, g_np2;
size_t g_n, g_m;  // multilook factors (row, col)
float *g_dat;     // input data [nband][nrow][ncol]
float *g_dat2;    // output data [nband][nrow2][ncol2]
str g_fn;

// parallel band read: each worker reads one band
void read_band(size_t k){
    FILE *f = fopen(g_fn.c_str(), "rb");
    fseek(f, k * g_np * sizeof(float), SEEK_SET);
    fread(&g_dat[k * g_np], sizeof(float), g_np, f);
    fclose(f);
}

// parallel processing: each worker handles one output row (all bands)
// accumulates and normalizes in one pass - no synchronization needed
void process_output_row(size_t ip){
    size_t i_start = ip * g_n;
    size_t i_end = min(i_start + g_n, g_nrow);

    for(size_t k = 0; k < g_nband; k++){
        float *dat_k = &g_dat[k * g_np];
        float *dat2_k = &g_dat2[k * g_np2];

        for(size_t jp = 0; jp < g_ncol2; jp++){
            size_t j_start = jp * g_m;
            size_t j_end = min(j_start + g_m, g_ncol);

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

            size_t ix2 = ip * g_ncol2 + jp;
            dat2_k[ix2] = (count > 0.f) ? (sum / count) : NAN;
        }
    }
}

// parallel band write: each worker writes one band
void write_band(size_t k){
    str ofn(g_fn + str("_mlk.bin"));
    FILE *g = fopen(ofn.c_str(), "r+b");
    fseek(g, k * g_np2 * sizeof(float), SEEK_SET);
    fwrite(&g_dat2[k * g_np2], sizeof(float), g_np2, g);
    fclose(g);
}

int main(int argc, char ** argv){
    if(argc < 3)
        err("multilook [input binary file name] [vertical or square multilook factor] "
            "# [optional: horiz multilook factor]");

    g_fn = str(argv[1]);
    str hfn(hdr_fn(g_fn));
    str ofn(g_fn + str("_mlk.bin"));
    str ohfn(g_fn + str("_mlk.hdr"));

    vector<str> band_names(parse_band_names(hfn));
    hread(hfn, g_nrow, g_ncol, g_nband);

    g_np = g_nrow * g_ncol;
    g_n = (size_t)atol(argv[2]);
    g_m = (argc > 3) ? (size_t)atol(argv[3]) : g_n;

    printf("multilook factor (row): %zu\n", g_n);
    printf("multilook factor (col): %zu\n", g_m);
    printf("input: %zu rows x %zu cols x %zu bands\n", g_nrow, g_ncol, g_nband);

    g_nrow2 = g_nrow / g_n;
    g_ncol2 = g_ncol / g_m;
    g_np2 = g_nrow2 * g_ncol2;

    printf("output: %zu rows x %zu cols x %zu bands\n", g_nrow2, g_ncol2, g_nband);

    // allocate memory for all bands
    g_dat = falloc(g_nband * g_np);
    g_dat2 = falloc(g_nband * g_np2);

    // parallel read: one worker per band
    printf("reading %zu bands in parallel...\n", g_nband);
    parfor(0, g_nband - 1, read_band, g_nband);

    // parallel processing: one worker per output row, max cores
    printf("processing...\n");
    parfor(0, g_nrow2 - 1, process_output_row);

    // create output file (pre-allocate for parallel writes)
    FILE *g = fopen(ofn.c_str(), "wb");
    fseek(g, g_nband * g_np2 * sizeof(float) - 1, SEEK_SET);
    fputc(0, g);
    fclose(g);

    // parallel write: one worker per band
    printf("writing %zu bands in parallel...\n", g_nband);
    parfor(0, g_nband - 1, write_band, g_nband);

    // write header
    hwrite(ohfn, g_nrow2, g_ncol2, g_nband, 4, band_names);
    printf("done: %zu x %zu x %zu\n", g_nrow2, g_ncol2, g_nband);

    free(g_dat);
    free(g_dat2);

    return 0;
}

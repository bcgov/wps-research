/*
 * band_timeseries_gui.cpp
 *
 * Fast interactive multi-band ENVI (.bin/.hdr) time-series viewer.
 *
 * Build (no GDAL dependency):
 *   g++ -O2 -std=c++17 band_timeseries_gui.cpp \
 *       -lGL -lGLU -lglut -lpthread -o band_ts
 *
 * Usage:
 *   ./band_ts [options]
 *
 *   -d <dir>    Directory containing .bin + .hdr files       (default: ".")
 *   -w <int>    Sampling-square side-length in pixels        (default: 10)
 *   -i <int>    Initial image index (0-based, -1 = last)    (default: 0)
 *   -e <ext>    File extension filter                       (default: "bin")
 *
 * Controls (work in ANY window):
 *   Left / Right arrow   Navigate backward / forward through dates
 *   c                    Clear all sampling squares and time-series traces
 *   q / Escape           Clean exit
 *   Left-click on image  Add sampling square (max 3, R/G/B); TS plots update
 *   Right-click on image Clear all squares and traces
 *
 * Image display:
 *   Uses a multilook scheme — non-overlapping M×M boxes averaged — so the
 *   displayed image is floor(nrow/M) × floor(ncol/M).  M is chosen so the
 *   image fills at most half the screen width.  Multilook results are cached
 *   in .restore_multilook_<filename> for instant restart.
 *
 * Time-series envelopes:
 *   Solid line = mean over the M×M box at the clicked pixel.
 *   Dashed lines = min and max over that same M×M box.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <climits>
#include <cfloat>
#include <fstream>
#include <sstream>
#include <iostream>
#include <unistd.h>
#include <pthread.h>
#include <dirent.h>
#include <sys/stat.h>
#include <set>
#include <map>
#include <functional>

#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/freeglut.h>

using namespace std;

#define for0(i,n) for(i = 0; i < n; i++)

static void err(const string& msg) { cerr << "Error: " << msg << endl; exit(1); }

static size_t fsize(const string& fn) {
    FILE* f = fopen(fn.c_str(), "rb");
    if (!f) return 0;
    fseek(f, 0L, SEEK_END);
    size_t sz = ftell(f);
    fclose(f);
    return sz;
}

static bool exists(const string& fn) { return fsize(fn) > 0; }

/* ─────────────────── ENVI header reading ────────────────────────── */

static string hdr_fn(const string& fn) {
    string base = fn.substr(0, fn.size() - 4);
    string hfn = base + ".hdr";
    if (exists(hfn)) return hfn;
    hfn = fn + ".hdr";
    if (exists(hfn)) return hfn;
    err(string("header not found for: ") + fn);
    return "";
}

static void hread(const string& hfn, size_t& nrow, size_t& ncol, size_t& nband,
                  vector<string>& band_names) {
    ifstream hf(hfn);
    if (!hf.is_open()) err(string("failed to open header: ") + hfn);
    nrow = ncol = nband = 0;
    string line;
    bool in_bn = false;
    string bn_accum;
    while (getline(hf, line)) {
        if (in_bn) { bn_accum += line; if (line.find('}') != string::npos) in_bn = false; continue; }
        size_t eq = line.find('=');
        if (eq == string::npos) continue;
        string key = line.substr(0, eq), val = line.substr(eq + 1);
        auto trim = [](string& s) {
            while (!s.empty() && isspace(s.front())) s.erase(s.begin());
            while (!s.empty() && isspace(s.back()))  s.pop_back();
        };
        trim(key); trim(val);
        if (key == "samples") ncol = atoi(val.c_str());
        else if (key == "lines") nrow = atoi(val.c_str());
        else if (key == "bands") nband = atoi(val.c_str());
        else if (key == "band names") { bn_accum = val; if (val.find('}') == string::npos) in_bn = true; }
    }
    hf.close();
    if (!bn_accum.empty()) {
        size_t i; string clean;
        for0(i, bn_accum.size()) { char c = bn_accum[i]; if (c != '{' && c != '}') clean += c; }
        istringstream iss(clean); string tok;
        while (getline(iss, tok, ',')) {
            while (!tok.empty() && isspace(tok.front())) tok.erase(tok.begin());
            while (!tok.empty() && isspace(tok.back()))  tok.pop_back();
            if (!tok.empty()) band_names.push_back(tok);
        }
    }
}

/* ──────────────────────── data types ────────────────────────────── */

struct BandImage {
    size_t width = 0, height = 0, nBands = 0;
    float* data = nullptr;
    vector<string> bandNames;
    string filename, timestamp;
    const float* band(size_t b) const { return data + b * width * height; }
    float* band(size_t b) { return data + b * width * height; }
};

struct Click { int x, y; };           /* coordinates in multilooked image space */
struct TSPoint { float mean, mn, mx; }; /* mean, min, max over M×M box */

/* ──────────────────────── globals ───────────────────────────────── */

static vector<BandImage> g_images;
static vector<string>    g_filenames;
static string            g_dir;
static int g_curIdx      = 0;
static int g_squareWidth = 10;

/* original full-res dimensions */
static int g_fullW = 0, g_fullH = 0;
static int g_nBands = 0;
static vector<string> g_bandNames;

/* multilook parameters */
static int g_mlFactor = 1;            /* M: multilook box size */
static int g_mlW = 0, g_mlH = 0;     /* multilooked image dims = floor(full/M) */

/* per-image multilooked band data: g_mlData[img_idx][band * mlW * mlH + pixel] */
static vector<float*> g_mlData;

/* NBR */
static int g_b08_idx = -1, g_b12_idx = -1;
static bool g_hasNBR = false;
static int g_winNBR = -1;

/* clicks: max 3, R/G/B */
static vector<Click> g_clicks;
static const int g_maxClicks = 3;
static const float g_colors[][3] = {
    {1.0f,0.0f,0.0f}, {0.0f,1.0f,0.0f}, {0.0f,0.0f,1.0f}
};
static const int g_nColors = 3;

static int g_winImage = -1;
static vector<int> g_winTS;

static vector<vector<unsigned char>> g_rgbTextures;
static GLuint g_texId = 0;
static bool g_texAllocated = false;
static int g_dispW = 800, g_dispH = 600;

#define MAX_TS_WINDOWS 32
static int g_tsWindowBandMap[MAX_TS_WINDOWS];

/* ─────────────── parfor ─────────────────────────────────────────── */

static pthread_mutex_t pf_mtx = PTHREAD_MUTEX_INITIALIZER;
static size_t pf_next_j, pf_end_j;
static void (*pf_eval)(size_t);

static void* pf_worker(void* arg) {
    (void)arg;
    while (1) {
        pthread_mutex_lock(&pf_mtx);
        size_t my_j = pf_next_j++;
        pthread_mutex_unlock(&pf_mtx);
        if (my_j >= pf_end_j) return nullptr;
        pf_eval(my_j);
    }
}

static void parfor(size_t start_j, size_t end_j, void(*eval)(size_t), int cores_use = 0) {
    if (end_j <= start_j) return;
    pf_eval = eval; pf_end_j = end_j; pf_next_j = start_j;
    int ca = sysconf(_SC_NPROCESSORS_ONLN);
    int nc = (cores_use > 0) ? min(cores_use, ca) : ca;
    nc = min(nc, (int)(end_j - start_j));
    if (nc < 1) nc = 1;
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    vector<pthread_t> threads(nc);
    size_t j;
    for0(j, (size_t)nc) pthread_create(&threads[j], &attr, pf_worker, (void*)j);
    for0(j, (size_t)nc) pthread_join(threads[j], nullptr);
    pthread_attr_destroy(&attr);
}

/* ─────────────── timestamp extraction ───────────────────────────── */

static string extract_timestamp(const string& fn) {
    for (size_t i = 0; i + 15 <= fn.size(); i++) {
        if (fn[i + 8] != 'T') continue;
        bool ok = true; size_t k;
        for0(k, (size_t)8) if (!isdigit(fn[i + k])) { ok = false; break; }
        if (!ok) continue;
        for (k = 9; k < 15; k++) if (!isdigit(fn[i + k])) { ok = false; break; }
        if (ok) return fn.substr(i, 15);
    }
    return "";
}

/* ─────────────── .restore_ persistence ──────────────────────────── */

static string restore_prefix() { return g_dir + "/.restore_"; }

static void save_duplicates(const set<size_t>& dups, const vector<string>& fnames) {
    string fn = restore_prefix() + "duplicates";
    ofstream f(fn); for (auto idx : dups) f << fnames[idx] << "\n"; f.close();
}

static set<string> load_duplicates() {
    set<string> r; string fn = restore_prefix() + "duplicates";
    ifstream f(fn); if (!f.is_open()) return r; string line;
    while (getline(f, line)) { while (!line.empty() && isspace(line.back())) line.pop_back(); if (!line.empty()) r.insert(line); }
    return r;
}

struct StretchLimits { float vmin[3], vmax[3]; };

static string stretch_restore_fn(const string& filename) { return restore_prefix() + "stretch_" + filename; }

static bool load_stretch(const string& fn, StretchLimits& sl) {
    FILE* f = fopen(stretch_restore_fn(fn).c_str(), "rb");
    if (!f) return false; size_t nr = fread(&sl, sizeof(sl), 1, f); fclose(f); return nr == 1;
}

static void save_stretch(const string& fn, const StretchLimits& sl) {
    FILE* f = fopen(stretch_restore_fn(fn).c_str(), "wb");
    if (f) { fwrite(&sl, sizeof(sl), 1, f); fclose(f); }
}

/* ─────────── multilook: cache file per image ────────────────────── */

static string ml_restore_fn(const string& filename, int M) {
    char buf[32]; snprintf(buf, sizeof(buf), "_M%d", M);
    return restore_prefix() + "multilook" + buf + "_" + filename;
}

static bool load_ml_cache(const string& filename, int M, float* dst, size_t nfloats) {
    string fn = ml_restore_fn(filename, M);
    FILE* f = fopen(fn.c_str(), "rb");
    if (!f) return false;
    size_t nr = fread(dst, sizeof(float), nfloats, f);
    fclose(f);
    return (nr == nfloats);
}

static void save_ml_cache(const string& filename, int M, const float* src, size_t nfloats) {
    string fn = ml_restore_fn(filename, M);
    FILE* f = fopen(fn.c_str(), "wb");
    if (f) { fwrite(src, sizeof(float), nfloats, f); fclose(f); }
}

/* ─────────── multilook computation for one image ────────────────── */
/*
 * For each band, for each M×M non-overlapping box, compute the mean
 * of finite values.  If all values in a box are NaN, result is NaN.
 * Edge boxes (right/bottom) that extend past the image boundary use
 * only the available pixels.
 *
 * Output layout: band-sequential, mlH rows × mlW cols per band.
 */
static void multilook_one(size_t img_idx) {
    const BandImage& img = g_images[img_idx];
    int M  = g_mlFactor;
    int mw = g_mlW, mh = g_mlH;
    size_t npix = (size_t)mw * mh;
    size_t nfloats = npix * img.nBands;
    float* ml = g_mlData[img_idx];

    /* try cache first */
    if (load_ml_cache(img.filename, M, ml, nfloats)) return;

    size_t b;
    for0(b, img.nBands) {
        const float* src = img.band(b);
        float* dst = ml + b * npix;
        for (int my = 0; my < mh; my++) {
            for (int mx = 0; mx < mw; mx++) {
                int y0 = my * M, x0 = mx * M;
                int y1 = min(y0 + M, (int)img.height);
                int x1 = min(x0 + M, (int)img.width);
                double sum = 0; int cnt = 0;
                for (int r = y0; r < y1; r++) {
                    size_t roff = (size_t)r * img.width;
                    for (int c = x0; c < x1; c++) {
                        float v = src[roff + c];
                        if (isfinite(v)) { sum += v; cnt++; }
                    }
                }
                dst[my * mw + mx] = (cnt > 0) ? (float)(sum / cnt) : NAN;
            }
        }
    }
    save_ml_cache(img.filename, M, ml, nfloats);
}

static void pf_multilook(size_t idx) { multilook_one(idx); }

/* ─────────── histogram stretch (operates on multilooked data) ──── */

static void histStretchToU8(const float* src, size_t n, unsigned char* dst,
                            float pct, float& out_vmin, float& out_vmax) {
    vector<float> vals; vals.reserve(n);
    size_t i;
    for0(i, n) if (isfinite(src[i])) vals.push_back(src[i]);
    if (vals.empty()) { memset(dst, 0, n); out_vmin = out_vmax = 0; return; }
    float frac = pct / 100.0f;
    int lo = (int)(vals.size() * frac), hi = (int)(vals.size() * (1.0f - frac));
    if (lo < 0) lo = 0; if (hi >= (int)vals.size()) hi = (int)vals.size() - 1; if (hi <= lo) hi = lo + 1;
    nth_element(vals.begin(), vals.begin() + lo, vals.end()); float vmin = vals[lo];
    nth_element(vals.begin(), vals.begin() + hi, vals.end()); float vmax = vals[hi];
    float range = vmax - vmin; if (range < 1e-12f) range = 1.0f;
    float scale = 255.0f / range;
    for0(i, n) { float v = (src[i] - vmin) * scale; dst[i] = (unsigned char)max(0.0f, min(255.0f, v)); }
    out_vmin = vmin; out_vmax = vmax;
}

/* ─────────── precompute RGB textures from multilooked data ──────── */

/* global averaged stretch limits (computed once, shared across all images) */
static StretchLimits g_avgStretch;
static bool g_avgStretchReady = false;

static void computeAveragedStretch() {
    /* check if cached */
    string fn = restore_prefix() + "stretch_averaged";
    FILE* cf = fopen(fn.c_str(), "rb");
    if (cf) {
        if (fread(&g_avgStretch, sizeof(StretchLimits), 1, cf) == 1) {
            fclose(cf);
            g_avgStretchReady = true;
            printf("loaded cached averaged stretch limits\n");
            return;
        }
        fclose(cf);
    }

    /* compute per-image stretch limits, then average them */
    size_t nImg = g_images.size();
    size_t n = (size_t)g_mlW * g_mlH;
    double sumMin[3] = {0,0,0}, sumMax[3] = {0,0,0};

    for (size_t img = 0; img < nImg; img++) {
        size_t rgbCount = min(g_images[img].nBands, (size_t)3);
        float* ml = g_mlData[img];
        for (int c = 0; c < 3; c++) {
            size_t srcBand = min((size_t)c, rgbCount - 1);
            const float* bandPtr = ml + srcBand * n;

            vector<float> vals;
            vals.reserve(n);
            size_t k;
            for0(k, n) if (isfinite(bandPtr[k])) vals.push_back(bandPtr[k]);
            if (vals.empty()) continue;

            float frac = 1.0f / 100.0f;
            int lo = (int)(vals.size() * frac);
            int hi = (int)(vals.size() * (1.0f - frac));
            if (lo < 0) lo = 0;
            if (hi >= (int)vals.size()) hi = (int)vals.size() - 1;
            if (hi <= lo) hi = lo + 1;

            nth_element(vals.begin(), vals.begin() + lo, vals.end());
            float vmin = vals[lo];
            nth_element(vals.begin(), vals.begin() + hi, vals.end());
            float vmax = vals[hi];

            sumMin[c] += vmin;
            sumMax[c] += vmax;
        }
    }
    for (int c = 0; c < 3; c++) {
        g_avgStretch.vmin[c] = (float)(sumMin[c] / nImg);
        g_avgStretch.vmax[c] = (float)(sumMax[c] / nImg);
    }
    g_avgStretchReady = true;

    /* save to cache */
    cf = fopen(fn.c_str(), "wb");
    if (cf) { fwrite(&g_avgStretch, sizeof(StretchLimits), 1, cf); fclose(cf); }
    printf("computed averaged stretch: R[%.1f,%.1f] G[%.1f,%.1f] B[%.1f,%.1f]\n",
           g_avgStretch.vmin[0], g_avgStretch.vmax[0],
           g_avgStretch.vmin[1], g_avgStretch.vmax[1],
           g_avgStretch.vmin[2], g_avgStretch.vmax[2]);
}

static void buildOneRGB(size_t idx) {
    size_t n = (size_t)g_mlW * g_mlH;
    auto& tex = g_rgbTextures[idx];
    tex.resize(n * 3);

    const BandImage& img = g_images[idx];
    size_t rgbCount = min(img.nBands, (size_t)3);
    float* ml = g_mlData[idx];
    size_t c, i;

    for0(c, (size_t)3) {
        size_t srcBand = min(c, rgbCount - 1);
        const float* bandPtr = ml + srcBand * n;
        float range = g_avgStretch.vmax[c] - g_avgStretch.vmin[c];
        if (range < 1e-12f) range = 1.0f;
        float scale = 255.0f / range;
        for0(i, n) {
            if (!isfinite(bandPtr[i])) { tex[i * 3 + c] = 0; continue; } /* NaN → black */
            float v = (bandPtr[i] - g_avgStretch.vmin[c]) * scale;
            tex[i * 3 + c] = (unsigned char)max(0.0f, min(255.0f, v));
        }
    }
}

static void pf_buildRGB(size_t idx) { buildOneRGB(idx); }

static void precomputeAllRGB() {
    size_t nImg = g_images.size();
    g_rgbTextures.resize(nImg);
    computeAveragedStretch();
    printf("precomputing %zu RGB textures (from multilook, averaged stretch)...\n", nImg);
    parfor(0, nImg, pf_buildRGB);
    printf("RGB precompute done\n");
}

/* ─────────── texture upload (multilooked dimensions) ────────────── */

static void uploadTexture() {
    glutSetWindow(g_winImage);
    if (g_texId == 0) glGenTextures(1, &g_texId);
    glBindTexture(GL_TEXTURE_2D, g_texId);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    const auto& tex = g_rgbTextures[g_curIdx];
    if (!g_texAllocated) {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, g_mlW, g_mlH, 0, GL_RGB, GL_UNSIGNED_BYTE, tex.data());
        g_texAllocated = true;
    } else {
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, g_mlW, g_mlH, GL_RGB, GL_UNSIGNED_BYTE, tex.data());
    }
}

/* ─────────── time-series: mean/min/max over M×M box ─────────────── */

static void computeTS(int clickIdx, int bandIdx, vector<TSPoint>& out) {
    out.resize(g_images.size());
    /* click coords are in multilooked image space; map back to full-res box */
    int mlx = g_clicks[clickIdx].x;
    int mly = g_clicks[clickIdx].y;
    int M   = g_mlFactor;
    int fy0 = mly * M, fx0 = mlx * M;
    int fy1 = min(fy0 + M, g_fullH);
    int fx1 = min(fx0 + M, g_fullW);
    size_t t;
    for0(t, g_images.size()) {
        const float* bdata = g_images[t].band(bandIdx);
        double sum = 0; float lo = FLT_MAX, hi = -FLT_MAX; int cnt = 0;
        for (int r = fy0; r < fy1; r++) {
            size_t roff = (size_t)r * g_images[t].width;
            for (int c = fx0; c < fx1; c++) {
                float v = bdata[roff + c];
                if (isfinite(v)) { sum += v; if (v < lo) lo = v; if (v > hi) hi = v; cnt++; }
            }
        }
        if (cnt > 0) out[t] = { (float)(sum / cnt), lo, hi };
        else         out[t] = { 0.0f, 0.0f, 0.0f };
    }
}

static void computeNBR_TS(int clickIdx, vector<TSPoint>& out) {
    out.resize(g_images.size());
    int mlx = g_clicks[clickIdx].x, mly = g_clicks[clickIdx].y;
    int M = g_mlFactor;
    int fy0 = mly * M, fx0 = mlx * M;
    int fy1 = min(fy0 + M, g_fullH), fx1 = min(fx0 + M, g_fullW);
    size_t t;
    for0(t, g_images.size()) {
        const float* b08 = g_images[t].band(g_b08_idx);
        const float* b12 = g_images[t].band(g_b12_idx);
        double sum = 0; float lo = FLT_MAX, hi = -FLT_MAX; int cnt = 0;
        for (int r = fy0; r < fy1; r++) {
            size_t roff = (size_t)r * g_images[t].width;
            for (int c = fx0; c < fx1; c++) {
                float v8 = b08[roff + c], v12 = b12[roff + c];
                if (isfinite(v8) && isfinite(v12)) {
                    float d = v8 + v12;
                    float nbr = (fabsf(d) > 1e-12f) ? (v8 - v12) / d : 0.0f;
                    sum += nbr; if (nbr < lo) lo = nbr; if (nbr > hi) hi = nbr; cnt++;
                }
            }
        }
        if (cnt > 0) out[t] = { (float)(sum / cnt), lo, hi };
        else         out[t] = { 0.0f, 0.0f, 0.0f };
    }
}

/* ─────────── GLUT display: image window ─────────────────────────── */

static void imgQuad(float& x0, float& y0, float& x1, float& y1) {
    float wa = (float)g_dispW / g_dispH, ia = (float)g_mlW / g_mlH;
    if (ia > wa) { x0 = -1; x1 = 1; float h = 2.0f * wa / ia; y0 = -h/2; y1 = h/2; }
    else { y0 = -1; y1 = 1; float w = 2.0f * ia / wa; x0 = -w/2; x1 = w/2; }
}

static void displayImage() {
    glutSetWindow(g_winImage);
    glClearColor(0.12f, 0.12f, 0.14f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, g_texId);
    glColor3f(1, 1, 1);
    float x0, y0, x1, y1; imgQuad(x0, y0, x1, y1);
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0); glVertex2f(x0, y1);
    glTexCoord2f(1, 0); glVertex2f(x1, y1);
    glTexCoord2f(1, 1); glVertex2f(x1, y0);
    glTexCoord2f(0, 1); glVertex2f(x0, y0);
    glEnd();
    glDisable(GL_TEXTURE_2D);

    /* draw sampling squares (1 pixel in multilooked space) */
    size_t i;
    for0(i, g_clicks.size()) {
        const float* col = g_colors[i % g_nColors];
        glColor3fv(col); glLineWidth(2.0f);
        float px0 = x0 + (x1 - x0) * (float)g_clicks[i].x / g_mlW;
        float py0 = y1 - (y1 - y0) * (float)g_clicks[i].y / g_mlH;
        float px1 = x0 + (x1 - x0) * (float)(g_clicks[i].x + 1) / g_mlW;
        float py1 = y1 - (y1 - y0) * (float)(g_clicks[i].y + 1) / g_mlH;
        glBegin(GL_LINE_LOOP);
        glVertex2f(px0, py0); glVertex2f(px1, py0);
        glVertex2f(px1, py1); glVertex2f(px0, py1);
        glEnd();
    }

    char title[512];
    snprintf(title, sizeof(title), "[%d/%d] %s  (M=%d, %dx%d)",
             g_curIdx + 1, (int)g_filenames.size(), g_filenames[g_curIdx].c_str(),
             g_mlFactor, g_mlW, g_mlH);
    glutSetWindowTitle(title);
    glutSwapBuffers();
}

/* ─────────── generic TS drawing (mean solid, min/max dashed) ───── */

/*
 * Robust LOWESS (locally weighted scatterplot smoothing):
 * Linear local regression with bisquare robust reweighting.
 * At each evaluation point, fits a weighted linear regression using
 * points within the window, then iteratively downweights outliers.
 *
 * x, y: input data (size n)
 * eval_x: x-values at which to evaluate the trend (size n_eval)
 * out_y: output smoothed values (size n_eval)
 * halfwin: number of points on each side of the evaluation point
 * n_robust: number of robustness iterations
 */
static void lowess_eval(const vector<float>& x, const vector<float>& y,
                        const vector<float>& eval_x, vector<float>& out_y,
                        int halfwin = 10, int n_robust = 3) {
    int n = (int)x.size();
    int ne = (int)eval_x.size();
    out_y.resize(ne);
    if (n < 2) { for (int i = 0; i < ne; i++) out_y[i] = (n == 1) ? y[0] : 0; return; }

    vector<float> weights(n, 1.0f);

    for (int robiter = 0; robiter <= n_robust; robiter++) {
        for (int ei = 0; ei < ne; ei++) {
            float xc = eval_x[ei];

            /* find window: nearest points within halfwin index range */
            /* map xc to nearest index in x */
            int center = 0;
            float bestd = fabsf(x[0] - xc);
            for (int j = 1; j < n; j++) {
                float d = fabsf(x[j] - xc);
                if (d < bestd) { bestd = d; center = j; }
            }
            int i0 = max(0, center - halfwin);
            int i1 = min(n - 1, center + halfwin);

            /* compute max distance for bisquare kernel */
            float maxdist = 0;
            for (int j = i0; j <= i1; j++) {
                float d = fabsf(x[j] - xc);
                if (d > maxdist) maxdist = d;
            }
            if (maxdist < 1e-12f) maxdist = 1.0f;

            /* weighted linear regression: y = a + b*(x - xc) */
            double sw = 0, swx = 0, swy = 0, swxx = 0, swxy = 0;
            for (int j = i0; j <= i1; j++) {
                float u = fabsf(x[j] - xc) / maxdist;
                /* bisquare kernel: (1 - u^2)^2 for u < 1 */
                float kern = (u < 1.0f) ? (1.0f - u * u) * (1.0f - u * u) : 0.0f;
                float w = kern * weights[j];
                float dx = x[j] - xc;
                sw   += w;
                swx  += w * dx;
                swy  += w * y[j];
                swxx += w * dx * dx;
                swxy += w * dx * y[j];
            }
            if (sw < 1e-12) { out_y[ei] = y[center]; continue; }
            double det = sw * swxx - swx * swx;
            double a, b_coeff;
            if (fabs(det) < 1e-20) {
                a = swy / sw; b_coeff = 0;
            } else {
                a = (swxx * swy - swx * swxy) / det;
                b_coeff = (sw * swxy - swx * swy) / det;
            }
            out_y[ei] = (float)a; /* evaluated at dx=0 by construction */
        }

        /* compute residuals and update weights for robustness */
        if (robiter < n_robust) {
            /* recompute fitted values at all data points for residuals */
            vector<float> fitted(n);
            for (int j = 0; j < n; j++) {
                /* quick: find nearest eval point */
                int best_ei = 0;
                float bd = fabsf(eval_x[0] - x[j]);
                for (int ei = 1; ei < ne; ei++) {
                    float d = fabsf(eval_x[ei] - x[j]);
                    if (d < bd) { bd = d; best_ei = ei; }
                }
                /* linear interp between two nearest eval points */
                fitted[j] = out_y[best_ei];
            }
            vector<float> resid(n);
            for (int j = 0; j < n; j++) resid[j] = fabsf(y[j] - fitted[j]);
            /* median absolute residual */
            vector<float> sorted_r(resid);
            nth_element(sorted_r.begin(), sorted_r.begin() + n/2, sorted_r.end());
            float med = sorted_r[n/2];
            float s = 6.0f * med; /* scale factor */
            if (s < 1e-12f) s = 1.0f;
            for (int j = 0; j < n; j++) {
                float u = resid[j] / s;
                weights[j] = (u < 1.0f) ? (1.0f - u * u) * (1.0f - u * u) : 0.0f;
            }
        }
    }
}

static void drawTSGeneric(const char* label,
                          function<void(int, vector<TSPoint>&)> computeFn) {
    glClearColor(0.95f, 0.95f, 0.93f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glutSetWindowTitle(label);
    int nT = (int)g_images.size();
    if (g_clicks.empty() || nT < 2) { glutSwapBuffers(); return; }

    float globalMin = FLT_MAX, globalMax = -FLT_MAX;
    vector<vector<TSPoint>> allTS(g_clicks.size());
    size_t c; int t;
    for0(c, g_clicks.size()) {
        computeFn((int)c, allTS[c]);
        for0(t, nT) {
            if (allTS[c][t].mn < globalMin) globalMin = allTS[c][t].mn;
            if (allTS[c][t].mx > globalMax) globalMax = allTS[c][t].mx;
        }
    }
    float yRange = globalMax - globalMin;
    if (yRange < 1e-12f) yRange = 1.0f;

    float ml = 0.12f, mr = 0.05f, mb = 0.10f, mt = 0.08f;
    float pw = 1.0f - ml - mr, ph = 1.0f - mb - mt;

    glColor3f(0.3f, 0.3f, 0.3f); glLineWidth(1.0f);
    glBegin(GL_LINE_LOOP);
    glVertex2f(ml, mb); glVertex2f(ml + pw, mb);
    glVertex2f(ml + pw, mb + ph); glVertex2f(ml, mb + ph);
    glEnd();

    { float xn = ml + pw * (float)g_curIdx / (nT - 1);
      glColor3f(0.6f, 0.6f, 0.6f); glEnable(GL_LINE_STIPPLE); glLineStipple(1, 0x00FF);
      glBegin(GL_LINES); glVertex2f(xn, mb); glVertex2f(xn, mb + ph); glEnd();
      glDisable(GL_LINE_STIPPLE); }

    for0(c, g_clicks.size()) {
        const float* col = g_colors[c % g_nColors];
        const auto& ts = allTS[c];

        /* mean (solid, thin) */
        glColor3fv(col); glLineWidth(1.0f);
        glBegin(GL_LINE_STRIP);
        for0(t, nT) { float xn = ml + pw * (float)t / (nT-1);
            float yn = mb + ph * (ts[t].mean - globalMin) / yRange; glVertex2f(xn, yn); }
        glEnd();

        /* max (dashed) */
        glLineWidth(1.0f); glEnable(GL_LINE_STIPPLE); glLineStipple(2, 0xAAAA);
        glBegin(GL_LINE_STRIP);
        for0(t, nT) { float xn = ml + pw * (float)t / (nT-1);
            float yn = mb + ph * (ts[t].mx - globalMin) / yRange; glVertex2f(xn, yn); }
        glEnd();

        /* min (dotted) */
        glLineStipple(1, 0x3333);
        glBegin(GL_LINE_STRIP);
        for0(t, nT) { float xn = ml + pw * (float)t / (nT-1);
            float yn = mb + ph * (ts[t].mn - globalMin) / yRange; glVertex2f(xn, yn); }
        glEnd();
        glDisable(GL_LINE_STIPPLE);

        /* LOWESS trend line: evaluate every 15 points, window half-width 10 */
        if (nT >= 15) {
            /* build x/y arrays from the mean time series */
            vector<float> tx(nT), ty(nT);
            for0(t, nT) { tx[t] = (float)t; ty[t] = ts[t].mean; }

            /* evaluation points: every 15th index, starting at index 14 (15th point) */
            vector<float> eval_x;
            for (int e = 14; e < nT; e += 15) eval_x.push_back((float)e);
            /* always include last point for a complete curve */
            if (eval_x.empty() || eval_x.back() != (float)(nT - 1))
                eval_x.push_back((float)(nT - 1));

            vector<float> eval_y;
            lowess_eval(tx, ty, eval_x, eval_y, 10, 3);

            /* draw trend as solid line, same color, thicker */
            glColor3fv(col); glLineWidth(3.0f);
            glBegin(GL_LINE_STRIP);
            for (size_t e = 0; e < eval_x.size(); e++) {
                float xn = ml + pw * eval_x[e] / (nT - 1);
                float yn = mb + ph * (eval_y[e] - globalMin) / yRange;
                glVertex2f(xn, yn);
            }
            glEnd();
            glLineWidth(1.0f);
        }
    }

    glColor3f(0.2f, 0.2f, 0.2f); char buf[64]; int yi;
    for0(yi, 5) { float frac = (float)yi / 4.0f; float val = globalMin + yRange * frac;
        float yn = mb + ph * frac; snprintf(buf, sizeof(buf), "%.3g", val);
        glRasterPos2f(0.005f, yn - 0.01f);
        for (char* p = buf; *p; p++) glutBitmapCharacter(GLUT_BITMAP_HELVETICA_10, *p); }

    glRasterPos2f(ml, 0.01f);
    { const char* p; for (p = g_filenames.front().c_str(); *p; p++) glutBitmapCharacter(GLUT_BITMAP_HELVETICA_10, *p); }
    float lx = ml + pw - 0.15f; if (lx < ml) lx = ml;
    glRasterPos2f(lx, 0.01f);
    { const char* p; for (p = g_filenames.back().c_str(); *p; p++) glutBitmapCharacter(GLUT_BITMAP_HELVETICA_10, *p); }
    glutSwapBuffers();
}

/* helper: extract yyyymmdd from current image timestamp */
static string curDateStr() {
    if (g_curIdx >= 0 && g_curIdx < (int)g_images.size()) {
        const string& ts = g_images[g_curIdx].timestamp;
        if (ts.size() >= 8) return ts.substr(0, 8);
    }
    return "--------";
}

static void drawTSForBand(int bandIdx) {
    char title[256];
    string ds = curDateStr();
    if (bandIdx < (int)g_bandNames.size())
        snprintf(title, sizeof(title), "%s (band %d) — %s", g_bandNames[bandIdx].c_str(), bandIdx + 1, ds.c_str());
    else snprintf(title, sizeof(title), "Band %d — %s", bandIdx + 1, ds.c_str());
    drawTSGeneric(title, [bandIdx](int ci, vector<TSPoint>& out) { computeTS(ci, bandIdx, out); });
}

static void displayNBR() {
    char title[256];
    snprintf(title, sizeof(title), "NBR = (B08-B12)/(B08+B12) — %s", curDateStr().c_str());
    drawTSGeneric(title, [](int ci, vector<TSPoint>& out) { computeNBR_TS(ci, out); });
}

static void displayTS_dispatch() {
    int win = glutGetWindow();
    if (g_hasNBR && win == g_winNBR) { displayNBR(); return; }
    size_t i; for0(i, g_winTS.size()) if (g_winTS[i] == win) { drawTSForBand(g_tsWindowBandMap[i]); return; }
}

/* ─────────── click on TS window → navigate to nearest date ──────── */

static void refreshAll(); /* forward declaration */

static void mouseTS(int button, int state, int mx, int /*my*/) {
    if (state != GLUT_DOWN || button != GLUT_LEFT_BUTTON) return;
    int nT = (int)g_images.size();
    if (nT < 2) return;

    /* get window width to map mx → fractional x position */
    int winW = glutGet(GLUT_WINDOW_WIDTH);
    float ml = 0.12f, mr = 0.05f;
    float pw = 1.0f - ml - mr;

    /* convert mx (pixels) to NDC [0,1] fraction within the plot area */
    float frac = ((float)mx / winW - ml) / pw;
    if (frac < 0.0f) frac = 0.0f;
    if (frac > 1.0f) frac = 1.0f;

    int idx = (int)roundf(frac * (nT - 1));
    idx = max(0, min(idx, nT - 1));

    if (idx != g_curIdx) {
        g_curIdx = idx;
        printf("TS click -> [%d] %s\n", g_curIdx, g_filenames[g_curIdx].c_str());
        refreshAll();
    }
}

/* ─────────── refresh ────────────────────────────────────────────── */

static void refreshAll() {
    uploadTexture();
    glutSetWindow(g_winImage); glutPostRedisplay();
    size_t i; for0(i, g_winTS.size()) { glutSetWindow(g_winTS[i]); glutPostRedisplay(); }
    if (g_hasNBR && g_winNBR > 0) { glutSetWindow(g_winNBR); glutPostRedisplay(); }
}

/* ─────────── clean exit ─────────────────────────────────────────── */

static void cleanExit() {
    printf("clean exit\n");
    size_t i; for0(i, g_images.size()) if (g_images[i].data) { free(g_images[i].data); g_images[i].data = nullptr; }
    for0(i, g_mlData.size()) if (g_mlData[i]) { free(g_mlData[i]); g_mlData[i] = nullptr; }
    exit(0);
}

/* ─────────── input callbacks ────────────────────────────────────── */

static void keyboardAll(unsigned char key, int, int) {
    if (key == 27) exit(0);  /* Escape: hard exit, no cleanup */
    if (key == 'q') exit(0);
    if (key == 'c' || key == 'C') { g_clicks.clear(); printf("cleared\n"); refreshAll(); }
}

static void specialAll(int key, int, int) {
    if (key == GLUT_KEY_RIGHT) {
        g_curIdx = (g_curIdx + 1) % (int)g_images.size();
        printf("-> [%d] %s\n", g_curIdx, g_filenames[g_curIdx].c_str()); refreshAll();
    } else if (key == GLUT_KEY_LEFT) {
        g_curIdx = (g_curIdx - 1 + (int)g_images.size()) % (int)g_images.size();
        printf("-> [%d] %s\n", g_curIdx, g_filenames[g_curIdx].c_str()); refreshAll();
    }
}

static void mouseImage(int button, int state, int mx, int my) {
    if (state != GLUT_DOWN) return;
    if (button == GLUT_RIGHT_BUTTON) { g_clicks.clear(); printf("cleared\n"); refreshAll(); return; }
    if (button != GLUT_LEFT_BUTTON) return;

    float x0, y0, x1, y1; imgQuad(x0, y0, x1, y1);
    float wx0 = (x0 + 1.0f) / 2.0f * g_dispW, wx1 = (x1 + 1.0f) / 2.0f * g_dispW;
    float wy0 = (1.0f - y1) / 2.0f * g_dispH, wy1 = (1.0f - y0) / 2.0f * g_dispH;
    float fx = (float)mx, fy = (float)my;
    if (fx < wx0 || fx > wx1 || fy < wy0 || fy > wy1) return;

    int px = (int)((fx - wx0) / (wx1 - wx0) * g_mlW);
    int py = (int)((fy - wy0) / (wy1 - wy0) * g_mlH);
    px = max(0, min(px, g_mlW - 1));
    py = max(0, min(py, g_mlH - 1));

    g_clicks.push_back({px, py});
    while ((int)g_clicks.size() > g_maxClicks) g_clicks.erase(g_clicks.begin());
    printf("click at ml(%d,%d) -> full-res box [%d:%d, %d:%d]\n",
           px, py, py * g_mlFactor, min(py * g_mlFactor + g_mlFactor, g_fullH),
           px * g_mlFactor, min(px * g_mlFactor + g_mlFactor, g_fullW));
    refreshAll();
}

static void reshapeImage(int w, int h) {
    g_dispW = w; g_dispH = h; glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION); glLoadIdentity(); gluOrtho2D(-1, 1, -1, 1);
    glMatrixMode(GL_MODELVIEW); glLoadIdentity();
}

static void reshapeTS(int w, int h) {
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION); glLoadIdentity(); gluOrtho2D(0, 1, 0, 1);
    glMatrixMode(GL_MODELVIEW); glLoadIdentity();
}

/* ─────────── file loading ───────────────────────────────────────── */

struct LoadSlot { string path; size_t nrow, ncol, nband; vector<string> bandNames; float* data; int ok; };
static vector<LoadSlot> g_loadSlots;
static void pf_loadFile(size_t idx) {
    LoadSlot& s = g_loadSlots[idx];
    FILE* f = fopen(s.path.c_str(), "rb"); if (!f) { s.ok = 0; return; }
    size_t nf = s.nrow * s.ncol * s.nband;
    s.data = (float*)malloc(nf * sizeof(float));
    if (!s.data) { fclose(f); s.ok = 0; return; }
    size_t nr = fread(s.data, sizeof(float), nf, f); fclose(f);
    s.ok = (nr == nf) ? 1 : 0; if (!s.ok) { free(s.data); s.data = nullptr; }
}

static vector<int> g_dupFlag;
static void pf_checkDup(size_t idx) {
    const BandImage& a = g_images[idx], &b = g_images[idx + 1];
    size_t n = a.width * a.height * a.nBands;
    if (memcmp(a.data, b.data, n * sizeof(float)) == 0) g_dupFlag[idx + 1] = 1;
}

/* ─────────── main ───────────────────────────────────────────────── */

int main(int argc, char** argv) {
    g_dir = "."; string ext = "bin"; int startIdx = 0; size_t i;

    for (int a = 1; a < argc; a++) {
        if (!strcmp(argv[a], "-d") && a+1 < argc) g_dir = argv[++a];
        else if (!strcmp(argv[a], "-w") && a+1 < argc) g_squareWidth = atoi(argv[++a]);
        else if (!strcmp(argv[a], "-i") && a+1 < argc) startIdx = atoi(argv[++a]);
        else if (!strcmp(argv[a], "-e") && a+1 < argc) ext = argv[++a];
        else if (!strcmp(argv[a], "-h") || !strcmp(argv[a], "--help")) {
            printf("Usage: %s [-d dir] [-w width] [-i index] [-e ext]\n", argv[0]); return 0;
        }
    }

    /* ═══════ 1. SCAN DIRECTORY ═══════════════════════════════════── */
    bool has_bin = false, has_tif = false;
    vector<string> allNames;
    { DIR* dp = opendir(g_dir.c_str());
      if (!dp) { fprintf(stderr, "cannot open %s\n", g_dir.c_str()); return 1; }
      struct dirent* ent;
      while ((ent = readdir(dp)) != nullptr) {
          string fname(ent->d_name); size_t len = fname.size();
          if (len > 4 && fname.compare(len-4,4,".bin")==0 && fname.compare(0,9,".restore_")!=0) has_bin = true;
          if (len > 4 && fname.compare(len-4,4,".tif")==0 && fname.compare(0,9,".restore_")!=0) has_tif = true;
          string dotExt = string(".") + ext;
          if (len > dotExt.size() && fname.compare(len-dotExt.size(), dotExt.size(), dotExt)==0 &&
              fname.compare(0,9,".restore_")!=0) allNames.push_back(fname);
      } closedir(dp); }
    if (has_bin && has_tif) err("directory has both .bin and .tif — separate them");
    if (allNames.empty()) { fprintf(stderr, "no .%s files in %s\n", ext.c_str(), g_dir.c_str()); return 1; }

    /* ═══════ 2. SORT BY TIMESTAMP ═══════════════════════════════── */
    map<string, string> fname_to_ts;
    for0(i, allNames.size()) fname_to_ts[allNames[i]] = extract_timestamp(allNames[i]);
    sort(allNames.begin(), allNames.end(), [&](const string& a, const string& b) {
        const string &ta = fname_to_ts[a], &tb = fname_to_ts[b];
        if (ta.empty() && tb.empty()) return a < b;
        if (ta.empty()) return false; if (tb.empty()) return true;
        return (ta != tb) ? ta < tb : a < b;
    });
    printf("found %zu .%s files (sorted by timestamp)\n", allNames.size(), ext.c_str());

    /* ═══════ 3. CACHED DUPLICATES ═══════════════════════════════── */
    set<string> cached_dups = load_duplicates();
    if (!cached_dups.empty()) {
        vector<string> filtered;
        for0(i, allNames.size()) if (cached_dups.count(allNames[i]) == 0) filtered.push_back(allNames[i]);
        allNames = move(filtered);
    }
    g_filenames = allNames; size_t nFiles = g_filenames.size();

    /* ═══════ 4. READ HEADERS & PRE-CHECK ════════════════════════── */
    g_loadSlots.resize(nFiles);
    for0(i, nFiles) {
        string path = g_dir + "/" + g_filenames[i], hfn = hdr_fn(path);
        g_loadSlots[i].path = path; g_loadSlots[i].data = nullptr; g_loadSlots[i].ok = 0;
        hread(hfn, g_loadSlots[i].nrow, g_loadSlots[i].ncol, g_loadSlots[i].nband, g_loadSlots[i].bandNames);
    }
    { map<string, int> dc;
      for0(i, nFiles) { char k[128]; snprintf(k,128,"%zu_%zu_%zu", g_loadSlots[i].nrow, g_loadSlots[i].ncol, g_loadSlots[i].nband); dc[string(k)]++; }
      string mk; int mc = 0; for (auto& kv : dc) if (kv.second > mc) { mc = kv.second; mk = kv.first; }
      size_t mr, mcc, mb; sscanf(mk.c_str(), "%zu_%zu_%zu", &mr, &mcc, &mb);
      size_t eb = mr * mcc * mb * sizeof(float); bool mis = false;
      for0(i, nFiles) { char k[128]; snprintf(k,128,"%zu_%zu_%zu", g_loadSlots[i].nrow, g_loadSlots[i].ncol, g_loadSlots[i].nband);
          if (string(k)!=mk) { fprintf(stderr,"DIM MISMATCH: %s\n", g_filenames[i].c_str()); mis=true; }
          if (fsize(g_loadSlots[i].path) != eb) { fprintf(stderr,"SIZE MISMATCH: %s\n", g_filenames[i].c_str()); mis=true; } }
      if (mis) err("dimension/size mismatches"); }

    /* ═══════ 5. PARALLEL LOAD ═══════════════════════════════════── */
    printf("loading %zu files...\n", nFiles);
    parfor(0, nFiles, pf_loadFile);
    { vector<string> gn;
      for0(i, nFiles) { LoadSlot& s = g_loadSlots[i]; if (!s.ok) continue;
          BandImage img; img.width=s.ncol; img.height=s.nrow; img.nBands=s.nband;
          img.data=s.data; img.bandNames=move(s.bandNames); img.filename=g_filenames[i];
          img.timestamp=fname_to_ts[g_filenames[i]]; g_images.push_back(move(img)); gn.push_back(g_filenames[i]); }
      g_filenames = move(gn); }
    if (g_images.empty()) err("no files loaded");

    /* ═══════ 6. DUPLICATE REMOVAL ═══════════════════════════════── */
    if (g_images.size() >= 2 && cached_dups.empty()) {
        g_dupFlag.assign(g_images.size(), 0);
        parfor(0, g_images.size()-1, pf_checkDup);
        set<size_t> to_rm;
        for (i=1; i < g_images.size(); i++) if (g_dupFlag[i]) to_rm.insert(i);
        if (!to_rm.empty()) { save_duplicates(to_rm, g_filenames);
            vector<size_t> sr(to_rm.rbegin(), to_rm.rend());
            for (auto idx : sr) { if (g_images[idx].data) free(g_images[idx].data);
                g_images.erase(g_images.begin()+idx); g_filenames.erase(g_filenames.begin()+idx); }
            printf("removed %zu duplicates\n", to_rm.size());
        } else { set<size_t> es; save_duplicates(es, g_filenames); }
    }
    if (g_images.empty()) err("no images remain");

    g_fullW  = (int)g_images[0].width;
    g_fullH  = (int)g_images[0].height;
    g_nBands = (int)g_images[0].nBands;
    g_bandNames = g_images[0].bandNames;

    /* ═══════ 7. NBR DETECTION ═══════════════════════════════════── */
    for0(i, g_bandNames.size()) {
        if (g_bandNames[i].find("B08") != string::npos) g_b08_idx = (int)i;
        if (g_bandNames[i].find("B12") != string::npos) g_b12_idx = (int)i;
    }
    g_hasNBR = (g_b08_idx >= 0 && g_b12_idx >= 0);
    if (g_hasNBR) printf("NBR enabled: B08=band %d, B12=band %d\n", g_b08_idx+1, g_b12_idx+1);

    /* ═══════ 8. COMPUTE MULTILOOK FACTOR M ══════════════════════── */
    /*
     * Choose M such that floor(ncol/M) <= screenW/2.
     * We query screen width early via a temporary GLUT init, or use a
     * conservative default.  M = ceil(ncol / (screenW/2)).
     */
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    int screenW = glutGet(GLUT_SCREEN_WIDTH);
    int screenH = glutGet(GLUT_SCREEN_HEIGHT);
    if (screenW < 800) screenW = 1920;
    if (screenH < 600) screenH = 1080;

    int halfW = screenW / 2;
    g_mlFactor = max(1, (int)ceil((double)g_fullW / halfW));
    /* also ensure height fits */
    int mfH = max(1, (int)ceil((double)g_fullH / (screenH - 80)));
    g_mlFactor = max(g_mlFactor, mfH);

    g_mlW = g_fullW / g_mlFactor;
    g_mlH = g_fullH / g_mlFactor;

    printf("multilook: M=%d, full %dx%d -> display %dx%d\n",
           g_mlFactor, g_fullW, g_fullH, g_mlW, g_mlH);

    /* ═══════ 9. PARALLEL MULTILOOK ══════════════════════════════── */
    {
        size_t mlPixels = (size_t)g_mlW * g_mlH;
        g_mlData.resize(g_images.size());
        for0(i, g_images.size()) {
            g_mlData[i] = (float*)malloc(mlPixels * g_images[i].nBands * sizeof(float));
            if (!g_mlData[i]) err("malloc multilook");
        }
        printf("computing multilook for %zu images...\n", g_images.size());
        parfor(0, g_images.size(), pf_multilook);
        printf("multilook done\n");
    }

    /* ═══════ 10. PRECOMPUTE RGB TEXTURES ════════════════════════── */
    precomputeAllRGB();

    if (startIdx < 0) startIdx = (int)g_images.size() + startIdx;
    startIdx = max(0, min(startIdx, (int)g_images.size() - 1));
    g_curIdx = startIdx;

    printf("image: %dx%d full, %dx%d display (M=%d), %d bands, %zu dates\n",
           g_fullW, g_fullH, g_mlW, g_mlH, g_mlFactor, g_nBands, g_images.size());
    printf("controls: left/right=navigate, click=sample (max 3, R/G/B), right-click/c=clear, q/Esc=quit\n");

    /* ═══════ 11. GLUT WINDOWS ═══════════════════════════════════── */
    g_dispW = g_mlW;
    g_dispH = g_mlH;

    glutInitWindowSize(g_dispW, g_dispH);
    glutInitWindowPosition(50, 50);
    g_winImage = glutCreateWindow("Band Viewer");
    glutDisplayFunc(displayImage);
    glutReshapeFunc(reshapeImage);
    glutMouseFunc(mouseImage);
    glutKeyboardFunc(keyboardAll);
    glutSpecialFunc(specialAll);
    uploadTexture();

    /* TS windows: single column to the right, filling screen height */
    int nTSBands = min(g_nBands, MAX_TS_WINDOWS);
    int nTSTotal = nTSBands + (g_hasNBR ? 1 : 0);
    int tsX = 50 + g_dispW + 10;
    int tsW = screenW - tsX - 20; if (tsW < 300) tsW = 300;
    int tsH = max(120, (screenH - 80) / max(nTSTotal, 1));

    for (int b = 0; b < nTSBands; b++) {
        glutInitWindowSize(tsW, tsH);
        glutInitWindowPosition(tsX, 50 + b * tsH);
        char name[128];
        if (b < (int)g_bandNames.size()) snprintf(name, 128, "%s (band %d)", g_bandNames[b].c_str(), b+1);
        else snprintf(name, 128, "Band %d", b+1);
        int win = glutCreateWindow(name);
        g_winTS.push_back(win); g_tsWindowBandMap[b] = b;
        glutDisplayFunc(displayTS_dispatch); glutReshapeFunc(reshapeTS);
        glutMouseFunc(mouseTS);
        glutKeyboardFunc(keyboardAll); glutSpecialFunc(specialAll);
    }
    if (g_hasNBR) {
        glutInitWindowSize(tsW, tsH);
        glutInitWindowPosition(tsX, 50 + nTSBands * tsH);
        g_winNBR = glutCreateWindow("NBR = (B08-B12)/(B08+B12)");
        glutDisplayFunc(displayTS_dispatch); glutReshapeFunc(reshapeTS);
        glutMouseFunc(mouseTS);
        glutKeyboardFunc(keyboardAll); glutSpecialFunc(specialAll);
    }

    glutMainLoop();
    return 0;
}



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
 *   Left-click on image  Add a sampling square; TS plots update
 *   Right-click on image Clear all squares and traces
 *
 * Features:
 *   - Pre-checks: validates all files share same dimensions/bands;
 *     exits if .tif and .bin coexist in the same directory.
 *   - Sorts files by embedded yyyymmddThhmmss timestamp.
 *   - Removes duplicate frames (identical binary content) via parallel
 *     pairwise comparison.
 *   - Computes NBR = (B08 - B12) / (B08 + B12) if both bands present;
 *     plots in an extra TS window.
 *   - Caches histogram stretch limits and duplicate-removal indices in
 *     .restore_* files for fast restart.
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
#include <regex>
#include <set>
#include <map>
#include <functional>

#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/freeglut.h>

using namespace std;

/* ─────────────────── misc-style macros and helpers ───────────────── */

#define for0(i,n) for(i = 0; i < n; i++)

static void err(const string& msg) {
    cerr << "Error: " << msg << endl;
    exit(1);
}

static size_t fsize(const string& fn) {
    FILE* f = fopen(fn.c_str(), "rb");
    if (!f) return 0;
    fseek(f, 0L, SEEK_END);
    size_t sz = ftell(f);
    fclose(f);
    return sz;
}

static bool exists(const string& fn) {
    return fsize(fn) > 0;
}

/* ─────────────────── ENVI header reading (from misc.cpp) ────────── */

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
    bool in_band_names = false;
    string bn_accum;

    while (getline(hf, line)) {
        if (in_band_names) {
            bn_accum += line;
            if (line.find('}') != string::npos) in_band_names = false;
            continue;
        }

        size_t eq = line.find('=');
        if (eq == string::npos) continue;
        string key = line.substr(0, eq);
        string val = line.substr(eq + 1);

        auto trim = [](string& s) {
            while (!s.empty() && isspace(s.front())) s.erase(s.begin());
            while (!s.empty() && isspace(s.back()))  s.pop_back();
        };
        trim(key); trim(val);

        if (key == "samples") ncol  = atoi(val.c_str());
        else if (key == "lines")   nrow  = atoi(val.c_str());
        else if (key == "bands")   nband = atoi(val.c_str());
        else if (key == "band names") {
            bn_accum = val;
            if (val.find('}') == string::npos) in_band_names = true;
        }
    }
    hf.close();

    if (!bn_accum.empty()) {
        size_t i;
        string clean;
        for0(i, bn_accum.size()) {
            char c = bn_accum[i];
            if (c != '{' && c != '}') clean += c;
        }
        istringstream iss(clean);
        string tok;
        while (getline(iss, tok, ',')) {
            while (!tok.empty() && isspace(tok.front())) tok.erase(tok.begin());
            while (!tok.empty() && isspace(tok.back()))  tok.pop_back();
            if (!tok.empty()) band_names.push_back(tok);
        }
    }
}

/* ──────────────────────────── data types ──────────────────────────── */

struct BandImage {
    size_t width  = 0;
    size_t height = 0;
    size_t nBands = 0;
    float* data   = nullptr;
    vector<string> bandNames;
    string filename;
    string timestamp; /* yyyymmddThhmmss extracted from filename */

    const float* band(size_t b) const { return data + b * width * height; }
    float* band(size_t b) { return data + b * width * height; }
};

struct Click { int x, y; };
struct TSPoint { float mean, stddev; };

/* ──────────────────────────── globals ─────────────────────────────── */

static vector<BandImage>  g_images;
static vector<string>     g_filenames;
static string             g_dir;
static int g_curIdx       = 0;
static int g_squareWidth  = 10;
static int g_imgW = 0, g_imgH = 0;
static int g_nBands = 0;
static vector<string> g_bandNames;

/* NBR band indices (-1 if not found) */
static int g_b08_idx = -1, g_b12_idx = -1;
static bool g_hasNBR = false;
static int g_winNBR = -1; /* NBR TS window ID */

static vector<Click> g_clicks;
static const float g_colors[][3] = {
    {0.2f,0.5f,1.0f}, {1.0f,0.3f,0.3f}, {1.0f,1.0f,0.2f},
    {0.1f,0.9f,0.1f}, {0.0f,1.0f,1.0f}, {1.0f,0.4f,1.0f}
};
static const int g_nColors = 6;

static int g_winImage = -1;
static vector<int> g_winTS;

static vector<vector<unsigned char>> g_rgbTextures;
static GLuint g_texId = 0;
static bool   g_texAllocated = false;

static int g_dispW = 800, g_dispH = 600;

#define MAX_TS_WINDOWS 32
static int g_tsWindowBandMap[MAX_TS_WINDOWS];

/* ─────────────── parfor: work-stealing parallel for ─────────────── */

static pthread_mutex_t pf_mtx = PTHREAD_MUTEX_INITIALIZER;
static size_t pf_next_j;
static size_t pf_end_j;
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
    pf_eval   = eval;
    pf_end_j  = end_j;
    pf_next_j = start_j;

    int cores_avail = sysconf(_SC_NPROCESSORS_ONLN);
    int n_cores = (cores_use > 0) ? min(cores_use, cores_avail) : cores_avail;
    n_cores = min(n_cores, (int)(end_j - start_j));
    if (n_cores < 1) n_cores = 1;

    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    vector<pthread_t> threads(n_cores);
    size_t j;
    for0(j, (size_t)n_cores)
        pthread_create(&threads[j], &attr, pf_worker, (void*)j);
    for0(j, (size_t)n_cores)
        pthread_join(threads[j], nullptr);
    pthread_attr_destroy(&attr);
}

/* ─────────────── timestamp extraction and sorting ───────────────── */

/*
 * Extract the first yyyymmddThhmmss pattern from a filename.
 * Sentinel-2 names contain this after the processing level field.
 * Returns empty string if no match.
 */
static string extract_timestamp(const string& fn) {
    /* look for 8 digits + 'T' + 6 digits */
    for (size_t i = 0; i + 15 <= fn.size(); i++) {
        if (fn[i + 8] != 'T') continue;
        bool ok = true;
        size_t k;
        for0(k, (size_t)8) if (!isdigit(fn[i + k])) { ok = false; break; }
        if (!ok) continue;
        for (k = 9; k < 15; k++) if (!isdigit(fn[i + k])) { ok = false; break; }
        if (ok) return fn.substr(i, 15);
    }
    return "";
}

/* ──────────── .restore_ persistence helpers ─────────────────────── */

static string restore_prefix() { return g_dir + "/.restore_"; }

static void save_duplicates(const set<size_t>& dups, const vector<string>& fnames) {
    string fn = restore_prefix() + "duplicates";
    ofstream f(fn);
    for (auto idx : dups) f << fnames[idx] << "\n";
    f.close();
    printf("saved %zu duplicate filenames to %s\n", dups.size(), fn.c_str());
}

static set<string> load_duplicates() {
    set<string> result;
    string fn = restore_prefix() + "duplicates";
    ifstream f(fn);
    if (!f.is_open()) return result;
    string line;
    while (getline(f, line)) {
        while (!line.empty() && isspace(line.back())) line.pop_back();
        if (!line.empty()) result.insert(line);
    }
    f.close();
    return result;
}

/* per-file stretch limits: .restore_stretch_<filename> */
struct StretchLimits { float vmin[3], vmax[3]; };

static string stretch_restore_fn(const string& filename) {
    return restore_prefix() + "stretch_" + filename;
}

static bool load_stretch(const string& filename, StretchLimits& sl) {
    string fn = stretch_restore_fn(filename);
    FILE* f = fopen(fn.c_str(), "rb");
    if (!f) return false;
    size_t nr = fread(&sl, sizeof(StretchLimits), 1, f);
    fclose(f);
    return (nr == 1);
}

static void save_stretch(const string& filename, const StretchLimits& sl) {
    string fn = stretch_restore_fn(filename);
    FILE* f = fopen(fn.c_str(), "wb");
    if (!f) return;
    fwrite(&sl, sizeof(StretchLimits), 1, f);
    fclose(f);
}

/* ──────────────── fast histogram stretch to u8 (O(n)) ───────────── */

static void histStretchToU8(const float* src, size_t n, unsigned char* dst,
                            float pct, float& out_vmin, float& out_vmax) {
    vector<float> vals;
    vals.reserve(n);
    size_t i;
    for0(i, n) {
        if (isfinite(src[i])) vals.push_back(src[i]);
    }
    if (vals.empty()) { memset(dst, 0, n); out_vmin = out_vmax = 0; return; }

    float frac = pct / 100.0f;
    int lo = (int)(vals.size() * frac);
    int hi = (int)(vals.size() * (1.0f - frac));
    if (lo < 0) lo = 0;
    if (hi >= (int)vals.size()) hi = (int)vals.size() - 1;
    if (hi <= lo) hi = lo + 1;

    nth_element(vals.begin(), vals.begin() + lo, vals.end());
    float vmin = vals[lo];
    nth_element(vals.begin(), vals.begin() + hi, vals.end());
    float vmax = vals[hi];

    float range = vmax - vmin;
    if (range < 1e-12f) range = 1.0f;
    float scale = 255.0f / range;

    for0(i, n) {
        float v = (src[i] - vmin) * scale;
        if (v < 0.f)   v = 0.f;
        if (v > 255.f) v = 255.f;
        dst[i] = (unsigned char)v;
    }
    out_vmin = vmin;
    out_vmax = vmax;
}

/* overload without out params for callers that don't need them */
static void histStretchToU8(const float* src, size_t n, unsigned char* dst, float pct = 1.0f) {
    float dummy1, dummy2;
    histStretchToU8(src, n, dst, pct, dummy1, dummy2);
}

/* ────────────── precompute RGB texture for one date ──────────────── */

static void buildOneRGB(size_t idx) {
    const BandImage& img = g_images[idx];
    size_t n = img.width * img.height;
    auto& tex = g_rgbTextures[idx];
    tex.resize(n * 3);

    size_t rgbCount = min(img.nBands, (size_t)3);
    vector<unsigned char> chan(n);
    size_t c, i;

    /* try to load cached stretch limits */
    StretchLimits sl;
    bool cached = load_stretch(img.filename, sl);

    for0(c, (size_t)3) {
        size_t srcBand = min(c, rgbCount - 1);
        if (cached) {
            /* apply cached limits directly */
            float range = sl.vmax[c] - sl.vmin[c];
            if (range < 1e-12f) range = 1.0f;
            float scale = 255.0f / range;
            const float* src = img.band(srcBand);
            for0(i, n) {
                float v = (src[i] - sl.vmin[c]) * scale;
                if (v < 0.f) v = 0.f;
                if (v > 255.f) v = 255.f;
                chan[i] = (unsigned char)v;
            }
        } else {
            histStretchToU8(img.band(srcBand), n, chan.data(), 1.0f,
                            sl.vmin[c], sl.vmax[c]);
        }
        for0(i, n)
            tex[i * 3 + c] = chan[i];
    }

    if (!cached) save_stretch(img.filename, sl);
}

static void pf_buildRGB(size_t idx) { buildOneRGB(idx); }

static void precomputeAllRGB() {
    size_t nImg = g_images.size();
    g_rgbTextures.resize(nImg);
    printf("precomputing %zu RGB textures...\n", nImg);
    parfor(0, nImg, pf_buildRGB);
    printf("RGB precompute done\n");
}

/* ──────────────────────────── texture upload ─────────────────────── */

static void uploadTexture() {
    glutSetWindow(g_winImage);
    if (g_texId == 0) glGenTextures(1, &g_texId);
    glBindTexture(GL_TEXTURE_2D, g_texId);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    const auto& tex = g_rgbTextures[g_curIdx];
    if (!g_texAllocated) {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, g_imgW, g_imgH, 0,
                     GL_RGB, GL_UNSIGNED_BYTE, tex.data());
        g_texAllocated = true;
    } else {
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, g_imgW, g_imgH,
                        GL_RGB, GL_UNSIGNED_BYTE, tex.data());
    }
}

/* ──────────────────────────── time-series stats ──────────────────── */

static void computeTS(int clickIdx, int bandIdx, vector<TSPoint>& out) {
    out.resize(g_images.size());
    int cx = g_clicks[clickIdx].x;
    int cy = g_clicks[clickIdx].y;
    int w  = g_squareWidth;
    size_t t;
    for0(t, g_images.size()) {
        const BandImage& img = g_images[t];
        const float* bdata = img.band(bandIdx);
        double sum = 0, sum2 = 0;
        int cnt = 0;
        int yEnd = min(cy + w, (int)img.height);
        int xEnd = min(cx + w, (int)img.width);
        for (int row = cy; row < yEnd; row++) {
            size_t rowOff = (size_t)row * img.width;
            for (int col = cx; col < xEnd; col++) {
                float v = bdata[rowOff + col];
                if (isfinite(v)) {
                    sum  += v;
                    sum2 += (double)v * v;
                    cnt++;
                }
            }
        }
        if (cnt > 0) {
            float m = (float)(sum / cnt);
            float s = (float)sqrt(max(0.0, sum2 / cnt - (double)m * m));
            out[t] = {m, s};
        } else {
            out[t] = {0.0f, 0.0f};
        }
    }
}

/* NBR = (B08 - B12) / (B08 + B12) */
static void computeNBR_TS(int clickIdx, vector<TSPoint>& out) {
    out.resize(g_images.size());
    int cx = g_clicks[clickIdx].x;
    int cy = g_clicks[clickIdx].y;
    int w  = g_squareWidth;
    size_t t;
    for0(t, g_images.size()) {
        const BandImage& img = g_images[t];
        const float* b08 = img.band(g_b08_idx);
        const float* b12 = img.band(g_b12_idx);
        double sum = 0, sum2 = 0;
        int cnt = 0;
        int yEnd = min(cy + w, (int)img.height);
        int xEnd = min(cx + w, (int)img.width);
        for (int row = cy; row < yEnd; row++) {
            size_t rowOff = (size_t)row * img.width;
            for (int col = cx; col < xEnd; col++) {
                float v8  = b08[rowOff + col];
                float v12 = b12[rowOff + col];
                if (isfinite(v8) && isfinite(v12)) {
                    float denom = v8 + v12;
                    float nbr = (fabsf(denom) > 1e-12f) ? (v8 - v12) / denom : 0.0f;
                    sum  += nbr;
                    sum2 += (double)nbr * nbr;
                    cnt++;
                }
            }
        }
        if (cnt > 0) {
            float m = (float)(sum / cnt);
            float s = (float)sqrt(max(0.0, sum2 / cnt - (double)m * m));
            out[t] = {m, s};
        } else {
            out[t] = {0.0f, 0.0f};
        }
    }
}

/* ──────────────── GLUT display: image window ─────────────────────── */

static void imgQuad(float& x0, float& y0, float& x1, float& y1) {
    float winAspect = (float)g_dispW / g_dispH;
    float imgAspect = (float)g_imgW / g_imgH;
    if (imgAspect > winAspect) {
        x0 = -1.0f; x1 = 1.0f;
        float h = 2.0f * winAspect / imgAspect;
        y0 = -h / 2; y1 = h / 2;
    } else {
        y0 = -1.0f; y1 = 1.0f;
        float w = 2.0f * imgAspect / winAspect;
        x0 = -w / 2; x1 = w / 2;
    }
}

static void displayImage() {
    glutSetWindow(g_winImage);
    glClearColor(0.12f, 0.12f, 0.14f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, g_texId);
    glColor3f(1, 1, 1);

    float x0, y0, x1, y1;
    imgQuad(x0, y0, x1, y1);

    glBegin(GL_QUADS);
    glTexCoord2f(0, 0); glVertex2f(x0, y1);
    glTexCoord2f(1, 0); glVertex2f(x1, y1);
    glTexCoord2f(1, 1); glVertex2f(x1, y0);
    glTexCoord2f(0, 1); glVertex2f(x0, y0);
    glEnd();
    glDisable(GL_TEXTURE_2D);

    size_t i;
    for0(i, g_clicks.size()) {
        const float* col = g_colors[i % g_nColors];
        glColor3fv(col);
        glLineWidth(2.0f);

        float px0 = x0 + (x1 - x0) * (float)g_clicks[i].x / g_imgW;
        float py0 = y1 - (y1 - y0) * (float)g_clicks[i].y / g_imgH;
        float px1 = x0 + (x1 - x0) * (float)(g_clicks[i].x + g_squareWidth) / g_imgW;
        float py1 = y1 - (y1 - y0) * (float)(g_clicks[i].y + g_squareWidth) / g_imgH;

        glBegin(GL_LINE_LOOP);
        glVertex2f(px0, py0); glVertex2f(px1, py0);
        glVertex2f(px1, py1); glVertex2f(px0, py1);
        glEnd();
    }

    char title[512];
    snprintf(title, sizeof(title), "[%d/%d] %s",
             g_curIdx + 1, (int)g_filenames.size(), g_filenames[g_curIdx].c_str());
    glutSetWindowTitle(title);
    glutSwapBuffers();
}

/* ──────────────── generic TS drawing (bands and NBR) ─────────────── */

static void drawTSGeneric(const char* label,
                          function<void(int, vector<TSPoint>&)> computeFn) {
    glClearColor(0.95f, 0.95f, 0.93f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glutSetWindowTitle(label);

    int nT = (int)g_images.size();
    if (g_clicks.empty() || nT < 2) { glutSwapBuffers(); return; }

    float globalMin = FLT_MAX, globalMax = -FLT_MAX;
    vector<vector<TSPoint>> allTS(g_clicks.size());
    size_t c;
    int t;
    for0(c, g_clicks.size()) {
        computeFn((int)c, allTS[c]);
        for0(t, nT) {
            float lo = allTS[c][t].mean - allTS[c][t].stddev;
            float hi = allTS[c][t].mean + allTS[c][t].stddev;
            if (lo < globalMin) globalMin = lo;
            if (hi > globalMax) globalMax = hi;
        }
    }
    float yRange = globalMax - globalMin;
    if (yRange < 1e-12f) yRange = 1.0f;

    float ml = 0.12f, mr = 0.05f, mb = 0.10f, mt = 0.08f;
    float pw = 1.0f - ml - mr, ph = 1.0f - mb - mt;

    glColor3f(0.3f, 0.3f, 0.3f);
    glLineWidth(1.0f);
    glBegin(GL_LINE_LOOP);
    glVertex2f(ml, mb); glVertex2f(ml + pw, mb);
    glVertex2f(ml + pw, mb + ph); glVertex2f(ml, mb + ph);
    glEnd();

    {
        float xn = ml + pw * (float)g_curIdx / (nT - 1);
        glColor3f(0.6f, 0.6f, 0.6f);
        glEnable(GL_LINE_STIPPLE);
        glLineStipple(1, 0x00FF);
        glBegin(GL_LINES); glVertex2f(xn, mb); glVertex2f(xn, mb + ph); glEnd();
        glDisable(GL_LINE_STIPPLE);
    }

    for0(c, g_clicks.size()) {
        const float* col = g_colors[c % g_nColors];
        const auto& ts = allTS[c];

        glColor3fv(col); glLineWidth(2.0f);
        glBegin(GL_LINE_STRIP);
        for0(t, nT) {
            float xn = ml + pw * (float)t / (nT - 1);
            float yn = mb + ph * (ts[t].mean - globalMin) / yRange;
            glVertex2f(xn, yn);
        }
        glEnd();

        glLineWidth(1.0f);
        glEnable(GL_LINE_STIPPLE);
        glLineStipple(2, 0xAAAA);
        glBegin(GL_LINE_STRIP);
        for0(t, nT) {
            float xn = ml + pw * (float)t / (nT - 1);
            float yn = mb + ph * ((ts[t].mean + ts[t].stddev) - globalMin) / yRange;
            glVertex2f(xn, yn);
        }
        glEnd();

        glLineStipple(1, 0x3333);
        glBegin(GL_LINE_STRIP);
        for0(t, nT) {
            float xn = ml + pw * (float)t / (nT - 1);
            float yn = mb + ph * ((ts[t].mean - ts[t].stddev) - globalMin) / yRange;
            glVertex2f(xn, yn);
        }
        glEnd();
        glDisable(GL_LINE_STIPPLE);
    }

    /* Y-axis labels */
    glColor3f(0.2f, 0.2f, 0.2f);
    char buf[64];
    int yi;
    for0(yi, 5) {
        float frac = (float)yi / 4.0f;
        float val  = globalMin + yRange * frac;
        float yn   = mb + ph * frac;
        snprintf(buf, sizeof(buf), "%.3g", val);
        glRasterPos2f(0.005f, yn - 0.01f);
        for (char* p = buf; *p; p++)
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_10, *p);
    }

    /* X-axis labels */
    glRasterPos2f(ml, 0.01f);
    { const char* p; for (p = g_filenames.front().c_str(); *p; p++)
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_10, *p); }
    float labelX = ml + pw - 0.15f;
    if (labelX < ml) labelX = ml;
    glRasterPos2f(labelX, 0.01f);
    { const char* p; for (p = g_filenames.back().c_str(); *p; p++)
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_10, *p); }

    glutSwapBuffers();
}

/* ──────────────── TS window dispatch ─────────────────────────────── */

static void drawTSForBand(int bandIdx) {
    char title[128];
    if (bandIdx < (int)g_bandNames.size())
        snprintf(title, sizeof(title), "%s (band %d)", g_bandNames[bandIdx].c_str(), bandIdx + 1);
    else
        snprintf(title, sizeof(title), "Band %d", bandIdx + 1);

    drawTSGeneric(title, [bandIdx](int ci, vector<TSPoint>& out) {
        computeTS(ci, bandIdx, out);
    });
}

static void displayNBR() {
    drawTSGeneric("NBR = (B08-B12)/(B08+B12)", [](int ci, vector<TSPoint>& out) {
        computeNBR_TS(ci, out);
    });
}

static void displayTS_dispatch() {
    int win = glutGetWindow();

    if (g_hasNBR && win == g_winNBR) { displayNBR(); return; }

    size_t i;
    for0(i, g_winTS.size()) {
        if (g_winTS[i] == win) {
            drawTSForBand(g_tsWindowBandMap[i]);
            return;
        }
    }
}

/* ──────────────────────── refresh all windows ────────────────────── */

static void refreshAll() {
    uploadTexture();
    glutSetWindow(g_winImage);
    glutPostRedisplay();

    size_t i;
    for0(i, g_winTS.size()) {
        glutSetWindow(g_winTS[i]);
        glutPostRedisplay();
    }
    if (g_hasNBR && g_winNBR > 0) {
        glutSetWindow(g_winNBR);
        glutPostRedisplay();
    }
}

/* ──────────────────── clean exit ─────────────────────────────────── */

static void cleanExit() {
    printf("clean exit\n");
    /* free image data */
    size_t i;
    for0(i, g_images.size()) {
        if (g_images[i].data) { free(g_images[i].data); g_images[i].data = nullptr; }
    }
    exit(0);
}

/* ──────────────────────── input callbacks ────────────────────────── */

static void keyboardAll(unsigned char key, int, int) {
    if (key == 'q' || key == 27) cleanExit();  /* q or Escape */
    if (key == 'c' || key == 'C') {
        g_clicks.clear();
        printf("cleared all squares\n");
        refreshAll();
    }
}

static void specialAll(int key, int, int) {
    if (key == GLUT_KEY_RIGHT) {
        g_curIdx = (g_curIdx + 1) % (int)g_images.size();
        printf("-> [%d] %s\n", g_curIdx, g_filenames[g_curIdx].c_str());
        refreshAll();
    } else if (key == GLUT_KEY_LEFT) {
        g_curIdx = (g_curIdx - 1 + (int)g_images.size()) % (int)g_images.size();
        printf("-> [%d] %s\n", g_curIdx, g_filenames[g_curIdx].c_str());
        refreshAll();
    }
}

static void mouseImage(int button, int state, int mx, int my) {
    if (state != GLUT_DOWN) return;

    if (button == GLUT_RIGHT_BUTTON) {
        g_clicks.clear();
        printf("cleared all squares\n");
        refreshAll();
        return;
    }
    if (button != GLUT_LEFT_BUTTON) return;

    float x0, y0, x1, y1;
    imgQuad(x0, y0, x1, y1);

    float wx0 = (x0 + 1.0f) / 2.0f * g_dispW;
    float wx1 = (x1 + 1.0f) / 2.0f * g_dispW;
    float wy0 = (1.0f - y1) / 2.0f * g_dispH;
    float wy1 = (1.0f - y0) / 2.0f * g_dispH;

    float fx = (float)mx, fy = (float)my;
    if (fx < wx0 || fx > wx1 || fy < wy0 || fy > wy1) return;

    int px = (int)((fx - wx0) / (wx1 - wx0) * g_imgW);
    int py = (int)((fy - wy0) / (wy1 - wy0) * g_imgH);
    px = max(0, min(px, g_imgW - 1));
    py = max(0, min(py, g_imgH - 1));

    g_clicks.push_back({px, py});
    printf("click #%d at pixel (%d, %d)\n", (int)g_clicks.size(), px, py);
    refreshAll();
}

static void reshapeImage(int w, int h) {
    g_dispW = w; g_dispH = h;
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION); glLoadIdentity();
    gluOrtho2D(-1, 1, -1, 1);
    glMatrixMode(GL_MODELVIEW); glLoadIdentity();
}

static void reshapeTS(int w, int h) {
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION); glLoadIdentity();
    gluOrtho2D(0, 1, 0, 1);
    glMatrixMode(GL_MODELVIEW); glLoadIdentity();
}

/* ──────────────── parallel file loading via parfor ───────────────── */

struct LoadSlot {
    string path;
    size_t nrow, ncol, nband;
    vector<string> bandNames;
    float* data;
    int ok;
};

static vector<LoadSlot> g_loadSlots;

static void pf_loadFile(size_t idx) {
    LoadSlot& s = g_loadSlots[idx];
    FILE* f = fopen(s.path.c_str(), "rb");
    if (!f) { s.ok = 0; return; }
    size_t nf = s.nrow * s.ncol * s.nband;
    s.data = (float*)malloc(nf * sizeof(float));
    if (!s.data) { fclose(f); s.ok = 0; return; }
    size_t nr = fread(s.data, sizeof(float), nf, f);
    fclose(f);
    s.ok = (nr == nf) ? 1 : 0;
    if (!s.ok) { free(s.data); s.data = nullptr; }
}

/* ──────── parallel pairwise duplicate detection ─────────────────── */

static vector<int> g_dupFlag; /* 1 = this index is a duplicate of previous */

static void pf_checkDup(size_t idx) {
    /* compare image idx with image idx+1 */
    const BandImage& a = g_images[idx];
    const BandImage& b = g_images[idx + 1];
    size_t n = a.width * a.height * a.nBands;
    if (memcmp(a.data, b.data, n * sizeof(float)) == 0) {
        g_dupFlag[idx + 1] = 1; /* mark the later one as duplicate */
    }
}

/* ──────────────────────────── main ───────────────────────────────── */

int main(int argc, char** argv) {
    g_dir = ".";
    string ext = "bin";
    int startIdx = 0;
    size_t i;

    for (int a = 1; a < argc; a++) {
        if (!strcmp(argv[a], "-d") && a + 1 < argc)      { g_dir = argv[++a]; }
        else if (!strcmp(argv[a], "-w") && a + 1 < argc)  { g_squareWidth = atoi(argv[++a]); }
        else if (!strcmp(argv[a], "-i") && a + 1 < argc)  { startIdx = atoi(argv[++a]); }
        else if (!strcmp(argv[a], "-e") && a + 1 < argc)  { ext = argv[++a]; }
        else if (!strcmp(argv[a], "-h") || !strcmp(argv[a], "--help")) {
            printf(
                "Usage: %s [-d dir] [-w square_width] [-i index] [-e ext]\n\n"
                "  -d <dir>   Directory of ENVI .bin/.hdr files      (default: .)\n"
                "  -w <int>   Sampling square side-length in pixels  (default: 10)\n"
                "  -i <int>   Initial image index (0-based, -1=last) (default: 0)\n"
                "  -e <ext>   File extension to filter for           (default: bin)\n\n"
                "Controls (work in every window):\n"
                "  Left/Right arrows  Navigate forward/backward in time\n"
                "  Left-click image   Add sampling square, update time series\n"
                "  Right-click image  Clear all squares and traces\n"
                "  c                  Clear all squares and traces\n"
                "  q / Escape         Clean exit\n",
                argv[0]);
            return 0;
        }
    }

    /* ═══════════════ 1. SCAN DIRECTORY ══════════════════════════════ */

    bool has_bin = false, has_tif = false;
    vector<string> allNames;
    {
        DIR* dp = opendir(g_dir.c_str());
        if (!dp) { fprintf(stderr, "cannot open directory %s\n", g_dir.c_str()); return 1; }
        struct dirent* ent;
        while ((ent = readdir(dp)) != nullptr) {
            string fname(ent->d_name);
            size_t len = fname.size();
            if (len > 4 && fname.compare(len - 4, 4, ".bin") == 0) has_bin = true;
            if (len > 4 && fname.compare(len - 4, 4, ".tif") == 0) has_tif = true;

            string dotExt = string(".") + ext;
            if (len > dotExt.size() &&
                fname.compare(len - dotExt.size(), dotExt.size(), dotExt) == 0) {
                allNames.push_back(fname);
            }
        }
        closedir(dp);
    }

    if (has_bin && has_tif) {
        err("directory contains both .bin and .tif files — please separate them");
    }

    if (allNames.empty()) {
        fprintf(stderr, "no .%s files found in %s\n", ext.c_str(), g_dir.c_str());
        return 1;
    }

    /* ═══════════════ 2. SORT BY TIMESTAMP ═══════════════════════════ */

    /* extract timestamps for sorting */
    map<string, string> fname_to_ts;
    for0(i, allNames.size()) {
        string ts = extract_timestamp(allNames[i]);
        fname_to_ts[allNames[i]] = ts;
    }

    sort(allNames.begin(), allNames.end(), [&](const string& a, const string& b) {
        const string& ta = fname_to_ts[a];
        const string& tb = fname_to_ts[b];
        if (ta.empty() && tb.empty()) return a < b;
        if (ta.empty()) return false; /* no timestamp sorts last */
        if (tb.empty()) return true;
        if (ta != tb) return ta < tb;
        return a < b; /* stable tie-break on full name */
    });

    printf("found %zu .%s files in %s (sorted by timestamp)\n",
           allNames.size(), ext.c_str(), g_dir.c_str());
    for0(i, allNames.size())
        printf("  [%zu] %s  ts=%s\n", i, allNames[i].c_str(), fname_to_ts[allNames[i]].c_str());

    /* ═══════════════ 3. CHECK FOR CACHED DUPLICATES ════════════════ */

    set<string> cached_dups = load_duplicates();
    if (!cached_dups.empty()) {
        printf("loaded %zu cached duplicate filenames from .restore_duplicates\n", cached_dups.size());
        vector<string> filtered;
        for0(i, allNames.size()) {
            if (cached_dups.count(allNames[i]) == 0)
                filtered.push_back(allNames[i]);
            else
                printf("  skipping cached duplicate: %s\n", allNames[i].c_str());
        }
        allNames = move(filtered);
    }

    g_filenames = allNames;
    size_t nFiles = g_filenames.size();

    /* ═══════════════ 4. READ HEADERS & PRE-CHECK DIMENSIONS ════════ */

    g_loadSlots.resize(nFiles);
    for0(i, nFiles) {
        string path = g_dir + "/" + g_filenames[i];
        string hfn  = hdr_fn(path);
        g_loadSlots[i].path = path;
        g_loadSlots[i].data = nullptr;
        g_loadSlots[i].ok = 0;
        hread(hfn, g_loadSlots[i].nrow, g_loadSlots[i].ncol,
              g_loadSlots[i].nband, g_loadSlots[i].bandNames);
    }

    /* find majority dimensions */
    {
        map<string, int> dim_counts;
        for0(i, nFiles) {
            char key[128];
            snprintf(key, sizeof(key), "%zu_%zu_%zu",
                     g_loadSlots[i].nrow, g_loadSlots[i].ncol, g_loadSlots[i].nband);
            dim_counts[string(key)]++;
        }
        /* find majority */
        string majority_key;
        int majority_count = 0;
        for (auto& kv : dim_counts) {
            if (kv.second > majority_count) {
                majority_count = kv.second;
                majority_key = kv.first;
            }
        }
        /* check expected file sizes */
        size_t maj_nrow, maj_ncol, maj_nband;
        sscanf(majority_key.c_str(), "%zu_%zu_%zu", &maj_nrow, &maj_ncol, &maj_nband);
        size_t expected_bytes = maj_nrow * maj_ncol * maj_nband * sizeof(float);

        bool mismatch = false;
        for0(i, nFiles) {
            char key[128];
            snprintf(key, sizeof(key), "%zu_%zu_%zu",
                     g_loadSlots[i].nrow, g_loadSlots[i].ncol, g_loadSlots[i].nband);
            if (string(key) != majority_key) {
                fprintf(stderr, "DIMENSION MISMATCH: %s has %zux%zu, %zu bands "
                        "(expected %zux%zu, %zu bands)\n",
                        g_filenames[i].c_str(),
                        g_loadSlots[i].ncol, g_loadSlots[i].nrow, g_loadSlots[i].nband,
                        maj_ncol, maj_nrow, maj_nband);
                mismatch = true;
            }
            /* also check file size on disk */
            size_t actual = fsize(g_loadSlots[i].path);
            if (actual != expected_bytes) {
                fprintf(stderr, "FILE SIZE MISMATCH: %s is %zu bytes (expected %zu)\n",
                        g_filenames[i].c_str(), actual, expected_bytes);
                mismatch = true;
            }
        }
        if (mismatch) err("dimension/size mismatches found — please fix or remove the offending files");
    }

    /* ═══════════════ 5. PARALLEL BINARY READS ══════════════════════ */

    printf("loading %zu files (parallel fread)...\n", nFiles);
    parfor(0, nFiles, pf_loadFile);

    /* collect successful loads */
    {
        vector<string> goodNames;
        for0(i, nFiles) {
            LoadSlot& s = g_loadSlots[i];
            if (!s.ok) {
                fprintf(stderr, "warning: failed to load %s\n", s.path.c_str());
                continue;
            }
            BandImage img;
            img.width     = s.ncol;
            img.height    = s.nrow;
            img.nBands    = s.nband;
            img.data      = s.data;
            img.bandNames = move(s.bandNames);
            img.filename  = g_filenames[i];
            img.timestamp = fname_to_ts[g_filenames[i]];
            g_images.push_back(move(img));
            goodNames.push_back(g_filenames[i]);
        }
        g_filenames = move(goodNames);
    }

    if (g_images.empty()) err("no files loaded successfully");

    /* ═══════════════ 6. PARALLEL DUPLICATE REMOVAL ═════════════════ */

    if (g_images.size() >= 2 && cached_dups.empty()) {
        printf("checking %zu pairs for duplicates (parallel memcmp)...\n", g_images.size() - 1);
        g_dupFlag.assign(g_images.size(), 0);
        parfor(0, g_images.size() - 1, pf_checkDup);

        /* walk pairwise: if idx is dup of idx-1, mark it; chain extends */
        set<size_t> to_remove;
        for (i = 1; i < g_images.size(); i++) {
            if (g_dupFlag[i]) {
                to_remove.insert(i);
                printf("  duplicate: [%zu] %s == [%zu] %s\n",
                       i, g_filenames[i].c_str(), i - 1, g_filenames[i - 1].c_str());
            }
        }

        if (!to_remove.empty()) {
            /* save for next run */
            set<size_t> dup_indices = to_remove;
            save_duplicates(dup_indices, g_filenames);

            /* remove from back to front */
            vector<size_t> sorted_remove(to_remove.rbegin(), to_remove.rend());
            for (auto idx : sorted_remove) {
                if (g_images[idx].data) free(g_images[idx].data);
                g_images.erase(g_images.begin() + idx);
                g_filenames.erase(g_filenames.begin() + idx);
            }
            printf("removed %zu duplicates, %zu images remain\n",
                   to_remove.size(), g_images.size());
        } else {
            printf("no duplicates found\n");
            /* save empty file so we skip the check next time */
            set<size_t> empty_set;
            save_duplicates(empty_set, g_filenames);
        }
    }

    if (g_images.empty()) err("no images remain after duplicate removal");

    g_imgW   = (int)g_images[0].width;
    g_imgH   = (int)g_images[0].height;
    g_nBands = (int)g_images[0].nBands;
    g_bandNames = g_images[0].bandNames;

    /* ═══════════════ 7. DETECT B08 / B12 FOR NBR ═══════════════════ */

    for0(i, g_bandNames.size()) {
        if (g_bandNames[i].find("B08") != string::npos) g_b08_idx = (int)i;
        if (g_bandNames[i].find("B12") != string::npos) g_b12_idx = (int)i;
    }
    g_hasNBR = (g_b08_idx >= 0 && g_b12_idx >= 0);
    if (g_hasNBR)
        printf("NBR enabled: B08=band %d, B12=band %d\n", g_b08_idx + 1, g_b12_idx + 1);
    else
        printf("NBR disabled (need both B08 and B12 band names)\n");

    if (startIdx < 0) startIdx = (int)g_images.size() + startIdx;
    startIdx = max(0, min(startIdx, (int)g_images.size() - 1));
    g_curIdx = startIdx;

    printf("image: %d x %d, %d bands, %zu dates\n", g_imgW, g_imgH, g_nBands, g_images.size());
    printf("band names:");
    for0(i, g_bandNames.size()) printf(" %s", g_bandNames[i].c_str());
    printf("\n");
    for0(i, g_images.size())
        printf("  [%zu] %s  ts=%s%s\n", i, g_filenames[i].c_str(),
               g_images[i].timestamp.c_str(),
               (int)i == g_curIdx ? " <-" : "");

    /* ═══════════════ 8. PRECOMPUTE RGB TEXTURES ════════════════════ */

    precomputeAllRGB();

    printf("controls: left/right=navigate, left-click=add square, right-click/c=clear, q/Esc=quit\n");

    /* ═══════════════ 9. GLUT INIT ══════════════════════════════════ */

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);

    g_dispW = min(g_imgW, 1200);
    g_dispH = (int)((float)g_dispW / g_imgW * g_imgH);
    if (g_dispH > 900) {
        g_dispH = 900;
        g_dispW = (int)((float)g_dispH / g_imgH * g_imgW);
    }

    /* image window */
    glutInitWindowSize(g_dispW, g_dispH);
    glutInitWindowPosition(50, 50);
    g_winImage = glutCreateWindow("Band Viewer");
    glutDisplayFunc(displayImage);
    glutReshapeFunc(reshapeImage);
    glutMouseFunc(mouseImage);
    glutKeyboardFunc(keyboardAll);
    glutSpecialFunc(specialAll);

    uploadTexture();

    /* one TS window per band */
    int tsW = 480, tsH = 220;
    int nTSBands = min(g_nBands, MAX_TS_WINDOWS);
    for (int b = 0; b < nTSBands; b++) {
        int col = b / 4;
        int row = b % 4;
        glutInitWindowSize(tsW, tsH);
        glutInitWindowPosition(60 + g_dispW + 20 + col * (tsW + 10),
                               50 + row * (tsH + 50));
        char name[128];
        if (b < (int)g_bandNames.size())
            snprintf(name, sizeof(name), "%s (band %d)", g_bandNames[b].c_str(), b + 1);
        else
            snprintf(name, sizeof(name), "Band %d", b + 1);
        int win = glutCreateWindow(name);
        g_winTS.push_back(win);
        g_tsWindowBandMap[b] = b;
        glutDisplayFunc(displayTS_dispatch);
        glutReshapeFunc(reshapeTS);
        glutKeyboardFunc(keyboardAll);
        glutSpecialFunc(specialAll);
    }

    /* NBR window (below the band windows) */
    if (g_hasNBR) {
        int nbrRow = nTSBands;
        int col = nbrRow / 4;
        int row = nbrRow % 4;
        glutInitWindowSize(tsW, tsH);
        glutInitWindowPosition(60 + g_dispW + 20 + col * (tsW + 10),
                               50 + row * (tsH + 50));
        g_winNBR = glutCreateWindow("NBR = (B08-B12)/(B08+B12)");
        glutDisplayFunc(displayTS_dispatch);
        glutReshapeFunc(reshapeTS);
        glutKeyboardFunc(keyboardAll);
        glutSpecialFunc(specialAll);
    }

    glutMainLoop();
    return 0;
}



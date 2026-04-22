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
 *   q / Escape           Quit
 *   Left-click on image  Add a sampling square; TS plots update
 *   Right-click on image Clear all squares and traces
 *
 * Data format:
 *   ENVI BSQ float32 files.  Header (.hdr) is located as either
 *   <name>.hdr or <name>.bin.hdr, and must contain samples, lines,
 *   bands fields.  Data type is assumed float32 (ENVI type 4).
 *   Band-sequential: the .bin file is [band0_all_pixels][band1_all_pixels]...
 *
 * Architecture:
 *   - ENVI files are read with direct fread (no GDAL), matching misc.cpp
 *     bread() / hread() semantics exactly.
 *   - Parallel file reading uses a parfor-style work-stealing thread pool.
 *   - RGB textures are precomputed for every date in parallel at startup;
 *     arrow-key navigation is a single glTexSubImage2D call.
 *   - Histogram stretch uses nth_element (O(n)) for percentile finding.
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
    /* try <name>.hdr first, then <name>.bin.hdr */
    string base = fn.substr(0, fn.size() - 4); /* strip .bin */
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
        /* accumulate band names block (may span multiple lines) */
        if (in_band_names) {
            bn_accum += line;
            if (line.find('}') != string::npos) in_band_names = false;
            continue;
        }

        size_t eq = line.find('=');
        if (eq == string::npos) continue;
        string key = line.substr(0, eq);
        string val = line.substr(eq + 1);

        /* trim whitespace */
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

    /* parse band names from accumulated string like {B12, B11, B09, B08} */
    if (!bn_accum.empty()) {
        /* strip braces */
        size_t i;
        string clean;
        for0(i, bn_accum.size()) {
            char c = bn_accum[i];
            if (c != '{' && c != '}') clean += c;
        }
        /* split on comma */
        istringstream iss(clean);
        string tok;
        while (getline(iss, tok, ',')) {
            while (!tok.empty() && isspace(tok.front())) tok.erase(tok.begin());
            while (!tok.empty() && isspace(tok.back()))  tok.pop_back();
            if (!tok.empty()) band_names.push_back(tok);
        }
    }

    printf("hread: %s nrow=%zu ncol=%zu nband=%zu\n", hfn.c_str(), nrow, ncol, nband);
}

/* ─────────────── ENVI BSQ float32 reading (from misc.cpp) ───────── */

/*
 * bread: read a BSQ float32 binary file into a flat float array.
 * Layout: band0[nrow*ncol], band1[nrow*ncol], ...
 * This matches misc.cpp bread() exactly.
 */
static float* bread(const string& bfn, size_t nrow, size_t ncol, size_t nband) {
    FILE* f = fopen(bfn.c_str(), "rb");
    if (!f) { err(string("failed to open: ") + bfn); return nullptr; }
    size_t nf = nrow * ncol * nband;
    float* dat = (float*)malloc(nf * sizeof(float));
    if (!dat) err("failed to allocate memory for image data");
    size_t nr = fread(dat, sizeof(float), nf, f);
    if (nr != nf) {
        printf("bread(%s): expected %zu floats, got %zu\n", bfn.c_str(), nf, nr);
        err("bread: unexpected read count");
    }
    fclose(f);
    return dat;
}

/* ──────────────────────────── data types ──────────────────────────── */

struct BandImage {
    size_t width  = 0;
    size_t height = 0;
    size_t nBands = 0;
    float* data   = nullptr; /* flat BSQ: band_b starts at data[b * width * height] */
    vector<string> bandNames;
    string filename;

    /* access band b as a contiguous float* of width*height pixels, row-major */
    const float* band(size_t b) const { return data + b * width * height; }
    float* band(size_t b) { return data + b * width * height; }
};

struct Click { int x, y; };
struct TSPoint { float mean, stddev; };

/* ──────────────────────────── globals ─────────────────────────────── */

static vector<BandImage>  g_images;
static vector<string>     g_filenames;
static int g_curIdx       = 0;
static int g_squareWidth  = 10;
static int g_imgW = 0, g_imgH = 0;
static int g_nBands = 0;
static vector<string> g_bandNames;

static vector<Click> g_clicks;
static const float g_colors[][3] = {
    {0.2f,0.5f,1.0f}, {1.0f,0.3f,0.3f}, {1.0f,1.0f,0.2f},
    {0.1f,0.9f,0.1f}, {0.0f,1.0f,1.0f}, {1.0f,0.4f,1.0f}
};
static const int g_nColors = 6;

static int g_winImage = -1;
static vector<int> g_winTS;

/* pre-built RGB textures: one per date, computed once at startup */
static vector<vector<unsigned char>> g_rgbTextures;
static GLuint g_texId = 0;
static bool   g_texAllocated = false;

static int g_dispW = 800, g_dispH = 600;

#define MAX_TS_WINDOWS 32
static int g_tsWindowBandMap[MAX_TS_WINDOWS];

/* ─────────────── parfor: work-stealing parallel for ─────────────── */
/*
 * Adapted from misc.cpp parfor pattern: a shared job counter protected
 * by a mutex, with worker threads pulling jobs until exhausted.
 */

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
    pf_eval  = eval;
    pf_end_j = end_j;
    pf_next_j = start_j;

    int cores_avail = sysconf(_SC_NPROCESSORS_ONLN);
    int n_cores = (cores_use > 0) ? min(cores_use, cores_avail) : cores_avail;
    n_cores = min(n_cores, (int)(end_j - start_j));
    if (n_cores < 1) n_cores = 1;
    printf("parfor: %zu jobs, %d threads\n", end_j - start_j, n_cores);

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

/* ──────────────── fast histogram stretch to u8 (O(n)) ───────────── */

static void histStretchToU8(const float* src, size_t n, unsigned char* dst, float pct = 1.0f) {
    vector<float> vals;
    vals.reserve(n);
    size_t i;
    for0(i, n) {
        if (isfinite(src[i])) vals.push_back(src[i]);
    }
    if (vals.empty()) { memset(dst, 0, n); return; }

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

    /* first 3 bands -> R, G, B; fewer bands => duplicate last */
    for0(c, (size_t)3) {
        size_t srcBand = min(c, rgbCount - 1);
        histStretchToU8(img.band(srcBand), n, chan.data(), 1.0f);
        for0(i, n)
            tex[i * 3 + c] = chan[i];
    }
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
    glutSetWindow(g_winImage);  /* ensure correct GL context */
    if (g_texId == 0) glGenTextures(1, &g_texId);
    glBindTexture(GL_TEXTURE_2D, g_texId);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    /*
     * Critical: GL_UNPACK_ALIGNMENT defaults to 4, meaning OpenGL expects
     * each row of pixel data to start on a 4-byte boundary.  With GL_RGB
     * (3 bytes/pixel), row stride = width * 3.  For width=5490, that's
     * 16470 bytes which is NOT a multiple of 4 — so OpenGL pads each row
     * by 2 bytes, shifting all subsequent rows and producing horizontal
     * colour stripes.  Setting alignment to 1 fixes this.
     */
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

/* ──────────────── GLUT display: image window ─────────────────────── */

/* compute the image quad corners in NDC [-1,1] given current window size */
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

    /*
     * Texture coords: (0,0) = top-left of image data, (1,1) = bottom-right.
     * BSQ row-major: row 0 is the first row in the file = top of image.
     * OpenGL: Y increases upward.
     * So we map: texcoord (0,0) -> vertex top-left = (x0, y1)
     *            texcoord (1,1) -> vertex bot-right = (x1, y0)
     */
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0); glVertex2f(x0, y1); /* TL */
    glTexCoord2f(1, 0); glVertex2f(x1, y1); /* TR */
    glTexCoord2f(1, 1); glVertex2f(x1, y0); /* BR */
    glTexCoord2f(0, 1); glVertex2f(x0, y0); /* BL */
    glEnd();
    glDisable(GL_TEXTURE_2D);

    /* draw sampling squares */
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
        glVertex2f(px0, py0);
        glVertex2f(px1, py0);
        glVertex2f(px1, py1);
        glVertex2f(px0, py1);
        glEnd();
    }

    char title[512];
    snprintf(title, sizeof(title), "[%d/%d] %s",
             g_curIdx + 1, (int)g_filenames.size(), g_filenames[g_curIdx].c_str());
    glutSetWindowTitle(title);

    glutSwapBuffers();
}

/* ──────────────── GLUT display: TS windows ───────────────────────── */

static void drawTSForBand(int bandIdx) {
    glClearColor(0.95f, 0.95f, 0.93f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    int nT = (int)g_images.size();

    /* window title: use band name if available */
    char title[128];
    if (bandIdx < (int)g_bandNames.size())
        snprintf(title, sizeof(title), "%s (band %d)", g_bandNames[bandIdx].c_str(), bandIdx + 1);
    else
        snprintf(title, sizeof(title), "Band %d", bandIdx + 1);
    glutSetWindowTitle(title);

    if (g_clicks.empty() || nT < 2) { glutSwapBuffers(); return; }

    float globalMin = FLT_MAX, globalMax = -FLT_MAX;
    vector<vector<TSPoint>> allTS(g_clicks.size());
    size_t c;
    int t;
    for0(c, g_clicks.size()) {
        computeTS((int)c, bandIdx, allTS[c]);
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

    /* axes box */
    glColor3f(0.3f, 0.3f, 0.3f);
    glLineWidth(1.0f);
    glBegin(GL_LINE_LOOP);
    glVertex2f(ml, mb);
    glVertex2f(ml + pw, mb);
    glVertex2f(ml + pw, mb + ph);
    glVertex2f(ml, mb + ph);
    glEnd();

    /* vertical marker for current time step */
    {
        float xn = ml + pw * (float)g_curIdx / (nT - 1);
        glColor3f(0.6f, 0.6f, 0.6f);
        glEnable(GL_LINE_STIPPLE);
        glLineStipple(1, 0x00FF);
        glBegin(GL_LINES);
        glVertex2f(xn, mb); glVertex2f(xn, mb + ph);
        glEnd();
        glDisable(GL_LINE_STIPPLE);
    }

    for0(c, g_clicks.size()) {
        const float* col = g_colors[c % g_nColors];
        const auto& ts = allTS[c];

        /* mean (solid) */
        glColor3fv(col);
        glLineWidth(2.0f);
        glBegin(GL_LINE_STRIP);
        for0(t, nT) {
            float xn = ml + pw * (float)t / (nT - 1);
            float yn = mb + ph * (ts[t].mean - globalMin) / yRange;
            glVertex2f(xn, yn);
        }
        glEnd();

        /* +std (dashed) */
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

        /* -std (dotted) */
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
        char* p;
        for (p = buf; *p; p++)
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_10, *p);
    }

    /* X-axis labels: first and last filename */
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

static void displayTS_dispatch() {
    int win = glutGetWindow();
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
}

/* ──────────────────────── input callbacks ────────────────────────── */

static void keyboardAll(unsigned char key, int, int) {
    if (key == 'q' || key == 27) exit(0);
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

    /* convert window pixel coords -> image pixel coords */
    float x0, y0, x1, y1;
    imgQuad(x0, y0, x1, y1);

    /* NDC -> window pixel mapping:
     *   window_x = (ndc_x + 1) / 2 * g_dispW
     *   window_y = (1 - ndc_y) / 2 * g_dispH   (GLUT y=0 at top)
     * So quad corners in window pixels: */
    float wx0 = (x0 + 1.0f) / 2.0f * g_dispW;
    float wx1 = (x1 + 1.0f) / 2.0f * g_dispW;
    float wy0 = (1.0f - y1) / 2.0f * g_dispH;  /* top of image in window */
    float wy1 = (1.0f - y0) / 2.0f * g_dispH;  /* bottom of image in window */

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

/*
 * File loading: each file is independent (separate fopen/fread/fclose),
 * so full parallelism is safe — unlike GDAL, there is no shared driver
 * state.  We read the header serially first to validate dimensions, then
 * parfor the actual fread which is the bottleneck.
 */

struct LoadSlot {
    string path;
    string hdr_path;
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

/* ──────────────────────────── main ───────────────────────────────── */

int main(int argc, char** argv) {
    string dir = ".";
    string ext = "bin";
    int startIdx = 0;
    size_t i;

    for (int a = 1; a < argc; a++) {
        if (!strcmp(argv[a], "-d") && a + 1 < argc)      { dir = argv[++a]; }
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
                "  q / Escape         Quit\n",
                argv[0]);
            return 0;
        }
    }

    /* scan directory using opendir (no popen/ls) */
    {
        string dotExt = string(".") + ext;
        DIR* dp = opendir(dir.c_str());
        if (!dp) { fprintf(stderr, "cannot open directory %s\n", dir.c_str()); return 1; }
        struct dirent* ent;
        while ((ent = readdir(dp)) != nullptr) {
            string fname(ent->d_name);
            if (fname.size() > dotExt.size() &&
                fname.compare(fname.size() - dotExt.size(), dotExt.size(), dotExt) == 0) {
                g_filenames.push_back(fname);
            }
        }
        closedir(dp);
    }
    sort(g_filenames.begin(), g_filenames.end());

    if (g_filenames.empty()) {
        fprintf(stderr, "no .%s files found in %s\n", ext.c_str(), dir.c_str());
        return 1;
    }
    printf("found %zu .%s files in %s\n", g_filenames.size(), ext.c_str(), dir.c_str());

    /* ---- phase 1: serial header reads (fast, validates dimensions) ---- */
    size_t nFiles = g_filenames.size();
    g_loadSlots.resize(nFiles);
    for0(i, nFiles) {
        string path = dir + "/" + g_filenames[i];
        string hfn  = hdr_fn(path);
        g_loadSlots[i].path = path;
        g_loadSlots[i].hdr_path = hfn;
        g_loadSlots[i].data = nullptr;
        g_loadSlots[i].ok = 0;
        hread(hfn, g_loadSlots[i].nrow, g_loadSlots[i].ncol,
              g_loadSlots[i].nband, g_loadSlots[i].bandNames);
    }

    /* ---- phase 2: parallel binary reads via parfor ---- */
    printf("loading %zu files (parallel fread)...\n", nFiles);
    parfor(0, nFiles, pf_loadFile);

    /* ---- collect successful loads ---- */
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
            img.data      = s.data;    /* transfer ownership */
            img.bandNames = move(s.bandNames);
            img.filename  = g_filenames[i];
            g_images.push_back(move(img));
            goodNames.push_back(g_filenames[i]);
            printf("  [%zu] %s  %zux%zu, %zu bands  ok\n",
                   goodNames.size(), g_filenames[i].c_str(), s.ncol, s.nrow, s.nband);
        }
        g_filenames = move(goodNames);
    }

    if (g_images.empty()) err("no files loaded successfully");

    g_imgW   = (int)g_images[0].width;
    g_imgH   = (int)g_images[0].height;
    g_nBands = (int)g_images[0].nBands;
    g_bandNames = g_images[0].bandNames;

    if (startIdx < 0) startIdx = (int)g_images.size() + startIdx;
    startIdx = max(0, min(startIdx, (int)g_images.size() - 1));
    g_curIdx = startIdx;

    printf("image: %d x %d, %d bands, %zu dates\n", g_imgW, g_imgH, g_nBands, g_images.size());
    printf("band names:");
    for0(i, g_bandNames.size()) printf(" %s", g_bandNames[i].c_str());
    printf("\n");
    printf("initial: [%d] %s\n", g_curIdx, g_filenames[g_curIdx].c_str());

    /* ---- precompute all RGB textures in parallel ---- */
    precomputeAllRGB();

    printf("controls: left/right=navigate, left-click=add square, right-click/c=clear, q=quit\n");

    /* ---- GLUT init ---- */
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

    /* one TS window per band, titled with band name */
    int tsW = 480, tsH = 220;
    for (int b = 0; b < g_nBands && b < MAX_TS_WINDOWS; b++) {
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

    glutMainLoop();
    return 0;
}

/*
 * band_timeseries_gui.cpp
 *
 * A fast, interactive multi-band satellite-image time-series viewer.
 *
 * Build:
 *   g++ -O2 -o band_ts band_timeseries_gui.cpp $(gdal-config --cflags --libs) -lGL -lGLU -lglut -lpthread -std=c++17
 *
 * Usage:
 *   ./band_ts [options]
 *
 *   -d <dir>    Directory containing GDAL-readable raster files (default: ".")
 *   -w <int>    Side-length of the sampling square in pixels   (default: 10)
 *   -i <int>    Initial image index, 0-based (default: 0). -1 = last.
 *   -e <ext>    File extension filter, e.g. "tif" or "bin"    (default: "tif")
 *
 * Controls (work in ANY window):
 *   Left / Right arrow   Navigate backward / forward through dates
 *   c                    Clear all sampling squares and time-series traces
 *   q / Escape           Quit
 *   Left-click on image  Add a sampling square; time-series plots update
 *   Right-click on image Clear all squares and traces
 *
 * Architecture:
 *   - One GLUT window shows the reference image (RGB composite of the first
 *     three bands, histogram-stretched to [0,1]).
 *   - One GLUT window per band shows the mean +/- std time series for every
 *     sampling square placed by the user.
 *   - All raster files are loaded in parallel with pthreads at startup.
 *   - Band data is kept in memory as float arrays for instant navigation.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <climits>
#include <cfloat>
#include <unistd.h>
#include <pthread.h>

#include <gdal_priv.h>
#include <cpl_conv.h>

#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/freeglut.h>

/* ──────────────────────────── data types ──────────────────────────── */

struct BandImage {
    int width  = 0;
    int height = 0;
    int nBands = 0;
    std::vector<std::vector<float>> bands; // bands[b][pixel]
    std::string filename;
};

struct Click {
    int x, y;
};

struct TSPoint {
    float mean, stddev;
};

/* ──────────────────────────── globals ─────────────────────────────── */

static std::vector<BandImage>   g_images;
static std::vector<std::string> g_filenames;
static int g_curIdx       = 0;
static int g_squareWidth  = 10;
static int g_imgW = 0, g_imgH = 0;
static int g_nBands = 0;

// clicks / colours
static std::vector<Click> g_clicks;
static const float g_colors[][3] = {
    {0.2f,0.5f,1.0f}, {1.0f,0.3f,0.3f}, {1.0f,1.0f,0.2f},
    {0.1f,0.9f,0.1f}, {0.0f,1.0f,1.0f}, {1.0f,0.4f,1.0f}
};
static const int g_nColors = 6;

// GLUT window IDs
static int g_winImage = -1;
static std::vector<int> g_winTS; // one per band

// Pre-built RGB texture for the current image
static std::vector<unsigned char> g_rgbTex;
static GLuint g_texId = 0;

// Display dimensions for the image window
static int g_dispW = 800, g_dispH = 600;

#define MAX_TS_WINDOWS 32
static int g_tsWindowBandMap[MAX_TS_WINDOWS];

/* ──────────────────────────── histogram stretch ──────────────────── */

static void histStretch(const float* src, int n, float* dst, float pct = 1.0f) {
    std::vector<float> vals;
    vals.reserve(n);
    for (int i = 0; i < n; i++) {
        if (std::isfinite(src[i])) vals.push_back(src[i]);
    }
    if (vals.empty()) { std::fill(dst, dst + n, 0.0f); return; }
    std::sort(vals.begin(), vals.end());
    float frac = pct / 100.0f;
    int   lo   = (int)std::floor(vals.size() * frac);
    int   hi   = (int)std::floor(vals.size() * (1.0f - frac));
    if (hi <= lo) hi = lo + 1;
    if (hi >= (int)vals.size()) hi = (int)vals.size() - 1;
    float vmin = vals[lo], vmax = vals[hi];
    float range = (vmax - vmin);
    if (range < 1e-12f) range = 1.0f;
    for (int i = 0; i < n; i++) {
        float v = (src[i] - vmin) / range;
        dst[i] = std::max(0.0f, std::min(1.0f, v));
    }
}

/* ──────────────────────────── build RGB texture ──────────────────── */

static void buildRGBTexture() {
    const BandImage& img = g_images[g_curIdx];
    int n = img.width * img.height;
    g_rgbTex.resize(n * 3);

    int rgbCount = std::min(img.nBands, 3);
    std::vector<float> stretched(n);
    std::vector<std::vector<float>> ch(3, std::vector<float>(n, 0.0f));

    for (int c = 0; c < rgbCount; c++) {
        histStretch(img.bands[c].data(), n, stretched.data(), 1.0f);
        ch[c].assign(stretched.begin(), stretched.end());
    }
    for (int i = 0; i < n; i++) {
        g_rgbTex[i * 3 + 0] = (unsigned char)(ch[0][i] * 255.0f);
        g_rgbTex[i * 3 + 1] = (unsigned char)(ch[std::min(1, rgbCount - 1)][i] * 255.0f);
        g_rgbTex[i * 3 + 2] = (unsigned char)(ch[std::min(2, rgbCount - 1)][i] * 255.0f);
    }
}

static void uploadTexture() {
    if (g_texId == 0) glGenTextures(1, &g_texId);
    glBindTexture(GL_TEXTURE_2D, g_texId);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, g_imgW, g_imgH, 0,
                 GL_RGB, GL_UNSIGNED_BYTE, g_rgbTex.data());
}

/* ──────────────────────────── time-series stats ──────────────────── */

static void computeTS(int clickIdx, int band, std::vector<TSPoint>& out) {
    out.resize(g_images.size());
    int cx = g_clicks[clickIdx].x;
    int cy = g_clicks[clickIdx].y;
    int w  = g_squareWidth;
    for (size_t t = 0; t < g_images.size(); t++) {
        const auto& img = g_images[t];
        double sum = 0, sum2 = 0;
        int cnt = 0;
        for (int dy = 0; dy < w && (cy + dy) < img.height; dy++) {
            for (int dx = 0; dx < w && (cx + dx) < img.width; dx++) {
                int idx = (cy + dy) * img.width + (cx + dx);
                float v = img.bands[band][idx];
                if (std::isfinite(v)) {
                    sum  += v;
                    sum2 += (double)v * v;
                    cnt++;
                }
            }
        }
        if (cnt > 0) {
            float m = (float)(sum / cnt);
            float s = (float)std::sqrt(std::max(0.0, sum2 / cnt - (double)m * m));
            out[t] = {m, s};
        } else {
            out[t] = {0.0f, 0.0f};
        }
    }
}

/* ──────────────────────────── GLUT display: image window ─────────── */

static void displayImage() {
    glutSetWindow(g_winImage);
    glClearColor(0.12f, 0.12f, 0.14f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, g_texId);
    glColor3f(1, 1, 1);

    float winAspect = (float)g_dispW / g_dispH;
    float imgAspect = (float)g_imgW / g_imgH;
    float x0, y0, x1, y1;
    if (imgAspect > winAspect) {
        x0 = -1.0f; x1 = 1.0f;
        float h = 2.0f * winAspect / imgAspect;
        y0 = -h / 2; y1 = h / 2;
    } else {
        y0 = -1.0f; y1 = 1.0f;
        float w = 2.0f * imgAspect / winAspect;
        x0 = -w / 2; x1 = w / 2;
    }

    glBegin(GL_QUADS);
    glTexCoord2f(0, 0); glVertex2f(x0, y1);
    glTexCoord2f(1, 0); glVertex2f(x1, y1);
    glTexCoord2f(1, 1); glVertex2f(x1, y0);
    glTexCoord2f(0, 1); glVertex2f(x0, y0);
    glEnd();
    glDisable(GL_TEXTURE_2D);

    // Draw sampling squares
    for (size_t i = 0; i < g_clicks.size(); i++) {
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

    // Window title
    char title[256];
    snprintf(title, sizeof(title), "[%d/%d] %s",
             g_curIdx + 1, (int)g_filenames.size(), g_filenames[g_curIdx].c_str());
    glutSetWindowTitle(title);

    glutSwapBuffers();
}

/* ──────────────────────────── GLUT display: TS windows ───────────── */

static void drawTSForBand(int bandIdx) {
    glClearColor(0.95f, 0.95f, 0.93f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    int nT = (int)g_images.size();

    // Window title always shows band name
    char title[64];
    snprintf(title, sizeof(title), "Band %d", bandIdx + 1);
    glutSetWindowTitle(title);

    if (g_clicks.empty() || nT < 2) {
        glutSwapBuffers();
        return;
    }

    // Compute global Y range across all clicks for this band
    float globalMin = FLT_MAX, globalMax = -FLT_MAX;
    std::vector<std::vector<TSPoint>> allTS(g_clicks.size());
    for (size_t c = 0; c < g_clicks.size(); c++) {
        computeTS((int)c, bandIdx, allTS[c]);
        for (int t = 0; t < nT; t++) {
            float lo = allTS[c][t].mean - allTS[c][t].stddev;
            float hi = allTS[c][t].mean + allTS[c][t].stddev;
            if (lo < globalMin) globalMin = lo;
            if (hi > globalMax) globalMax = hi;
        }
    }
    float yRange = globalMax - globalMin;
    if (yRange < 1e-12f) yRange = 1.0f;

    // Margins in NDC [0,1]
    float ml = 0.12f, mr = 0.05f, mb = 0.10f, mt = 0.08f;
    float pw = 1.0f - ml - mr, ph = 1.0f - mb - mt;

    // Axes box
    glColor3f(0.3f, 0.3f, 0.3f);
    glLineWidth(1.0f);
    glBegin(GL_LINE_LOOP);
    glVertex2f(ml, mb);
    glVertex2f(ml + pw, mb);
    glVertex2f(ml + pw, mb + ph);
    glVertex2f(ml, mb + ph);
    glEnd();

    // Vertical marker for current time step
    {
        float xn = ml + pw * (float)g_curIdx / (nT - 1);
        glColor3f(0.6f, 0.6f, 0.6f);
        glEnable(GL_LINE_STIPPLE);
        glLineStipple(1, 0x00FF);
        glBegin(GL_LINES);
        glVertex2f(xn, mb);
        glVertex2f(xn, mb + ph);
        glEnd();
        glDisable(GL_LINE_STIPPLE);
    }

    // Plot each click's time series
    for (size_t c = 0; c < g_clicks.size(); c++) {
        const float* col = g_colors[c % g_nColors];
        const auto& ts = allTS[c];

        // Mean line (solid, thick)
        glColor3fv(col);
        glLineWidth(2.0f);
        glBegin(GL_LINE_STRIP);
        for (int t = 0; t < nT; t++) {
            float xn = ml + pw * (float)t / (nT - 1);
            float yn = mb + ph * (ts[t].mean - globalMin) / yRange;
            glVertex2f(xn, yn);
        }
        glEnd();

        // +std (dashed)
        glLineWidth(1.0f);
        glEnable(GL_LINE_STIPPLE);
        glLineStipple(2, 0xAAAA);
        glBegin(GL_LINE_STRIP);
        for (int t = 0; t < nT; t++) {
            float xn = ml + pw * (float)t / (nT - 1);
            float yn = mb + ph * ((ts[t].mean + ts[t].stddev) - globalMin) / yRange;
            glVertex2f(xn, yn);
        }
        glEnd();

        // -std (dotted)
        glLineStipple(1, 0x3333);
        glBegin(GL_LINE_STRIP);
        for (int t = 0; t < nT; t++) {
            float xn = ml + pw * (float)t / (nT - 1);
            float yn = mb + ph * ((ts[t].mean - ts[t].stddev) - globalMin) / yRange;
            glVertex2f(xn, yn);
        }
        glEnd();
        glDisable(GL_LINE_STIPPLE);
    }

    // Y-axis labels
    glColor3f(0.2f, 0.2f, 0.2f);
    char buf[64];
    for (int i = 0; i <= 4; i++) {
        float frac = (float)i / 4.0f;
        float val  = globalMin + yRange * frac;
        float yn   = mb + ph * frac;
        snprintf(buf, sizeof(buf), "%.3g", val);
        glRasterPos2f(0.005f, yn - 0.01f);
        for (char* p = buf; *p; p++)
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_10, *p);
    }

    // X-axis: first and last filename
    glRasterPos2f(ml, 0.01f);
    for (const char* p = g_filenames.front().c_str(); *p; p++)
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_10, *p);

    // right-align last label roughly
    float labelX = ml + pw - 0.15f;
    if (labelX < ml) labelX = ml;
    glRasterPos2f(labelX, 0.01f);
    for (const char* p = g_filenames.back().c_str(); *p; p++)
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_10, *p);

    glutSwapBuffers();
}

static void displayTS_dispatch() {
    int win = glutGetWindow();
    for (size_t i = 0; i < g_winTS.size(); i++) {
        if (g_winTS[i] == win) {
            drawTSForBand(g_tsWindowBandMap[i]);
            return;
        }
    }
}

/* ──────────────────────────── refresh all windows ────────────────── */

static void refreshAll() {
    buildRGBTexture();

    glutSetWindow(g_winImage);
    uploadTexture();
    glutPostRedisplay();

    for (size_t i = 0; i < g_winTS.size(); i++) {
        glutSetWindow(g_winTS[i]);
        glutPostRedisplay();
    }
}

/* ──────────────────────────── input callbacks ────────────────────── */

static void keyboardAll(unsigned char key, int /*x*/, int /*y*/) {
    if (key == 'q' || key == 27) {
        exit(0);
    }
    if (key == 'c' || key == 'C') {
        g_clicks.clear();
        printf("cleared all squares\n");
        refreshAll();
    }
}

static void specialAll(int key, int /*x*/, int /*y*/) {
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

    // Convert window coords -> image pixel coords
    float winAspect = (float)g_dispW / g_dispH;
    float imgAspect = (float)g_imgW / g_imgH;
    float x0, y0f, x1, y1;
    if (imgAspect > winAspect) {
        x0 = 0; x1 = (float)g_dispW;
        float h = (float)g_dispW / imgAspect;
        y0f = ((float)g_dispH - h) / 2.0f;
        y1 = y0f + h;
    } else {
        y0f = 0; y1 = (float)g_dispH;
        float w = (float)g_dispH * imgAspect;
        x0 = ((float)g_dispW - w) / 2.0f;
        x1 = x0 + w;
    }

    float fx = (float)mx;
    float fy = (float)my;
    if (fx < x0 || fx > x1 || fy < y0f || fy > y1) return;

    int px = (int)((fx - x0) / (x1 - x0) * g_imgW);
    int py = (int)((fy - y0f) / (y1 - y0f) * g_imgH);
    px = std::max(0, std::min(px, g_imgW - 1));
    py = std::max(0, std::min(py, g_imgH - 1));

    g_clicks.push_back({px, py});
    printf("click #%d at pixel (%d, %d)\n", (int)g_clicks.size(), px, py);
    refreshAll();
}

static void reshapeImage(int w, int h) {
    g_dispW = w; g_dispH = h;
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(-1, 1, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

static void reshapeTS(int w, int h) {
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, 1, 0, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

/* ──────────────────────────── parallel file loading ──────────────── */

struct LoadTask {
    std::string path;
    BandImage   result;
    int         ok;
};

static void* loadWorker(void* arg) {
    LoadTask* task = (LoadTask*)arg;
    task->ok = 0;

    GDALDataset* ds = (GDALDataset*)GDALOpen(task->path.c_str(), GA_ReadOnly);
    if (!ds) {
        fprintf(stderr, "GDAL: cannot open %s\n", task->path.c_str());
        return nullptr;
    }

    int w  = ds->GetRasterXSize();
    int h  = ds->GetRasterYSize();
    int nb = ds->GetRasterCount();

    task->result.width  = w;
    task->result.height = h;
    task->result.nBands = nb;
    task->result.bands.resize(nb);

    for (int b = 0; b < nb; b++) {
        task->result.bands[b].resize(w * h);
        GDALRasterBand* band = ds->GetRasterBand(b + 1);
        CPLErr err = band->RasterIO(GF_Read, 0, 0, w, h,
                                    task->result.bands[b].data(),
                                    w, h, GDT_Float32, 0, 0);
        if (err != CE_None) {
            fprintf(stderr, "GDAL: RasterIO error band %d in %s\n",
                    b + 1, task->path.c_str());
            GDALClose(ds);
            return nullptr;
        }
    }
    GDALClose(ds);
    task->ok = 1;
    return nullptr;
}

/* ──────────────────────────── main ───────────────────────────────── */

int main(int argc, char** argv) {
    std::string dir = ".";
    std::string ext = "tif";
    int startIdx    = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-d") && i + 1 < argc)      { dir = argv[++i]; }
        else if (!strcmp(argv[i], "-w") && i + 1 < argc)  { g_squareWidth = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "-i") && i + 1 < argc)  { startIdx = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "-e") && i + 1 < argc)  { ext = argv[++i]; }
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            printf(
                "Usage: %s [-d dir] [-w square_width] [-i index] [-e ext]\n"
                "\n"
                "  -d <dir>   Directory of GDAL-readable raster files  (default: .)\n"
                "  -w <int>   Sampling square side-length in pixels     (default: 10)\n"
                "  -i <int>   Initial image index (0-based, -1=last)   (default: 0)\n"
                "  -e <ext>   File extension to filter for             (default: tif)\n"
                "\n"
                "Controls (work in every window):\n"
                "  Left/Right arrows  Navigate images forward/backward in time\n"
                "  Left-click image   Add sampling square, update time series\n"
                "  Right-click image  Clear all squares and time-series traces\n"
                "  c                  Clear all squares and traces\n"
                "  q / Escape         Quit\n",
                argv[0]);
            return 0;
        }
    }

    GDALAllRegister();

    // Scan directory for files matching extension
    {
        std::string dotExt = "." + ext;
        char cmd[1024];
        snprintf(cmd, sizeof(cmd), "ls -1 \"%s\" 2>/dev/null", dir.c_str());
        FILE* pipe = popen(cmd, "r");
        if (!pipe) { fprintf(stderr, "cannot list directory %s\n", dir.c_str()); return 1; }
        char line[512];
        while (fgets(line, sizeof(line), pipe)) {
            size_t len = strlen(line);
            while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r'))
                line[--len] = 0;
            std::string fname(line);
            if (fname.size() > dotExt.size() &&
                fname.compare(fname.size() - dotExt.size(), dotExt.size(), dotExt) == 0) {
                g_filenames.push_back(fname);
            }
        }
        pclose(pipe);
    }
    std::sort(g_filenames.begin(), g_filenames.end());

    if (g_filenames.empty()) {
        fprintf(stderr, "no .%s files found in %s\n", ext.c_str(), dir.c_str());
        return 1;
    }
    printf("found %zu .%s files in %s\n", g_filenames.size(), ext.c_str(), dir.c_str());

    // Parallel load with pthreads
    int nFiles = (int)g_filenames.size();
    std::vector<LoadTask> tasks(nFiles);
    for (int i = 0; i < nFiles; i++)
        tasks[i].path = dir + "/" + g_filenames[i];

    int nCPU    = (int)sysconf(_SC_NPROCESSORS_ONLN);
    int nWorkers = std::min(nFiles, std::max(1, nCPU));
    printf("loading %d files with up to %d threads...\n", nFiles, nWorkers);

    std::vector<pthread_t> threads(nWorkers);
    for (int base = 0; base < nFiles; base += nWorkers) {
        int batch = std::min(nWorkers, nFiles - base);
        for (int j = 0; j < batch; j++)
            pthread_create(&threads[j], nullptr, loadWorker, &tasks[base + j]);
        for (int j = 0; j < batch; j++)
            pthread_join(threads[j], nullptr);
    }

    // Collect successful loads
    g_filenames.clear();
    for (int i = 0; i < nFiles; i++) {
        if (!tasks[i].ok) {
            fprintf(stderr, "warning: failed to load %s, skipping\n",
                    tasks[i].path.c_str());
            continue;
        }
        tasks[i].result.filename = tasks[i].path.substr(tasks[i].path.rfind('/') + 1);
        g_images.push_back(std::move(tasks[i].result));
        g_filenames.push_back(g_images.back().filename);
    }

    if (g_images.empty()) {
        fprintf(stderr, "no files loaded successfully\n");
        return 1;
    }

    g_imgW   = g_images[0].width;
    g_imgH   = g_images[0].height;
    g_nBands = g_images[0].nBands;

    if (startIdx < 0) startIdx = (int)g_images.size() + startIdx;
    startIdx = std::max(0, std::min(startIdx, (int)g_images.size() - 1));
    g_curIdx = startIdx;

    printf("image: %d x %d, %d bands, %zu dates\n", g_imgW, g_imgH, g_nBands, g_images.size());
    for (size_t i = 0; i < g_filenames.size(); i++) {
        const char* mark = ((int)i == g_curIdx) ? " <-" : "";
        printf("  [%zu] %s%s\n", i, g_filenames[i].c_str(), mark);
    }
    printf("controls: left/right=navigate, left-click=add square, right-click/c=clear, q=quit\n");

    // Init GLUT
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);

    // Compute reasonable image window size
    g_dispW = std::min(g_imgW, 1200);
    g_dispH = (int)((float)g_dispW / g_imgW * g_imgH);
    if (g_dispH > 900) {
        g_dispH = 900;
        g_dispW = (int)((float)g_dispH / g_imgH * g_imgW);
    }

    // --- Image window ---
    glutInitWindowSize(g_dispW, g_dispH);
    glutInitWindowPosition(50, 50);
    g_winImage = glutCreateWindow("Band Viewer");
    glutDisplayFunc(displayImage);
    glutReshapeFunc(reshapeImage);
    glutMouseFunc(mouseImage);
    glutKeyboardFunc(keyboardAll);
    glutSpecialFunc(specialAll);

    buildRGBTexture();
    uploadTexture();

    // --- One TS window per band ---
    int tsW = 480, tsH = 220;
    for (int b = 0; b < g_nBands && b < MAX_TS_WINDOWS; b++) {
        int col = b / 4;
        int row = b % 4;
        glutInitWindowSize(tsW, tsH);
        glutInitWindowPosition(60 + g_dispW + 20 + col * (tsW + 10),
                               50 + row * (tsH + 50));
        char name[64];
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



/*
 * zprdemo.cu - 3D Point Cloud Viewer with ZPR (Zoom-Pan-Rotate)
 * 
 * Usage: ./zprdemo field_x field_y field_z
 * 
 * Loads all obs/*_OBS.csv files automatically
 * Arrow keys: Left/Right to cycle through years
 * 
 * Compile with: nvcc -O3 -o zprdemo zprdemo.cu -lGL -lGLU -lglut -lGLEW -lm
 * 
 * Original ZPR module by Nigel Stewart, RMIT University
 */

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <set>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <dirent.h>

#include <cuda_runtime.h>  // For float3, float4, make_float3

// GLEW must be included before any other OpenGL headers
#include <GL/glew.h>

#ifdef __APPLE__
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#include <GLUT/glut.h>
#else
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>
#endif

#define MAXBUFFERSIZE 1000000
#define STR_MAX 1000
#define MYFONT GLUT_BITMAP_HELVETICA_12
#define STARTX 800
#define STARTY 800

// Per-file data structure
struct FileData {
    std::string filename;
    GLuint vbo;
    int numPoints;
    int startIndex;  // Global start index for picking
    std::vector<std::string> csvLines;
    std::string csvHeader;
    std::vector<float4> points;  // Temporary storage before VBO creation
};

// All loaded files
std::vector<FileData> allFiles;
int totalPoints = 0;

// Currently displayed buffer indices (3 buffers: red, green, blue)
int bufferIndex[3] = {0, 1, 2};

// Legacy single-file variables (kept for compatibility)
GLuint vbo = 0;
int numPoints = 0;

float3 dataMin, dataMax, dataCenter;
float dataScale = 1.0f;

// Store original CSV lines for picking output
std::vector<std::string> csvLines;
std::string csvHeader;

// Special points data
GLuint vboSpecial = 0;
int numSpecialPoints = 0;
std::vector<std::string> specialCsvLines;
std::string specialCsvHeader;

// Field names for axis labels
std::string fieldNameX, fieldNameY, fieldNameZ;

// Column indices for extracting values
int colIdxX = -1, colIdxY = -1, colIdxZ = -1;

// Global state
int WINDOWX = 800;
int WINDOWY = 800;

int _mouseX = 0;
int _mouseY = 0;
int _mouseLeft = 0;
int _mouseMiddle = 0;
int _mouseRight = 0;

double _dragPosX = 0.0;
double _dragPosY = 0.0;
double _dragPosZ = 0.0;
double _left = -1.0;
double _right = 1.0;
double _top = 1.0;
double _bottom = -1.0;
double _zNear = -1.0;
double _zFar = 1.0;

std::set<GLint> myPickNames;

double _matrix[16];
double _matrixInverse[16];
GLfloat zprReferencePoint[4] = {0.0f, 0.0f, 0.0f, 0.0f};

char console_string[STR_MAX];
int console_position = 0;
int renderflag = 0;
int fullscreen = 0;

// Function pointers for callbacks
void (*selection)(void) = NULL;
void (*pick)(GLint name) = NULL;

// Lighting parameters
static GLfloat light_ambient[] = {0.0, 0.0, 0.0, 1.0};
static GLfloat light_diffuse[] = {1.0, 1.0, 1.0, 1.0};
static GLfloat light_specular[] = {1.0, 1.0, 1.0, 1.0};
static GLfloat light_position[] = {1.0, 1.0, 1.0, 0.0};

static GLfloat mat_ambient[] = {0.7, 0.7, 0.7, 1.0};
static GLfloat mat_diffuse[] = {0.8, 0.8, 0.8, 1.0};
static GLfloat mat_specular[] = {1.0, 1.0, 1.0, 1.0};
static GLfloat high_shininess[] = {100.0};

// CSV parsing functions
std::vector<std::string> splitCSVLine(const std::string &line) {
    std::vector<std::string> result;
    std::stringstream ss(line);
    std::string item;
    while (std::getline(ss, item, ',')) {
        // Trim whitespace
        size_t start = item.find_first_not_of(" \t\r\n");
        size_t end = item.find_last_not_of(" \t\r\n");
        if (start != std::string::npos) {
            std::string val = item.substr(start, end - start + 1);
            // Strip surrounding quotes if present
            if (val.length() >= 2 && val[0] == '"' && val[val.length()-1] == '"') {
                val = val.substr(1, val.length() - 2);
            }
            result.push_back(val);
        } else {
            result.push_back("");
        }
    }
    return result;
}

// Generate cache filename based on field names
std::string getCacheBasename(const char *field1, const char *field2, const char *field3) {
    std::string base = "cache_";
    base += field1;
    base += "_";
    base += field2;
    base += "_";
    base += field3;
    return base;
}

// Save bounds to cache file
bool saveBoundsCache(const char *field1, const char *field2, const char *field3) {
    std::string filename = getCacheBasename(field1, field2, field3) + "_bounds.bin";
    FILE *f = fopen(filename.c_str(), "wb");
    if (!f) return false;
    
    fwrite(&dataMin.x, sizeof(float), 1, f);
    fwrite(&dataMin.y, sizeof(float), 1, f);
    fwrite(&dataMin.z, sizeof(float), 1, f);
    fwrite(&dataMax.x, sizeof(float), 1, f);
    fwrite(&dataMax.y, sizeof(float), 1, f);
    fwrite(&dataMax.z, sizeof(float), 1, f);
    fwrite(&colIdxX, sizeof(int), 1, f);
    fwrite(&colIdxY, sizeof(int), 1, f);
    fwrite(&colIdxZ, sizeof(int), 1, f);
    fclose(f);
    
    printf("Saved bounds cache to %s\n", filename.c_str());
    return true;
}

// Load bounds from cache file
bool loadBoundsCache(const char *field1, const char *field2, const char *field3) {
    std::string filename = getCacheBasename(field1, field2, field3) + "_bounds.bin";
    FILE *f = fopen(filename.c_str(), "rb");
    if (!f) return false;
    
    bool ok = true;
    ok = ok && fread(&dataMin.x, sizeof(float), 1, f) == 1;
    ok = ok && fread(&dataMin.y, sizeof(float), 1, f) == 1;
    ok = ok && fread(&dataMin.z, sizeof(float), 1, f) == 1;
    ok = ok && fread(&dataMax.x, sizeof(float), 1, f) == 1;
    ok = ok && fread(&dataMax.y, sizeof(float), 1, f) == 1;
    ok = ok && fread(&dataMax.z, sizeof(float), 1, f) == 1;
    ok = ok && fread(&colIdxX, sizeof(int), 1, f) == 1;
    ok = ok && fread(&colIdxY, sizeof(int), 1, f) == 1;
    ok = ok && fread(&colIdxZ, sizeof(int), 1, f) == 1;
    fclose(f);
    
    if (ok) {
        printf("Loaded bounds cache from %s\n", filename.c_str());
        printf("  X[%.3f, %.3f] Y[%.3f, %.3f] Z[%.3f, %.3f]\n",
               dataMin.x, dataMax.x, dataMin.y, dataMax.y, dataMin.z, dataMax.z);
    }
    return ok;
}

// Save single file data to binary cache
bool saveFileCache(const FileData &fd, const char *field1, const char *field2, const char *field3) {
    std::string basename = getBasename(fd.filename);
    std::string filename = getCacheBasename(field1, field2, field3) + "_" + basename + ".bin";
    FILE *f = fopen(filename.c_str(), "wb");
    if (!f) return false;
    
    int numPts = fd.points.size();
    fwrite(&numPts, sizeof(int), 1, f);
    fwrite(fd.points.data(), sizeof(float4), numPts, f);
    
    // Save CSV lines count and data
    int numLines = fd.csvLines.size();
    fwrite(&numLines, sizeof(int), 1, f);
    for (int i = 0; i < numLines; i++) {
        int len = fd.csvLines[i].length();
        fwrite(&len, sizeof(int), 1, f);
        fwrite(fd.csvLines[i].c_str(), 1, len, f);
    }
    
    // Save header
    int headerLen = fd.csvHeader.length();
    fwrite(&headerLen, sizeof(int), 1, f);
    fwrite(fd.csvHeader.c_str(), 1, headerLen, f);
    
    fclose(f);
    return true;
}

// Load single file data from binary cache
bool loadFileCache(const std::string &originalFilename, const char *field1, const char *field2, const char *field3, FileData &fd) {
    std::string basename = getBasename(originalFilename);
    std::string filename = getCacheBasename(field1, field2, field3) + "_" + basename + ".bin";
    FILE *f = fopen(filename.c_str(), "rb");
    if (!f) return false;
    
    fd.filename = originalFilename;
    
    int numPts = 0;
    if (fread(&numPts, sizeof(int), 1, f) != 1) { fclose(f); return false; }
    
    fd.points.resize(numPts);
    if (fread(fd.points.data(), sizeof(float4), numPts, f) != (size_t)numPts) { fclose(f); return false; }
    fd.numPoints = numPts;
    
    int numLines = 0;
    if (fread(&numLines, sizeof(int), 1, f) != 1) { fclose(f); return false; }
    fd.csvLines.resize(numLines);
    for (int i = 0; i < numLines; i++) {
        int len = 0;
        if (fread(&len, sizeof(int), 1, f) != 1) { fclose(f); return false; }
        fd.csvLines[i].resize(len);
        if (fread(&fd.csvLines[i][0], 1, len, f) != (size_t)len) { fclose(f); return false; }
    }
    
    int headerLen = 0;
    if (fread(&headerLen, sizeof(int), 1, f) != 1) { fclose(f); return false; }
    fd.csvHeader.resize(headerLen);
    if (fread(&fd.csvHeader[0], 1, headerLen, f) != (size_t)headerLen) { fclose(f); return false; }
    
    fd.vbo = 0;
    fclose(f);
    return true;
}

int findColumnIndex(const std::vector<std::string> &headers, const char *fieldName) {
    for (size_t i = 0; i < headers.size(); i++) {
        if (headers[i] == fieldName) {
            return (int)i;
        }
    }
    return -1;
}

// First pass: scan all files to determine global bounds
bool scanAllFilesForBounds(const std::vector<std::string> &filenames, 
                           const char *field1, const char *field2, const char *field3) {
    printf("Scanning %zu files for bounds with 1%% histogram trimming...\n", filenames.size());
    
    // Accumulators for averaging trimmed bounds across files
    double sumMinX = 0, sumMaxX = 0;
    double sumMinY = 0, sumMaxY = 0;
    double sumMinZ = 0, sumMaxZ = 0;
    int numFilesProcessed = 0;
    
    for (size_t f = 0; f < filenames.size(); f++) {
        std::ifstream file(filenames[f]);
        if (!file.is_open()) continue;
        
        std::string line;
        if (!std::getline(file, line)) continue;
        
        std::vector<std::string> headers = splitCSVLine(line);
        int idx1 = findColumnIndex(headers, field1);
        int idx2 = findColumnIndex(headers, field2);
        int idx3 = findColumnIndex(headers, field3);
        
        if (idx1 < 0 || idx2 < 0 || idx3 < 0) continue;
        
        // Store indices on first successful file
        if (colIdxX < 0) {
            colIdxX = idx1;
            colIdxY = idx2;
            colIdxZ = idx3;
        }
        
        // Collect all values for this file
        std::vector<float> valuesX, valuesY, valuesZ;
        
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            std::vector<std::string> values = splitCSVLine(line);
            
            bool hasTime12 = false;
            for (size_t i = 0; i < values.size(); i++) {
                if (values[i].length() == 10 && values[i].substr(8, 2) == "12") {
                    hasTime12 = true;
                    break;
                }
            }
            if (!hasTime12) continue;
            
            int maxIdx = std::max({idx1, idx2, idx3});
            if ((int)values.size() <= maxIdx) continue;
            if (values[idx1].empty() || values[idx2].empty() || values[idx3].empty()) continue;
            
            try {
                valuesX.push_back(std::stof(values[idx1]));
                valuesY.push_back(std::stof(values[idx2]));
                valuesZ.push_back(std::stof(values[idx3]));
            } catch (...) {}
        }
        file.close();
        
        if (valuesX.empty()) continue;
        
        // Sort each dimension
        std::sort(valuesX.begin(), valuesX.end());
        std::sort(valuesY.begin(), valuesY.end());
        std::sort(valuesZ.begin(), valuesZ.end());
        
        // Calculate 1% trimmed indices
        size_t n = valuesX.size();
        size_t trimIdx = (size_t)(n * 0.01);
        size_t lowIdx = trimIdx;
        size_t highIdx = n - 1 - trimIdx;
        if (highIdx <= lowIdx) {
            lowIdx = 0;
            highIdx = n - 1;
        }
        
        // Accumulate trimmed bounds
        sumMinX += valuesX[lowIdx];
        sumMaxX += valuesX[highIdx];
        sumMinY += valuesY[lowIdx];
        sumMaxY += valuesY[highIdx];
        sumMinZ += valuesZ[lowIdx];
        sumMaxZ += valuesZ[highIdx];
        numFilesProcessed++;
        
        printf("\r  Scanned %zu/%zu files", f + 1, filenames.size());
        fflush(stdout);
    }
    printf("\n");
    
    if (numFilesProcessed == 0) {
        printf("No valid files found!\n");
        dataMin = make_float3(0, 0, 0);
        dataMax = make_float3(1, 1, 1);
        return false;
    }
    
    // Average the trimmed bounds across all files
    dataMin.x = (float)(sumMinX / numFilesProcessed);
    dataMax.x = (float)(sumMaxX / numFilesProcessed);
    dataMin.y = (float)(sumMinY / numFilesProcessed);
    dataMax.y = (float)(sumMaxY / numFilesProcessed);
    dataMin.z = (float)(sumMinZ / numFilesProcessed);
    dataMax.z = (float)(sumMaxZ / numFilesProcessed);
    
    printf("Global bounds (1%% trimmed, averaged over %d files):\n", numFilesProcessed);
    printf("  X[%.3f, %.3f] Y[%.3f, %.3f] Z[%.3f, %.3f]\n",
           dataMin.x, dataMax.x, dataMin.y, dataMax.y, dataMin.z, dataMax.z);
    return true;
}

// Load a single file with pre-computed global bounds
bool loadSingleFile(const std::string &filename, const char *field1, const char *field2, const char *field3,
                    FileData &fileData) {
    std::ifstream file(filename);
    if (!file.is_open()) return false;
    
    fileData.filename = filename;
    fileData.numPoints = 0;
    fileData.csvLines.clear();
    
    std::string line;
    if (!std::getline(file, line)) return false;
    
    fileData.csvHeader = line;
    
    std::vector<std::string> headers = splitCSVLine(line);
    int idx1 = findColumnIndex(headers, field1);
    int idx2 = findColumnIndex(headers, field2);
    int idx3 = findColumnIndex(headers, field3);
    
    if (idx1 < 0 || idx2 < 0 || idx3 < 0) return false;
    
    float rangeX = dataMax.x - dataMin.x;
    float rangeY = dataMax.y - dataMin.y;
    float rangeZ = dataMax.z - dataMin.z;
    
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::vector<std::string> values = splitCSVLine(line);
        
        bool hasTime12 = false;
        for (size_t i = 0; i < values.size(); i++) {
            if (values[i].length() == 10 && values[i].substr(8, 2) == "12") {
                hasTime12 = true;
                break;
            }
        }
        if (!hasTime12) continue;
        
        int maxIdx = std::max({idx1, idx2, idx3});
        if ((int)values.size() <= maxIdx) continue;
        if (values[idx1].empty() || values[idx2].empty() || values[idx3].empty()) continue;
        
        try {
            float4 pt;
            pt.x = std::stof(values[idx1]);
            pt.y = std::stof(values[idx2]);
            pt.z = std::stof(values[idx3]);
            pt.w = 1.0f;
            
            // Normalize using global (trimmed) bounds - NO clipping
            // Values outside trimmed bounds will be <0 or >1
            pt.x = (rangeX > 0) ? (pt.x - dataMin.x) / rangeX : 0.5f;
            pt.y = (rangeY > 0) ? (pt.y - dataMin.y) / rangeY : 0.5f;
            pt.z = (rangeZ > 0) ? (pt.z - dataMin.z) / rangeZ : 0.5f;
            
            fileData.points.push_back(pt);
            fileData.csvLines.push_back(line);
        } catch (...) {}
    }
    file.close();
    
    if (fileData.points.empty()) return false;
    
    fileData.numPoints = fileData.points.size();
    fileData.vbo = 0;  // VBO will be created later after GL init
    return true;
}

// Create VBO for a FileData (must be called after GL context is ready)
void createFileVBO(FileData &fd) {
    if (fd.points.empty() || fd.vbo != 0) return;
    
    glGenBuffers(1, &fd.vbo);
    glBindBuffer(GL_ARRAY_BUFFER, fd.vbo);
    glBufferData(GL_ARRAY_BUFFER, fd.points.size() * sizeof(float4), fd.points.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    
    // Free CPU memory after uploading to GPU
    fd.points.clear();
    fd.points.shrink_to_fit();
}

// Legacy single-file load function (kept for compatibility)
bool loadCSV(const char *filename, const char *field1, const char *field2, const char *field3,
             std::vector<float4> &points) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        fprintf(stderr, "Error: Cannot open file %s\n", filename);
        return false;
    }
    
    // Store field names for axis labels
    fieldNameX = field1;
    fieldNameY = field2;
    fieldNameZ = field3;

    std::string line;
    
    // Read header
    if (!std::getline(file, line)) {
        fprintf(stderr, "Error: Empty file\n");
        return false;
    }
    
    csvHeader = line;  // Store header for output
    
    std::vector<std::string> headers = splitCSVLine(line);
    
    int idx1 = findColumnIndex(headers, field1);
    int idx2 = findColumnIndex(headers, field2);
    int idx3 = findColumnIndex(headers, field3);
    
    if (idx1 < 0) { fprintf(stderr, "Error: Field '%s' not found\n", field1); return false; }
    if (idx2 < 0) { fprintf(stderr, "Error: Field '%s' not found\n", field2); return false; }
    if (idx3 < 0) { fprintf(stderr, "Error: Field '%s' not found\n", field3); return false; }
    
    // Store indices globally for pick callback
    colIdxX = idx1;
    colIdxY = idx2;
    colIdxZ = idx3;
    
    printf("Found fields: %s(col %d), %s(col %d), %s(col %d)\n", 
           field1, idx1, field2, idx2, field3, idx3);
    
    // Initialize bounds
    dataMin = make_float3(1e30f, 1e30f, 1e30f);
    dataMax = make_float3(-1e30f, -1e30f, -1e30f);
    
    // Counters for diagnostics
    int linesRead = 0;
    int linesMatchedTime = 0;
    int linesWithData = 0;
    
    // Read data
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        linesRead++;
        
        std::vector<std::string> values = splitCSVLine(line);
        
        // Filter: only keep lines where a datetime field ends with "12" (noon hour)
        // This is when daily summary fields (FWI, ISI, FFMC) are populated
        // Format is YYYYMMDDHH like "2023060612"
        bool hasTime12 = false;
        for (size_t i = 0; i < values.size(); i++) {
            const std::string &v = values[i];
            // Check for YYYYMMDDHH format (10 digits) ending in 12
            if (v.length() == 10 && v.substr(8, 2) == "12") {
                hasTime12 = true;
                break;
            }
        }
        if (!hasTime12) continue;
        linesMatchedTime++;
        
        int maxIdx = std::max({idx1, idx2, idx3});
        if ((int)values.size() <= maxIdx) continue;
        
        // Skip rows where the data fields are empty
        if (values[idx1].empty() || values[idx2].empty() || values[idx3].empty()) continue;
        linesWithData++;
        
        try {
            float4 pt;
            pt.x = std::stof(values[idx1]);
            pt.y = std::stof(values[idx2]);
            pt.z = std::stof(values[idx3]);
            pt.w = 1.0f; // For alignment and potential color/size data
            
            // Update bounds
            dataMin.x = fminf(dataMin.x, pt.x);
            dataMin.y = fminf(dataMin.y, pt.y);
            dataMin.z = fminf(dataMin.z, pt.z);
            dataMax.x = fmaxf(dataMax.x, pt.x);
            dataMax.y = fmaxf(dataMax.y, pt.y);
            dataMax.z = fmaxf(dataMax.z, pt.z);
            
            points.push_back(pt);
            csvLines.push_back(line);  // Store original line for picking
        } catch (...) {
            // Skip invalid lines
            continue;
        }
    }
    
    file.close();
    
    printf("CSV reading: %d total lines, %d matched time filter, %d had valid data\n",
           linesRead, linesMatchedTime, linesWithData);
    printf("Loaded %zu points\n", points.size());
    printf("Raw bounds: X[%.3f, %.3f] Y[%.3f, %.3f] Z[%.3f, %.3f]\n",
           dataMin.x, dataMax.x, dataMin.y, dataMax.y, dataMin.z, dataMax.z);
    
    // Normalize points to 0-1 range for each axis independently
    float rangeX = dataMax.x - dataMin.x;
    float rangeY = dataMax.y - dataMin.y;
    float rangeZ = dataMax.z - dataMin.z;
    
    for (size_t i = 0; i < points.size(); i++) {
        points[i].x = (rangeX > 0) ? (points[i].x - dataMin.x) / rangeX : 0.5f;
        points[i].y = (rangeY > 0) ? (points[i].y - dataMin.y) / rangeY : 0.5f;
        points[i].z = (rangeZ > 0) ? (points[i].z - dataMin.z) / rangeZ : 0.5f;
    }
    
    printf("Points normalized to [0,1] range on each axis\n");
    
    // Set center and scale for normalized data
    dataCenter = make_float3(0.5f, 0.5f, 0.5f);
    dataScale = 1.0f;
    
    return points.size() > 0;
}

void initVBO(const std::vector<float4> &points) {
    numPoints = points.size();
    
    // Clear any existing GL errors
    while (glGetError() != GL_NO_ERROR);
    
    // Create VBO - data is already normalized to [0,1] in loadCSV
    glGenBuffers(1, &vbo);
    GLenum glErr = glGetError();
    if (glErr != GL_NO_ERROR) {
        fprintf(stderr, "GL error after glGenBuffers: 0x%x\n", glErr);
        exit(1);
    }
    
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, numPoints * sizeof(float4), points.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    
    glErr = glGetError();
    if (glErr != GL_NO_ERROR) {
        fprintf(stderr, "GL error after buffer setup: 0x%x\n", glErr);
        exit(1);
    }
    
    printf("VBO initialized with %d points\n", numPoints);
}

// Load special.csv if it exists
bool loadSpecialCSV(const char *field1, const char *field2, const char *field3,
                    std::vector<float4> &points) {
    std::ifstream file("special.csv");
    if (!file.is_open()) {
        printf("No special.csv found (optional)\n");
        return false;
    }

    std::string line;
    
    // Read header
    if (!std::getline(file, line)) {
        return false;
    }
    
    specialCsvHeader = line;
    
    std::vector<std::string> headers = splitCSVLine(line);
    
    int idx1 = findColumnIndex(headers, field1);
    int idx2 = findColumnIndex(headers, field2);
    int idx3 = findColumnIndex(headers, field3);
    
    if (idx1 < 0 || idx2 < 0 || idx3 < 0) {
        printf("special.csv does not contain required fields, skipping\n");
        return false;
    }
    
    printf("Loading special.csv with fields: %s(col %d), %s(col %d), %s(col %d)\n", 
           field1, idx1, field2, idx2, field3, idx3);
    
    // Read data
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        std::vector<std::string> values = splitCSVLine(line);
        
        int maxIdx = std::max({idx1, idx2, idx3});
        if ((int)values.size() <= maxIdx) continue;
        
        if (values[idx1].empty() || values[idx2].empty() || values[idx3].empty()) continue;

        try {
            float4 pt;
            pt.x = std::stof(values[idx1]);
            pt.y = std::stof(values[idx2]);
            pt.z = std::stof(values[idx3]);
            pt.w = 1.0f;
            
            points.push_back(pt);
            specialCsvLines.push_back(line);
        } catch (...) {
            continue;
        }
    }
    
    file.close();
    printf("Loaded %zu special points from special.csv\n", points.size());
    
    return points.size() > 0;
}

// Transform special points using the same scale as main data (trimmed bounds, no clipping)
void transformSpecialPoints(std::vector<float4> &points) {
    float rangeX = dataMax.x - dataMin.x;
    float rangeY = dataMax.y - dataMin.y;
    float rangeZ = dataMax.z - dataMin.z;
    
    for (size_t i = 0; i < points.size(); i++) {
        // NO clipping - values outside trimmed bounds will be <0 or >1
        points[i].x = (rangeX > 0) ? (points[i].x - dataMin.x) / rangeX : 0.5f;
        points[i].y = (rangeY > 0) ? (points[i].y - dataMin.y) / rangeY : 0.5f;
        points[i].z = (rangeZ > 0) ? (points[i].z - dataMin.z) / rangeZ : 0.5f;
    }
}

void initSpecialVBO(const std::vector<float4> &points) {
    numSpecialPoints = points.size();
    if (numSpecialPoints == 0) return;
    
    glGenBuffers(1, &vboSpecial);
    glBindBuffer(GL_ARRAY_BUFFER, vboSpecial);
    glBufferData(GL_ARRAY_BUFFER, numSpecialPoints * sizeof(float4), points.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    
    printf("Special VBO initialized with %d points\n", numSpecialPoints);
}

void cleanupVBO() {
    if (vbo) {
        glDeleteBuffers(1, &vbo);
        vbo = 0;
    }
    if (vboSpecial) {
        glDeleteBuffers(1, &vboSpecial);
        vboSpecial = 0;
    }
}

// Helper to extract just the filename from path
std::string getBasename(const std::string &path) {
    size_t pos = path.find_last_of("/\\");
    if (pos != std::string::npos) return path.substr(pos + 1);
    return path;
}

// Draw labeled axes
void drawAxes() {
    glDisable(GL_LIGHTING);
    glLineWidth(2.0f);
    
    // X axis - Red
    glColor3f(1.0f, 0.0f, 0.0f);
    glBegin(GL_LINES);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(1.0f, 0.0f, 0.0f);
    glEnd();
    
    // Y axis - Green
    glColor3f(0.0f, 1.0f, 0.0f);
    glBegin(GL_LINES);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, 1.0f, 0.0f);
    glEnd();
    
    // Z axis - Blue
    glColor3f(0.0f, 0.0f, 1.0f);
    glBegin(GL_LINES);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, 1.0f);
    glEnd();
    
    // Draw axis labels with field names
    glColor3f(1.0f, 0.3f, 0.3f);
    glRasterPos3f(1.05f, 0.0f, 0.0f);
    for (const char *c = fieldNameX.c_str(); *c; c++) {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_10, *c);
    }
    
    glColor3f(0.3f, 1.0f, 0.3f);
    glRasterPos3f(0.0f, 1.05f, 0.0f);
    for (const char *c = fieldNameY.c_str(); *c; c++) {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_10, *c);
    }
    
    glColor3f(0.3f, 0.3f, 1.0f);
    glRasterPos3f(0.0f, 0.0f, 1.05f);
    for (const char *c = fieldNameZ.c_str(); *c; c++) {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_10, *c);
    }
    
    // Draw min/max values on axes
    char buf[32];
    
    // X axis min (at origin) and max (at 1.0)
    glColor3f(1.0f, 0.3f, 0.3f);
    snprintf(buf, sizeof(buf), "%.1f", dataMin.x);
    glRasterPos3f(-0.02f, -0.05f, 0.0f);
    for (const char *c = buf; *c; c++) {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_10, *c);
    }
    snprintf(buf, sizeof(buf), "%.1f", dataMax.x);
    glRasterPos3f(0.95f, -0.05f, 0.0f);
    for (const char *c = buf; *c; c++) {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_10, *c);
    }
    
    // Y axis min (at origin) and max (at 1.0)
    glColor3f(0.3f, 1.0f, 0.3f);
    snprintf(buf, sizeof(buf), "%.1f", dataMin.y);
    glRasterPos3f(-0.08f, 0.0f, 0.0f);
    for (const char *c = buf; *c; c++) {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_10, *c);
    }
    snprintf(buf, sizeof(buf), "%.1f", dataMax.y);
    glRasterPos3f(-0.08f, 1.0f, 0.0f);
    for (const char *c = buf; *c; c++) {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_10, *c);
    }
    
    // Z axis min (at origin) and max (at 1.0)
    glColor3f(0.3f, 0.3f, 1.0f);
    snprintf(buf, sizeof(buf), "%.1f", dataMin.z);
    glRasterPos3f(0.0f, -0.05f, 0.0f);
    for (const char *c = buf; *c; c++) {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_10, *c);
    }
    snprintf(buf, sizeof(buf), "%.1f", dataMax.z);
    glRasterPos3f(0.0f, -0.05f, 1.0f);
    for (const char *c = buf; *c; c++) {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_10, *c);
    }
    
    // Draw selected file names near axes (if multi-file mode)
    if (!allFiles.empty()) {
        float yOffset = -0.15f;
        
        // Red buffer (index 0)
        if (bufferIndex[0] >= 0 && bufferIndex[0] < (int)allFiles.size()) {
            glColor3f(1.0f, 0.0f, 0.0f);
            glRasterPos3f(0.0f, yOffset, 0.0f);
            std::string name = getBasename(allFiles[bufferIndex[0]].filename);
            for (const char *c = name.c_str(); *c; c++) {
                glutBitmapCharacter(GLUT_BITMAP_HELVETICA_10, *c);
            }
        }
        
        // Green buffer (index 1)
        if (bufferIndex[1] >= 0 && bufferIndex[1] < (int)allFiles.size()) {
            glColor3f(0.0f, 1.0f, 0.0f);
            glRasterPos3f(0.0f, yOffset - 0.05f, 0.0f);
            std::string name = getBasename(allFiles[bufferIndex[1]].filename);
            for (const char *c = name.c_str(); *c; c++) {
                glutBitmapCharacter(GLUT_BITMAP_HELVETICA_10, *c);
            }
        }
        
        // Blue buffer (index 2)
        if (bufferIndex[2] >= 0 && bufferIndex[2] < (int)allFiles.size()) {
            glColor3f(0.0f, 0.0f, 1.0f);
            glRasterPos3f(0.0f, yOffset - 0.1f, 0.0f);
            std::string name = getBasename(allFiles[bufferIndex[2]].filename);
            for (const char *c = name.c_str(); *c; c++) {
                glutBitmapCharacter(GLUT_BITMAP_HELVETICA_10, *c);
            }
        }
    }
    
    glEnable(GL_LIGHTING);
}

// Helper to draw a single file's points with a given color
void drawFilePoints(int fileIndex, float r, float g, float b, int nameOffset) {
    if (fileIndex < 0 || fileIndex >= (int)allFiles.size()) return;
    FileData &fd = allFiles[fileIndex];
    if (fd.vbo == 0 || fd.numPoints == 0) return;
    
    glDisable(GL_LIGHTING);
    glEnable(GL_POINT_SMOOTH);
    glPointSize(0.6f);
    
    glEnableClientState(GL_VERTEX_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, fd.vbo);
    glVertexPointer(3, GL_FLOAT, sizeof(float4), 0);
    
    GLint renderMode;
    glGetIntegerv(GL_RENDER_MODE, &renderMode);
    
    if (renderMode == GL_SELECT) {
        float4 *pts = (float4*)glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);
        if (pts) {
            for (int i = 0; i < fd.numPoints; i++) {
                glLoadName(nameOffset + i);
                glBegin(GL_POINTS);
                glVertex3f(pts[i].x, pts[i].y, pts[i].z);
                glEnd();
            }
            glUnmapBuffer(GL_ARRAY_BUFFER);
        }
    } else {
        glColor3f(r, g, b);
        glDrawArrays(GL_POINTS, 0, fd.numPoints);
        
        // Highlight picked points from this buffer
        if (!myPickNames.empty()) {
            glPointSize(1.2f);
            glColor3f(1.0f, 1.0f, 0.0f);
            
            float4 *pts = (float4*)glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);
            if (pts) {
                glBegin(GL_POINTS);
                for (std::set<GLint>::iterator it = myPickNames.begin(); it != myPickNames.end(); ++it) {
                    int idx = *it - nameOffset;
                    if (idx >= 0 && idx < fd.numPoints) {
                        glVertex3f(pts[idx].x, pts[idx].y, pts[idx].z);
                    }
                }
                glEnd();
                glUnmapBuffer(GL_ARRAY_BUFFER);
            }
            glPointSize(0.6f);
        }
    }
    
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDisableClientState(GL_VERTEX_ARRAY);
    glEnable(GL_LIGHTING);
}

void drawPoints() {
    if (allFiles.empty()) {
        // Legacy single-file mode
        if (vbo == 0 || numPoints == 0) return;
        
        glDisable(GL_LIGHTING);
        glEnable(GL_POINT_SMOOTH);
        glPointSize(0.6f);
        
        glEnableClientState(GL_VERTEX_ARRAY);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glVertexPointer(3, GL_FLOAT, sizeof(float4), 0);
        
        GLint renderMode;
        glGetIntegerv(GL_RENDER_MODE, &renderMode);
        
        if (renderMode == GL_SELECT) {
            float4 *pts = (float4*)glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);
            if (pts) {
                for (int i = 0; i < numPoints; i++) {
                    glLoadName(i);
                    glBegin(GL_POINTS);
                    glVertex3f(pts[i].x, pts[i].y, pts[i].z);
                    glEnd();
                }
                glUnmapBuffer(GL_ARRAY_BUFFER);
            }
        } else {
            glColor3f(0.2f, 0.8f, 1.0f);
            glDrawArrays(GL_POINTS, 0, numPoints);
        }
        
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glDisableClientState(GL_VERTEX_ARRAY);
        glEnable(GL_LIGHTING);
        return;
    }
    
    // Multi-file mode: draw three selected buffers
    // Calculate name offsets for picking
    int offset0 = 0;
    int offset1 = (bufferIndex[0] >= 0 && bufferIndex[0] < (int)allFiles.size()) ? 
                   allFiles[bufferIndex[0]].numPoints : 0;
    int offset2 = offset1 + ((bufferIndex[1] >= 0 && bufferIndex[1] < (int)allFiles.size()) ? 
                   allFiles[bufferIndex[1]].numPoints : 0);
    
    // Red buffer
    drawFilePoints(bufferIndex[0], 1.0f, 0.0f, 0.0f, offset0);
    // Green buffer
    drawFilePoints(bufferIndex[1], 0.0f, 1.0f, 0.0f, offset1);
    // Blue buffer
    drawFilePoints(bufferIndex[2], 0.0f, 0.0f, 1.0f, offset2);
}

// Draw special points (red, 10x larger)
// Calculate name offset for special points (after all displayed regular points)
int getSpecialPointsNameOffset() {
    if (allFiles.empty()) return numPoints;
    
    int offset = 0;
    for (int i = 0; i < 3; i++) {
        if (bufferIndex[i] >= 0 && bufferIndex[i] < (int)allFiles.size()) {
            offset += allFiles[bufferIndex[i]].numPoints;
        }
    }
    return offset;
}

void drawSpecialPoints() {
    if (vboSpecial == 0 || numSpecialPoints == 0) return;
    
    int nameOffset = getSpecialPointsNameOffset();
    
    glDisable(GL_LIGHTING);
    glEnable(GL_POINT_SMOOTH);
    glPointSize(6.0f);  // 10x larger than regular points (0.6 * 10)
    
    glEnableClientState(GL_VERTEX_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, vboSpecial);
    glVertexPointer(3, GL_FLOAT, sizeof(float4), 0);
    
    // Check if we're in selection mode
    GLint renderMode;
    glGetIntegerv(GL_RENDER_MODE, &renderMode);
    
    if (renderMode == GL_SELECT) {
        // In selection mode: draw each special point with name offset
        float4 *pts = NULL;
        
        glBindBuffer(GL_ARRAY_BUFFER, vboSpecial);
        pts = (float4*)glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);
        
        if (pts) {
            for (int i = 0; i < numSpecialPoints; i++) {
                glLoadName(nameOffset + i);
                glBegin(GL_POINTS);
                glVertex3f(pts[i].x, pts[i].y, pts[i].z);
                glEnd();
            }
            glUnmapBuffer(GL_ARRAY_BUFFER);
        }
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    } else {
        // Normal render mode
        glColor3f(1.0f, 0.0f, 1.0f);  // Magenta color
        glDrawArrays(GL_POINTS, 0, numSpecialPoints);
        
        // Highlight picked special points
        if (!myPickNames.empty()) {
            glPointSize(12.0f);  // Even larger when picked
            glColor3f(1.0f, 1.0f, 0.0f);  // Yellow highlight
            
            float4 *pts = NULL;
            glBindBuffer(GL_ARRAY_BUFFER, vboSpecial);
            pts = (float4*)glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);
            if (pts) {
                glBegin(GL_POINTS);
                for (std::set<GLint>::iterator it = myPickNames.begin(); it != myPickNames.end(); ++it) {
                    int idx = *it - nameOffset;
                    if (idx >= 0 && idx < numSpecialPoints) {
                        glVertex3f(pts[idx].x, pts[idx].y, pts[idx].z);
                    }
                }
                glEnd();
                glUnmapBuffer(GL_ARRAY_BUFFER);
            }
            glBindBuffer(GL_ARRAY_BUFFER, 0);
        }
    }
    
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDisableClientState(GL_VERTEX_ARRAY);
    
    glEnable(GL_LIGHTING);
}

// Forward declarations
static void zprReshape(int w, int h);
static void zprMouse(int button, int state, int x, int y);
static void zprMotion(int x, int y);
static void zprPick(GLdouble x, GLdouble y, GLdouble delX, GLdouble delY);
static void getMatrix();
static void invertMatrix(const GLdouble *m, GLdouble *out);

// Utility functions
static double vlen(double x, double y, double z) {
    return sqrt(x*x + y*y + z*z);
}

static void pos(double *px, double *py, double *pz, int x, int y, const int *viewport) {
    *px = (double)(x - viewport[0]) / (double)(viewport[2]);
    *py = (double)(y - viewport[1]) / (double)(viewport[3]);
    *px = _left + (*px) * (_right - _left);
    *py = _top + (*py) * (_bottom - _top);
    *pz = _zNear;
}

static void getMatrix() {
    glGetDoublev(GL_MODELVIEW_MATRIX, _matrix);
    invertMatrix(_matrix, _matrixInverse);
}

static void invertMatrix(const GLdouble *m, GLdouble *out) {
    #define MAT(m,r,c) (m)[(c)*4+(r)]
    #define m11 MAT(m,0,0)
    #define m12 MAT(m,0,1)
    #define m13 MAT(m,0,2)
    #define m14 MAT(m,0,3)
    #define m21 MAT(m,1,0)
    #define m22 MAT(m,1,1)
    #define m23 MAT(m,1,2)
    #define m24 MAT(m,1,3)
    #define m31 MAT(m,2,0)
    #define m32 MAT(m,2,1)
    #define m33 MAT(m,2,2)
    #define m34 MAT(m,2,3)
    #define m41 MAT(m,3,0)
    #define m42 MAT(m,3,1)
    #define m43 MAT(m,3,2)
    #define m44 MAT(m,3,3)

    GLdouble det;
    GLdouble d12, d13, d23, d24, d34, d41;
    GLdouble tmp[16];

    d12 = (m31*m42 - m41*m32);
    d13 = (m31*m43 - m41*m33);
    d23 = (m32*m43 - m42*m33);
    d24 = (m32*m44 - m42*m34);
    d34 = (m33*m44 - m43*m34);
    d41 = (m34*m41 - m44*m31);

    tmp[0] =  (m22 * d34 - m23 * d24 + m24 * d23);
    tmp[1] = -(m21 * d34 + m23 * d41 + m24 * d13);
    tmp[2] =  (m21 * d24 + m22 * d41 + m24 * d12);
    tmp[3] = -(m21 * d23 - m22 * d13 + m23 * d12);

    det = m11 * tmp[0] + m12 * tmp[1] + m13 * tmp[2] + m14 * tmp[3];

    if (det != 0.0) {
        GLdouble invDet = 1.0 / det;
        tmp[0] *= invDet;
        tmp[1] *= invDet;
        tmp[2] *= invDet;
        tmp[3] *= invDet;

        tmp[4] = -(m12 * d34 - m13 * d24 + m14 * d23) * invDet;
        tmp[5] =  (m11 * d34 + m13 * d41 + m14 * d13) * invDet;
        tmp[6] = -(m11 * d24 + m12 * d41 + m14 * d12) * invDet;
        tmp[7] =  (m11 * d23 - m12 * d13 + m13 * d12) * invDet;

        d12 = m11*m22 - m21*m12;
        d13 = m11*m23 - m21*m13;
        d23 = m12*m23 - m22*m13;
        d24 = m12*m24 - m22*m14;
        d34 = m13*m24 - m23*m14;
        d41 = m14*m21 - m24*m11;

        tmp[8]  =  (m42 * d34 - m43 * d24 + m44 * d23) * invDet;
        tmp[9]  = -(m41 * d34 + m43 * d41 + m44 * d13) * invDet;
        tmp[10] =  (m41 * d24 + m42 * d41 + m44 * d12) * invDet;
        tmp[11] = -(m41 * d23 - m42 * d13 + m43 * d12) * invDet;
        tmp[12] = -(m32 * d34 - m33 * d24 + m34 * d23) * invDet;
        tmp[13] =  (m31 * d34 + m33 * d41 + m34 * d13) * invDet;
        tmp[14] = -(m31 * d24 + m32 * d41 + m34 * d12) * invDet;
        tmp[15] =  (m31 * d23 - m32 * d13 + m33 * d12) * invDet;

        memcpy(out, tmp, 16 * sizeof(GLdouble));
    }

    #undef m11
    #undef m12
    #undef m13
    #undef m14
    #undef m21
    #undef m22
    #undef m23
    #undef m24
    #undef m31
    #undef m32
    #undef m33
    #undef m34
    #undef m41
    #undef m42
    #undef m43
    #undef m44
    #undef MAT
}

// ZPR initialization
void zprInit() {
    zprReferencePoint[0] = 0.0f;
    zprReferencePoint[1] = 0.0f;
    zprReferencePoint[2] = 0.0f;
    zprReferencePoint[3] = 0.0f;
    getMatrix();

    glutReshapeFunc(zprReshape);
    glutMouseFunc(zprMouse);
    glutMotionFunc(zprMotion);
}

void zprSelectionFunc(void (*f)(void)) {
    selection = f;
}

void zprPickFunc(void (*f)(GLint name)) {
    pick = f;
}

static void zprReshape(int w, int h) {
    GLfloat ratio;
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    getMatrix();

    if (w <= h) {
        ratio = (GLfloat)h / (GLfloat)w;
        glOrtho(-1.0f, 1.0f, -1.0f * ratio, 1.0f * ratio, -1.0f, 1.0f);
        _bottom = -1.0 * ratio;
        _top = 1.0 * ratio;
    } else {
        ratio = (GLfloat)w / (GLfloat)h;
        glOrtho(-1.0f * ratio, 1.0f * ratio, -1.0f, 1.0f, -1.0f, 1.0f);
        _left = -1.0 * ratio;
        _right = 1.0 * ratio;
    }

    glMatrixMode(GL_MODELVIEW);
}

static void processHits(GLint hits, GLuint buffer[]) {
    myPickNames.clear();
    GLuint *ptr = buffer;
    
    for (int i = 0; i < hits; i++) {
        GLuint names = *ptr++;
        ptr++; // mindepth
        ptr++; // maxdepth
        for (GLuint j = 0; j < names; j++) {
            GLint name = *ptr++;
            if (name >= 0) {
                myPickNames.insert(name);
            }
        }
    }
}

static void zprPick(GLdouble x, GLdouble y, GLdouble delX, GLdouble delY) {
    GLuint buffer[MAXBUFFERSIZE];
    const int bufferSize = sizeof(buffer) / sizeof(GLuint);
    GLint viewport[4];
    GLdouble projection[16];

    glSelectBuffer(bufferSize, buffer);
    glRenderMode(GL_SELECT);
    glInitNames();
    glPushName(0);  // Push initial name so glLoadName works

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glGetIntegerv(GL_VIEWPORT, viewport);
    glGetDoublev(GL_PROJECTION_MATRIX, projection);
    glLoadIdentity();
    gluPickMatrix(x, y, delX, delY, viewport);
    glMultMatrixd(projection);

    glMatrixMode(GL_MODELVIEW);

    if (selection) {
        selection();
    }

    GLint hits = glRenderMode(GL_RENDER);
    processHits(hits, buffer);

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
}

static void zprMouse(int button, int state, int x, int y) {
    GLint viewport[4];

    if (state == GLUT_DOWN && button == GLUT_LEFT_BUTTON) {
        zprPick(x, glutGet(GLUT_WINDOW_HEIGHT) - 1 - y, 10, 10);  // Larger pick region
        if (pick && !myPickNames.empty()) {
            pick(*myPickNames.begin());
        }
    } else {
        if (pick) pick(-1);
    }

    _mouseX = x;
    _mouseY = y;

    if (state == GLUT_UP) {
        switch (button) {
            case GLUT_LEFT_BUTTON:   _mouseLeft = 0; myPickNames.clear(); break;
            case GLUT_MIDDLE_BUTTON: _mouseMiddle = 0; break;
            case GLUT_RIGHT_BUTTON:  _mouseRight = 0; break;
        }
    } else {
        switch (button) {
            case GLUT_LEFT_BUTTON:   _mouseLeft = 1; break;
            case GLUT_MIDDLE_BUTTON: _mouseMiddle = 1; break;
            case GLUT_RIGHT_BUTTON:  _mouseRight = 1; break;
        }
    }

    glGetIntegerv(GL_VIEWPORT, viewport);
    pos(&_dragPosX, &_dragPosY, &_dragPosZ, x, y, viewport);
    glutPostRedisplay();
}

static void zprMotion(int x, int y) {
    int changed = 0;
    const int dx = x - _mouseX;
    const int dy = y - _mouseY;

    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);

    if (dx == 0 && dy == 0) return;

    if (_mouseMiddle || (_mouseLeft && _mouseRight)) {
        // Zoom
        double s = exp((double)dy * 0.01);
        glTranslatef(zprReferencePoint[0], zprReferencePoint[1], zprReferencePoint[2]);
        glScalef(s, s, s);
        glTranslatef(-zprReferencePoint[0], -zprReferencePoint[1], -zprReferencePoint[2]);
        changed = 1;
    } else if (_mouseLeft) {
        // Rotate
        double ax = dy, ay = dx, az = 0.0;
        double angle = vlen(ax, ay, az) / (double)(viewport[2] + 1) * 180.0;
        double bx = _matrixInverse[0]*ax + _matrixInverse[4]*ay + _matrixInverse[8]*az;
        double by = _matrixInverse[1]*ax + _matrixInverse[5]*ay + _matrixInverse[9]*az;
        double bz = _matrixInverse[2]*ax + _matrixInverse[6]*ay + _matrixInverse[10]*az;

        glTranslatef(zprReferencePoint[0], zprReferencePoint[1], zprReferencePoint[2]);
        glRotatef(angle, bx, by, bz);
        glTranslatef(-zprReferencePoint[0], -zprReferencePoint[1], -zprReferencePoint[2]);
        changed = 1;
    } else if (_mouseRight) {
        // Pan
        double px, py, pz;
        pos(&px, &py, &pz, x, y, viewport);
        glLoadIdentity();
        glTranslatef(px - _dragPosX, py - _dragPosY, pz - _dragPosZ);
        glMultMatrixd(_matrix);
        _dragPosX = px;
        _dragPosY = py;
        _dragPosZ = pz;
        changed = 1;
    }

    _mouseX = x;
    _mouseY = y;

    if (changed) {
        getMatrix();
        glutPostRedisplay();
    }
}

// Pick callback - output CSV data for picked points
void _pick(GLint name) {
    if (myPickNames.empty()) return;
    
    // Calculate offsets for the three displayed buffers
    int offset0 = 0;
    int offset1 = 0, offset2 = 0, specialOffset = 0;
    
    if (!allFiles.empty()) {
        if (bufferIndex[0] >= 0 && bufferIndex[0] < (int)allFiles.size()) {
            offset1 = allFiles[bufferIndex[0]].numPoints;
        }
        if (bufferIndex[1] >= 0 && bufferIndex[1] < (int)allFiles.size()) {
            offset2 = offset1 + allFiles[bufferIndex[1]].numPoints;
        }
        specialOffset = offset2;
        if (bufferIndex[2] >= 0 && bufferIndex[2] < (int)allFiles.size()) {
            specialOffset += allFiles[bufferIndex[2]].numPoints;
        }
    } else {
        specialOffset = numPoints;
    }
    
    // Count points from each source
    int buf0Picked = 0, buf1Picked = 0, buf2Picked = 0, specialPicked = 0;
    for (std::set<GLint>::iterator it = myPickNames.begin(); it != myPickNames.end(); ++it) {
        int idx = *it;
        if (!allFiles.empty()) {
            if (idx < offset1) buf0Picked++;
            else if (idx < offset2) buf1Picked++;
            else if (idx < specialOffset) buf2Picked++;
            else specialPicked++;
        } else {
            if (idx < numPoints) buf0Picked++;
            else specialPicked++;
        }
    }
    
    // Print summary
    printf("\n=== Picked %zu points ===\n", myPickNames.size());
    if (!allFiles.empty()) {
        if (bufferIndex[0] >= 0 && bufferIndex[0] < (int)allFiles.size())
            printf("  %s (Red): %d of %d\n", getBasename(allFiles[bufferIndex[0]].filename).c_str(), 
                   buf0Picked, allFiles[bufferIndex[0]].numPoints);
        if (bufferIndex[1] >= 0 && bufferIndex[1] < (int)allFiles.size())
            printf("  %s (Green): %d of %d\n", getBasename(allFiles[bufferIndex[1]].filename).c_str(), 
                   buf1Picked, allFiles[bufferIndex[1]].numPoints);
        if (bufferIndex[2] >= 0 && bufferIndex[2] < (int)allFiles.size())
            printf("  %s (Blue): %d of %d\n", getBasename(allFiles[bufferIndex[2]].filename).c_str(), 
                   buf2Picked, allFiles[bufferIndex[2]].numPoints);
    } else {
        printf("  Regular: %d of %d\n", buf0Picked, numPoints);
    }
    printf("  special.csv (Magenta): %d of %d\n", specialPicked, numSpecialPoints);
    
    for (std::set<GLint>::iterator it = myPickNames.begin(); it != myPickNames.end(); ++it) {
        int idx = *it;
        
        if (!allFiles.empty()) {
            // Multi-file mode
            int bufNum = -1, fileIdx = -1, localIdx = -1;
            
            if (idx < offset1) {
                bufNum = 0; fileIdx = bufferIndex[0]; localIdx = idx;
            } else if (idx < offset2) {
                bufNum = 1; fileIdx = bufferIndex[1]; localIdx = idx - offset1;
            } else if (idx < specialOffset) {
                bufNum = 2; fileIdx = bufferIndex[2]; localIdx = idx - offset2;
            }
            
            if (bufNum >= 0 && fileIdx >= 0 && fileIdx < (int)allFiles.size()) {
                FileData &fd = allFiles[fileIdx];
                if (localIdx >= 0 && localIdx < (int)fd.csvLines.size()) {
                    printf("File: %s\n", getBasename(fd.filename).c_str());
                    printf("Header: %s\n", fd.csvHeader.c_str());
                    printf("---\n");
                    
                    std::vector<std::string> values = splitCSVLine(fd.csvLines[localIdx]);
                    
                    std::string valX = (colIdxX >= 0 && colIdxX < (int)values.size()) ? values[colIdxX] : "N/A";
                    std::string valY = (colIdxY >= 0 && colIdxY < (int)values.size()) ? values[colIdxY] : "N/A";
                    std::string valZ = (colIdxZ >= 0 && colIdxZ < (int)values.size()) ? values[colIdxZ] : "N/A";
                    
                    printf("%s=%s\n", fieldNameX.c_str(), valX.c_str());
                    printf("%s=%s\n", fieldNameY.c_str(), valY.c_str());
                    printf("%s=%s\n", fieldNameZ.c_str(), valZ.c_str());
                    printf("[%d] %s\n", localIdx, fd.csvLines[localIdx].c_str());
                    
                    // Update center of rotation
                    glBindBuffer(GL_ARRAY_BUFFER, fd.vbo);
                    float4 *pts = (float4*)glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);
                    if (pts) {
                        zprReferencePoint[0] = pts[localIdx].x;
                        zprReferencePoint[1] = pts[localIdx].y;
                        zprReferencePoint[2] = pts[localIdx].z;
                        printf("Center of rotation set to (%.3f, %.3f, %.3f)\n",
                               zprReferencePoint[0], zprReferencePoint[1], zprReferencePoint[2]);
                        glUnmapBuffer(GL_ARRAY_BUFFER);
                    }
                    glBindBuffer(GL_ARRAY_BUFFER, 0);
                }
            } else if (idx >= specialOffset) {
                // Special point
                int specialIdx = idx - specialOffset;
                if (specialIdx >= 0 && specialIdx < (int)specialCsvLines.size()) {
                    printf("File: special.csv\n");
                    printf("Header: %s\n", specialCsvHeader.c_str());
                    printf("---\n");
                    
                    std::vector<std::string> values = splitCSVLine(specialCsvLines[specialIdx]);
                    std::vector<std::string> headers = splitCSVLine(specialCsvHeader);
                    int sIdx1 = findColumnIndex(headers, fieldNameX.c_str());
                    int sIdx2 = findColumnIndex(headers, fieldNameY.c_str());
                    int sIdx3 = findColumnIndex(headers, fieldNameZ.c_str());
                    
                    std::string valX = (sIdx1 >= 0 && sIdx1 < (int)values.size()) ? values[sIdx1] : "N/A";
                    std::string valY = (sIdx2 >= 0 && sIdx2 < (int)values.size()) ? values[sIdx2] : "N/A";
                    std::string valZ = (sIdx3 >= 0 && sIdx3 < (int)values.size()) ? values[sIdx3] : "N/A";
                    
                    printf("%s=%s\n", fieldNameX.c_str(), valX.c_str());
                    printf("%s=%s\n", fieldNameY.c_str(), valY.c_str());
                    printf("%s=%s\n", fieldNameZ.c_str(), valZ.c_str());
                    printf("[special:%d] %s\n", specialIdx, specialCsvLines[specialIdx].c_str());
                    
                    // Update center of rotation
                    if (vboSpecial) {
                        glBindBuffer(GL_ARRAY_BUFFER, vboSpecial);
                        float4 *pts = (float4*)glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);
                        if (pts) {
                            zprReferencePoint[0] = pts[specialIdx].x;
                            zprReferencePoint[1] = pts[specialIdx].y;
                            zprReferencePoint[2] = pts[specialIdx].z;
                            printf("Center of rotation set to (%.3f, %.3f, %.3f)\n",
                                   zprReferencePoint[0], zprReferencePoint[1], zprReferencePoint[2]);
                            glUnmapBuffer(GL_ARRAY_BUFFER);
                        }
                        glBindBuffer(GL_ARRAY_BUFFER, 0);
                    }
                }
            }
        } else {
            // Legacy single-file mode
            if (idx < numPoints) {
                if (idx >= 0 && idx < (int)csvLines.size()) {
                    printf("Header: %s\n", csvHeader.c_str());
                    printf("---\n");
                    
                    std::vector<std::string> values = splitCSVLine(csvLines[idx]);
                    
                    std::string valX = (colIdxX >= 0 && colIdxX < (int)values.size()) ? values[colIdxX] : "N/A";
                    std::string valY = (colIdxY >= 0 && colIdxY < (int)values.size()) ? values[colIdxY] : "N/A";
                    std::string valZ = (colIdxZ >= 0 && colIdxZ < (int)values.size()) ? values[colIdxZ] : "N/A";
                    
                    printf("%s=%s\n", fieldNameX.c_str(), valX.c_str());
                    printf("%s=%s\n", fieldNameY.c_str(), valY.c_str());
                    printf("%s=%s\n", fieldNameZ.c_str(), valZ.c_str());
                    printf("[%d] %s\n", idx, csvLines[idx].c_str());
                }
            } else {
                // Special point in legacy mode
                int specialIdx = idx - numPoints;
                if (specialIdx >= 0 && specialIdx < (int)specialCsvLines.size()) {
                    printf("File: special.csv\n");
                    printf("Header: %s\n", specialCsvHeader.c_str());
                    printf("---\n");
                    
                    std::vector<std::string> values = splitCSVLine(specialCsvLines[specialIdx]);
                    std::vector<std::string> headers = splitCSVLine(specialCsvHeader);
                    int sIdx1 = findColumnIndex(headers, fieldNameX.c_str());
                    int sIdx2 = findColumnIndex(headers, fieldNameY.c_str());
                    int sIdx3 = findColumnIndex(headers, fieldNameZ.c_str());
                    
                    std::string valX = (sIdx1 >= 0 && sIdx1 < (int)values.size()) ? values[sIdx1] : "N/A";
                    std::string valY = (sIdx2 >= 0 && sIdx2 < (int)values.size()) ? values[sIdx2] : "N/A";
                    std::string valZ = (sIdx3 >= 0 && sIdx3 < (int)values.size()) ? values[sIdx3] : "N/A";
                    
                    printf("%s=%s\n", fieldNameX.c_str(), valX.c_str());
                    printf("%s=%s\n", fieldNameY.c_str(), valY.c_str());
                    printf("%s=%s\n", fieldNameZ.c_str(), valZ.c_str());
                    printf("[special:%d] %s\n", specialIdx, specialCsvLines[specialIdx].c_str());
                }
            }
        }
    }
    printf("===\n\n");
    fflush(stdout);
}

// Text rendering
void renderBitmapString(float x, float y, void *font, const char *string) {
    glRasterPos2f(x, y);
    for (const char *c = string; *c != '\0'; c++) {
        glutBitmapCharacter(font, *c);
    }
}

void setOrthographicProjection() {
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(0, WINDOWX, 0, WINDOWY);
    glScalef(1, -1, 1);
    glTranslatef(0, -WINDOWY, 0);
    glMatrixMode(GL_MODELVIEW);
}

void resetPerspectiveProjection() {
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
}

void drawText() {
    glColor3f(0.0f, 1.0f, 0.0f);
    setOrthographicProjection();
    glPushMatrix();
    glLoadIdentity();
    int lightingState = glIsEnabled(GL_LIGHTING);
    glDisable(GL_LIGHTING);
    renderBitmapString(3, WINDOWY - 3, (void *)MYFONT, console_string);
    if (lightingState) glEnable(GL_LIGHTING);
    glPopMatrix();
    resetPerspectiveProjection();
}

// Draw the scene - axes and point cloud
void drawScene() {
    drawAxes();
    drawPoints();
    drawSpecialPoints();
}

void display() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    drawScene();
    drawText();
    glutSwapBuffers();
    renderflag = 0;
}

void keyboard(unsigned char key, int x, int y) {
    switch (key) {
        case 8:   // Backspace
        case 127: // Delete (on some systems)
            if (console_position > 0) {
                console_position--;
                console_string[console_position] = '\0';
                printf("STRING: %s\n", console_string);
                display();
            }
            break;

        case 13: // Enter
            console_string[0] = '\0';
            console_position = 0;
            display();
            break;

        case 27: // Escape
            exit(0);
            break;

        default:
            console_string[console_position++] = (char)key;
            console_string[console_position] = '\0';
            printf("STRING: %s\n", console_string);
            display();
            break;
    }
}

// Special keyboard callback for arrow keys
void specialKeyboard(int key, int x, int y) {
    if (allFiles.empty()) return;
    
    int numFiles = (int)allFiles.size();
    
    switch (key) {
        case GLUT_KEY_RIGHT:
            // Increment all buffer indices
            for (int i = 0; i < 3; i++) {
                bufferIndex[i] = (bufferIndex[i] + 1) % numFiles;
            }
            printf("Buffers: Red=%s, Green=%s, Blue=%s\n",
                   getBasename(allFiles[bufferIndex[0]].filename).c_str(),
                   getBasename(allFiles[bufferIndex[1]].filename).c_str(),
                   getBasename(allFiles[bufferIndex[2]].filename).c_str());
            glutPostRedisplay();
            break;
            
        case GLUT_KEY_LEFT:
            // Decrement all buffer indices (wrap around)
            for (int i = 0; i < 3; i++) {
                bufferIndex[i] = (bufferIndex[i] - 1 + numFiles) % numFiles;
            }
            printf("Buffers: Red=%s, Green=%s, Blue=%s\n",
                   getBasename(allFiles[bufferIndex[0]].filename).c_str(),
                   getBasename(allFiles[bufferIndex[1]].filename).c_str(),
                   getBasename(allFiles[bufferIndex[2]].filename).c_str());
            glutPostRedisplay();
            break;
    }
}

void idle() {
    if (renderflag) {
        glFlush();
        glutPostRedisplay();
    }
}

// Get list of OBS files from obs folder
std::vector<std::string> getObsFiles() {
    std::vector<std::string> files;
    DIR *dir = opendir("obs");
    if (dir) {
        struct dirent *entry;
        while ((entry = readdir(dir)) != NULL) {
            std::string name = entry->d_name;
            if (name.find("_OBS.csv") != std::string::npos) {
                files.push_back("obs/" + name);
            }
        }
        closedir(dir);
        std::sort(files.begin(), files.end());
    }
    return files;
}

int main(int argc, char *argv[]) {
    // Check command line arguments - now only field names
    if (argc != 4) {
        fprintf(stderr, "Usage: %s field_x field_y field_z\n", argv[0]);
        fprintf(stderr, "Example: %s FIRE_WEATHER_INDEX INITIAL_SPREAD_INDEX FINE_FUEL_MOISTURE_CODE\n", argv[0]);
        fprintf(stderr, "Loads all obs/*_OBS.csv files automatically\n");
        return 1;
    }
    
    const char *field1 = argv[1];
    const char *field2 = argv[2];
    const char *field3 = argv[3];
    
    // Store field names for axis labels
    fieldNameX = field1;
    fieldNameY = field2;
    fieldNameZ = field3;
    
    // Get list of OBS files
    std::vector<std::string> obsFiles = getObsFiles();
    if (obsFiles.empty()) {
        fprintf(stderr, "No *_OBS.csv files found in obs/ folder\n");
        return 1;
    }
    
    printf("Found %zu OBS files\n", obsFiles.size());
    
    // Try to load bounds from cache, otherwise compute and save
    if (!loadBoundsCache(field1, field2, field3)) {
        printf("Pass 1: Scanning for global bounds (no cache found)...\n");
        scanAllFilesForBounds(obsFiles, field1, field2, field3);
        saveBoundsCache(field1, field2, field3);
    }
    
    // Load all files - try cache first, otherwise load from CSV and cache
    printf("Pass 2: Loading all files...\n");
    totalPoints = 0;
    int cacheHits = 0, cacheMisses = 0;
    for (size_t i = 0; i < obsFiles.size(); i++) {
        FileData fd;
        if (loadFileCache(obsFiles[i], field1, field2, field3, fd)) {
            // Loaded from cache
            cacheHits++;
        } else {
            // Load from CSV and save to cache
            if (!loadSingleFile(obsFiles[i], field1, field2, field3, fd)) {
                continue;
            }
            saveFileCache(fd, field1, field2, field3);
            cacheMisses++;
        }
        fd.startIndex = totalPoints;
        totalPoints += fd.numPoints;
        allFiles.push_back(fd);
        printf("\r  Loaded %zu/%zu: %s (%d points, total: %d)%s", 
               i + 1, obsFiles.size(), getBasename(fd.filename).c_str(), fd.numPoints, totalPoints,
               (cacheHits > cacheMisses) ? " [cached]" : "");
        fflush(stdout);
    }
    printf("\n");
    printf("Loaded %zu files with %d total points (%d from cache, %d recomputed)\n", 
           allFiles.size(), totalPoints, cacheHits, cacheMisses);
    
    // Initialize buffer indices to first three files
    bufferIndex[0] = 0;
    bufferIndex[1] = (allFiles.size() > 1) ? 1 : 0;
    bufferIndex[2] = (allFiles.size() > 2) ? 2 : 0;

    pick = _pick;
    renderflag = 0;
    console_position = 0;
    console_string[0] = '\0';
    fullscreen = 0;

    // Initialize GLUT
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(STARTX, STARTY);
    
    char title[256];
    snprintf(title, sizeof(title), "Point Cloud Viewer (%d points from %zu files)", totalPoints, allFiles.size());
    glutCreateWindow(title);
    
    // Initialize GLEW
    GLenum glewErr = glewInit();
    if (glewErr != GLEW_OK) {
        fprintf(stderr, "GLEW error: %s\n", glewGetErrorString(glewErr));
        return 1;
    }
    printf("GLEW initialized, GL version: %s\n", glGetString(GL_VERSION));
    
    // Now create VBOs for all loaded files (GL context is ready)
    printf("Creating VBOs for %zu files...\n", allFiles.size());
    for (size_t i = 0; i < allFiles.size(); i++) {
        createFileVBO(allFiles[i]);
        printf("\r  Created VBO %zu/%zu", i + 1, allFiles.size());
        fflush(stdout);
    }
    printf("\n");
    
    zprInit();

    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutSpecialFunc(specialKeyboard);  // Arrow keys
    glutIdleFunc(idle);
    
    // Load and initialize special points if special.csv exists
    std::vector<float4> specialPoints;
    if (loadSpecialCSV(field1, field2, field3, specialPoints)) {
        transformSpecialPoints(specialPoints);
        initSpecialVBO(specialPoints);
    }

    zprSelectionFunc(drawScene);
    zprPickFunc(pick);

    // Initialize lighting (for potential future use)
    glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
    glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
    glLightfv(GL_LIGHT0, GL_POSITION, light_position);

    glMaterialfv(GL_FRONT, GL_AMBIENT, mat_ambient);
    glMaterialfv(GL_FRONT, GL_DIFFUSE, mat_diffuse);
    glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
    glMaterialfv(GL_FRONT, GL_SHININESS, high_shininess);

    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glDepthFunc(GL_LESS);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_NORMALIZE);
    glEnable(GL_COLOR_MATERIAL);
    
    // Dark background for better point visibility
    glClearColor(0.1f, 0.1f, 0.15f, 1.0f);

    // Register cleanup
    atexit(cleanupVBO);
    
    // Print initial buffer selection
    printf("Initial buffers: Red=%s, Green=%s, Blue=%s\n",
           getBasename(allFiles[bufferIndex[0]].filename).c_str(),
           getBasename(allFiles[bufferIndex[1]].filename).c_str(),
           getBasename(allFiles[bufferIndex[2]].filename).c_str());
    printf("Use Left/Right arrow keys to cycle through years\n");

    glutMainLoop();;
    return 0;
}




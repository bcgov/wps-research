/* 20220830 dem.cpp: plot an image in 3d
 * OPTIMIZED VERSION - Uses VBOs for GPU-accelerated rendering
 *
 * usage:
 *   dem [input raster file] [z coordinate band] [r coord band] [g coord band] [b coord band] 
 *
 * 20220831 add left-right arrows to shift z band, up-down arrows to shift r,g,b bands
 * 20250120 OPTIMIZED: VBOs, indexed rendering, reduced CPU-GPU transfers
 *
 * COMPILE WITH: g++ -std=c++11 -O3 -o dem dem_optimized.cpp -lGLEW -lGL -lGLU -lglut -lpthread
 * INSTALL GLEW: sudo apt-get install libglew-dev
 */

#define Z_SCALE 0.25
long int ri, gi, bi, zi;
#define MYFONT GLUT_BITMAP_HELVETICA_12
#define STR_MAX 1000

// GLEW must be included BEFORE any GL headers
#include <GL/glew.h>

#include "newzpr.h"
#include "pthread.h"
#include "time.h"
#include "vec3d.h"
#include "misc.h"
#include <stdio.h>
#include <stdlib.h>
#include <cfloat>
#include <vector>

vector<string> band_names;
size_t np;

vec3d rX; // 3d reference point
char console_string[STR_MAX];
size_t nrow, ncol, nband;  // base image data

float *dat, zmax, zmin; // dem data: nrow * ncol linear array of floats
float *rgb; // basemap data

vec3d *points; // 3d points for visualization

// ============================================================================
// VBO OPTIMIZATION ADDITIONS
// ============================================================================
GLuint vbo_vertices = 0;
GLuint vbo_colors = 0;
GLuint vbo_indices = 0;
size_t num_indices = 0;

// Cached vertex and color data for efficient updates
std::vector<float> vertex_data;
std::vector<float> color_data;
std::vector<GLuint> index_data;

// Flags to track what needs updating
bool vertices_dirty = true;
bool colors_dirty = true;

// Flag to check if VBOs are supported and initialized
bool use_vbos = false;

void initVBOs() {
    // Initialize GLEW (must be done after OpenGL context creation)
    glewExperimental = GL_TRUE;  // Needed for core profile
    GLenum err = glewInit();
    if (err != GLEW_OK) {
        fprintf(stderr, "GLEW init failed: %s\n", glewGetErrorString(err));
        fprintf(stderr, "Falling back to immediate mode rendering\n");
        use_vbos = false;
        return;
    }
    
    // Clear any GL errors that glewInit might have caused
    while (glGetError() != GL_NO_ERROR) {}
    
    printf("GLEW %s initialized successfully\n", glewGetString(GLEW_VERSION));
    printf("OpenGL version: %s\n", glGetString(GL_VERSION));
    printf("OpenGL renderer: %s\n", glGetString(GL_RENDERER));
    
    // Check for VBO support (available since OpenGL 1.5, or via ARB extension)
    if (!GLEW_VERSION_1_5 && !GLEW_ARB_vertex_buffer_object) {
        fprintf(stderr, "VBOs not supported, falling back to immediate mode\n");
        use_vbos = false;
        return;
    }
    
    use_vbos = true;
    printf("VBO support confirmed\n");
    
    // Generate buffer objects
    glGenBuffers(1, &vbo_vertices);
    glGenBuffers(1, &vbo_colors);
    glGenBuffers(1, &vbo_indices);
    
    // Check for errors
    GLenum glerr = glGetError();
    if (glerr != GL_NO_ERROR) {
        fprintf(stderr, "OpenGL error during VBO creation: 0x%x\n", glerr);
        use_vbos = false;
        return;
    }
    
    // Pre-allocate vertex data array
    vertex_data.resize(nrow * ncol * 3);
    
    // Pre-allocate color data array
    color_data.resize(nrow * ncol * 3);
    
    // Build index buffer for triangles (this never changes)
    // Each quad becomes 2 triangles
    index_data.reserve((nrow - 1) * (ncol - 1) * 6);
    
    for (size_t i = 0; i < nrow - 1; i++) {
        for (size_t j = 0; j < ncol - 1; j++) {
            size_t k = i * ncol + j;
            
            // First triangle of quad
            index_data.push_back(k);
            index_data.push_back(k + ncol);
            index_data.push_back(k + ncol + 1);
            
            // Second triangle of quad
            index_data.push_back(k);
            index_data.push_back(k + ncol + 1);
            index_data.push_back(k + 1);
        }
    }
    
    num_indices = index_data.size();
    
    // Upload index buffer (static - only done once)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_indices);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 
                 index_data.size() * sizeof(GLuint), 
                 index_data.data(), 
                 GL_STATIC_DRAW);
    
    // Allocate vertex buffer (dynamic - will be updated)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_vertices);
    glBufferData(GL_ARRAY_BUFFER, 
                 vertex_data.size() * sizeof(float), 
                 NULL, 
                 GL_DYNAMIC_DRAW);
    
    // Allocate color buffer (dynamic - will be updated)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_colors);
    glBufferData(GL_ARRAY_BUFFER, 
                 color_data.size() * sizeof(float), 
                 NULL, 
                 GL_DYNAMIC_DRAW);
    
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    
    printf("VBOs initialized: %zu vertices, %zu indices (%zu triangles)\n", 
           nrow * ncol, num_indices, num_indices / 3);
}

void updateVertexBuffer() {
    if (!use_vbos || !vertices_dirty) return;
    
    // Build vertex data from points array
    for (size_t i = 0; i < nrow * ncol; i++) {
        vertex_data[i * 3 + 0] = points[i].x;
        vertex_data[i * 3 + 1] = points[i].y;
        vertex_data[i * 3 + 2] = points[i].z;
    }
    
    // Upload to GPU
    glBindBuffer(GL_ARRAY_BUFFER, vbo_vertices);
    glBufferSubData(GL_ARRAY_BUFFER, 0, 
                    vertex_data.size() * sizeof(float), 
                    vertex_data.data());
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    
    vertices_dirty = false;
}

void updateColorBuffer() {
    if (!use_vbos || !colors_dirty) return;
    
    // Build color data from band selections
    for (size_t k = 0; k < nrow * ncol; k++) {
        float R = dat[ri * np + k];
        float G = dat[gi * np + k];
        float B = dat[bi * np + k];
        
        // Handle NaN values - set to black (or could use a sentinel color)
        color_data[k * 3 + 0] = isnan(R) ? 0.0f : R;
        color_data[k * 3 + 1] = isnan(G) ? 0.0f : G;
        color_data[k * 3 + 2] = isnan(B) ? 0.0f : B;
    }
    
    // Upload to GPU
    glBindBuffer(GL_ARRAY_BUFFER, vbo_colors);
    glBufferSubData(GL_ARRAY_BUFFER, 0, 
                    color_data.size() * sizeof(float), 
                    color_data.data());
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    
    colors_dirty = false;
}

void cleanupVBOs() {
    if (use_vbos) {
        if (vbo_vertices) glDeleteBuffers(1, &vbo_vertices);
        if (vbo_colors) glDeleteBuffers(1, &vbo_colors);
        if (vbo_indices) glDeleteBuffers(1, &vbo_indices);
    }
}

// ============================================================================
// END VBO ADDITIONS
// ============================================================================

void setOrthographicProjection() {
    int h = WINDOWX;
    int w = WINDOWY;
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(0., (float)w, 0., (float)h);
    glScalef(1., -1., 1.);
    glTranslatef(0, -1.*(float)h, 0);
    glMatrixMode(GL_MODELVIEW);
}

void resetPerspectiveProjection() {
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
}

void world2screen(float x, float y, float z, float &screenX, float &screenY, float &screenZ) {
    double vx, vy, vz;
    GLint view[4];
    GLdouble proj[16];
    GLdouble model[16];
    glGetDoublev(GL_PROJECTION_MATRIX, proj);
    glGetDoublev(GL_MODELVIEW_MATRIX, model);
    glGetIntegerv(GL_VIEWPORT, view);
    gluProject(x, y, z, model, proj, view, &vx, &vy, &vz);
    screenY = (float)(vy - glutGet(GLUT_WINDOW_HEIGHT));
    screenX = (float)vx;
    screenZ = (float)vz;
}

void renderBitmapString(float x, float y, void *font, const char *string) {
    const char *c;
    glRasterPos2f(x, y);
    for (c = string; *c != '\0'; c++)
        glutBitmapCharacter(font, *c);
}

void drawText(char *s, int offset) {
    glColor3f(0.0f, 1.0f, 0.0f);
    setOrthographicProjection();
    glPushMatrix();
    glLoadIdentity();
    int lightingState = glIsEnabled(GL_LIGHTING);
    glDisable(GL_LIGHTING);
    renderBitmapString(3, WINDOWY - 3 - offset, (void *)MYFONT, (const char *)s);
    if (lightingState) glEnable(GL_LIGHTING);
    glPopMatrix();
    resetPerspectiveProjection();
}

GLint selected;
int special_key;

#define STARTX 500
#define STARTY 500
int fullscreen;
clock_t start_time;
clock_t stop_time;
#define SECONDS_PAUSE 0.4
int console_position;
int renderflag;

void _pick(GLint name) {
    if (myPickNames.size() > 0) {
        cout << "PickSet:";
        std::set<GLint>::iterator it;
        for (it = myPickNames.begin(); it != myPickNames.end(); it++)
            cout << *it << ",";
        cout << endl;
        fflush(stdout);
    }
}

float a1, a2, a3;

// ============================================================================
// FALLBACK: Original immediate mode rendering (for systems without VBO support)
// ============================================================================
void display_immediate_mode() {
    size_t i, j, k;
    float R, G, B;
    
    for0(i, nrow) {
        for0(j, ncol) {
            k = i * ncol + j;
            R = dat[ri * np + k];
            G = dat[gi * np + k];
            B = dat[bi * np + k];

            if (!(isnan(R) || isnan(G) || isnan(B))) {
                glColor3f(R, G, B);
                if (i + 1 < nrow && j + 1 < ncol) {
                    size_t i_1 = (i + 1) * ncol + j;
                    glBegin(GL_POLYGON);
                    points[k].vertex();
                    points[i_1].vertex();
                    points[i_1 + 1].vertex();
                    points[i_1 - ncol + 1].vertex();
                    glEnd();
                }
            }
        }
    }
}

// ============================================================================
// OPTIMIZED DISPLAY FUNCTION
// ============================================================================
void display(void) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glPushMatrix();
    glTranslatef(-rX.x, -rX.y, -rX.z);

    // Draw axes (optional, can be disabled for more speed)
    if (true) {
        vec3d X;
        X.x = X.y = X.z = 0; X.axis(.01);
        X.x = X.y = 1;       X.axis(.01);
        X.x = 1; X.y = 0;    X.axis(.01);
        X.x = 0; X.y = 1;    X.axis(.01);
        rX.axis(.1);
    }

    if (use_vbos) {
        // ====================================================================
        // VBO-BASED RENDERING - Much faster than immediate mode!
        // ====================================================================
        
        // Update buffers if needed (only when data changes)
        updateVertexBuffer();
        updateColorBuffer();
        
        // Enable vertex arrays
        glEnableClientState(GL_VERTEX_ARRAY);
        glEnableClientState(GL_COLOR_ARRAY);
        
        // Bind vertex buffer and set pointer
        glBindBuffer(GL_ARRAY_BUFFER, vbo_vertices);
        glVertexPointer(3, GL_FLOAT, 0, NULL);
        
        // Bind color buffer and set pointer
        glBindBuffer(GL_ARRAY_BUFFER, vbo_colors);
        glColorPointer(3, GL_FLOAT, 0, NULL);
        
        // Bind index buffer and draw all triangles in one call
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_indices);
        glDrawElements(GL_TRIANGLES, num_indices, GL_UNSIGNED_INT, NULL);
        
        // Cleanup state
        glDisableClientState(GL_VERTEX_ARRAY);
        glDisableClientState(GL_COLOR_ARRAY);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    } else {
        // Fallback to original immediate mode
        display_immediate_mode();
    }
    
    glPopMatrix();
    glutSwapBuffers();
    renderflag = false;
}

void quitme() {
    cleanupVBOs();
    exit(0);
}

void special(int key, int x, int y) {
    printf("special %d\n", key);
    special_key = key;

    if (key == GLUT_KEY_LEFT || key == GLUT_KEY_RIGHT) {
        if (key == GLUT_KEY_LEFT) {
            printf("LEFT\n");
            zi--;
            if (zi < 0) {
                zi = nband - 1;
            }
        }
        if (key == GLUT_KEY_RIGHT) {
            printf("RIGHT\n");
            zi++;
            if (zi >= nband) {
                zi = 0;
            }
        }

        // Update z coordinates
        size_t i, j, k;
        for0(i, nrow) {
            for0(j, ncol) {
                k = (i * ncol) + j;
                points[k].z = dat[k + (zi * np)];
                points[k].z *= Z_SCALE;
            }
        }
        
        // Mark vertices as needing update
        vertices_dirty = true;

        printf("zi %ld\n", zi);
        str title(str("z=(") + band_names[zi] + str(") r=(") + band_names[ri] + 
                  str(") g=(") + band_names[gi] + str(") b=(") + band_names[bi] + str(")"));
        glutSetWindowTitle(title.c_str());
        glutPostRedisplay();
    }

    if (key == GLUT_KEY_UP || key == GLUT_KEY_DOWN) {
        if (key == GLUT_KEY_UP) {
            ri++;
            gi++;
            bi++;
            if (ri >= nband) ri = 0;
            if (gi >= nband) gi = 0;
            if (bi >= nband) bi = 0;
        }
        if (key == GLUT_KEY_DOWN) {
            ri--;
            gi--;
            bi--;
            if (ri < 0) ri = nband - 1;
            if (gi < 0) gi = nband - 1;
            if (bi < 0) bi = nband - 1;
        }
        
        // Mark colors as needing update
        colors_dirty = true;
        
        str title(str("z=(") + band_names[zi] + str(") r=(") + band_names[ri] + 
                  str(") g=(") + band_names[gi] + str(") b=(") + band_names[bi] + str(")"));
        glutSetWindowTitle(title.c_str());
        glutPostRedisplay();
    }
}

void keyboard(unsigned char key, int x, int y) {
    switch (key) {
        case 8:
        case 127:
            if (console_position > 0) {
                console_position--;
                console_string[console_position] = '\0';
                printf("STRING: %s\n", &console_string[0]);
                glutPostRedisplay();
            }
            break;

        case 13:  // Enter
            if (true) {
                str S(console_string);
                cout << "[" << S << "]" << endl;

                long int x;

                if ((console_string[0] == 'r') ||
                    (console_string[0] == 'g') ||
                    (console_string[0] == 'b') ||
                    (console_string[0] == 'z')) {

                    x = atol(str(&console_string[1]).c_str());
                    printf("%ci %ld\n", console_string[0], x);

                    if (console_string[0] == 'r') {
                        if (x >= 0 && x < nband) {
                            ri = x;
                            colors_dirty = true;
                        }
                    }
                    if (console_string[0] == 'g') {
                        if (x >= 0 && x < nband) {
                            gi = x;
                            colors_dirty = true;
                        }
                    }
                    if (console_string[0] == 'b') {
                        if (x >= 0 && x < nband) {
                            bi = x;
                            colors_dirty = true;
                        }
                    }
                    if (console_string[0] == 'z') {
                        if (x >= 0 && x < nband) {
                            zi = x;
                            size_t i, j, k;
                            for0(i, nrow) {
                                for0(j, ncol) {
                                    k = (i * ncol) + j;
                                    points[k].z = dat[k + (zi * np)];
                                    points[k].z *= Z_SCALE;
                                }
                            }
                            vertices_dirty = true;
                        }
                    }

                    str title(str("z=(") + band_names[zi] + str(") r=(") + band_names[ri] + 
                              str(") g=(") + band_names[gi] + str(") b=(") + band_names[bi] + str(")"));
                    glutSetWindowTitle(title.c_str());
                }

                console_string[0] = '\0';
                console_position = 0;
                glutPostRedisplay();
            }
            break;

        case 27:  // Escape
            quitme();
            break;

        default:
            console_string[console_position++] = (char)key;
            console_string[console_position] = '\0';
            printf("STRING: %s\n", &console_string[0]);
            glutPostRedisplay();
            break;
    }
}

static GLfloat light_ambient[] = {0.0, 0.0, 0.0, 1.0};
static GLfloat light_diffuse[] = {1.0, 1.0, 1.0, 1.0};
static GLfloat light_specular[] = {1.0, 1.0, 1.0, 1.0};
static GLfloat light_position[] = {1.0, 1.0, 1.0, 0.0};

static GLfloat mat_ambient[] = {0.7, 0.7, 0.7, 1.0};
static GLfloat mat_diffuse[] = {0.8, 0.8, 0.8, 1.0};
static GLfloat mat_specular[] = {1.0, 1.0, 1.0, 1.0};
static GLfloat high_shininess[] = {100.0};

void idle() {
    if (renderflag) {
        glutPostRedisplay();
    }
}

int main(int argc, char **argv) {
    ri = gi = bi = zi = 0;
    
    // DEM data file
    str fn(argv[1]);

    if (!exists(fn)) err("please check input file");

    str hfn(hdr_fn(fn));

    size_t i, j;
    rX.x = rX.y = rX.z = 0.;

    band_names = vector<string>();

    hread(hfn, nrow, ncol, nband, band_names);
    dat = bread(fn, nrow, ncol, nband);
    np = nrow * ncol;

    for0(i, nband) {
        cout << "\"" << band_names[i] << "\"" << endl;
    }

    ri = nband - 1;
    gi = nband - 2;
    bi = nband - 3;

    zmax = -(float)FLT_MAX;
    zmin = +(float)FLT_MAX;

    // Initialize points array
    points = new vec3d[nrow * ncol];
    for0(i, nrow) {
        for0(j, ncol) {
            size_t k = (i * ncol) + j;
            points[k].x = ((float)j) / ((float)nrow);
            points[k].y = 1. - ((float)i) / ((float)nrow);
            points[k].z = dat[k];
        }
    }

    printf("zmin %f zmax %f\n", zmin, zmax);

    // Apply Z scaling
    for0(i, nrow) {
        for0(j, ncol) {
            size_t k = (i * ncol) + j;
            points[k].z *= Z_SCALE;
        }
    }

    // Set reference point to center
    rX.x = points[ncol / 2 + (nrow / 2) * ncol].x;
    rX.y = points[ncol / 2 + (nrow / 2) * ncol].y;
    rX.z = points[ncol / 2 + (nrow / 2) * ncol].z;

    pick = _pick;
    special_key = selected = -1;
    renderflag = false;
    a1 = a2 = a3 = 1;
    console_position = 0;
    fullscreen = 0;

    str title(str("z=(") + band_names[zi] + str(") r=(") + band_names[ri] + 
              str(") g=(") + band_names[gi] + str(") b=(") + band_names[bi] + str(")"));

    printf("glutInit()\n");
    glutInit(&argc, argv);
    
    // Request double buffering, RGB color, depth buffer, and multisampling for AA
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH | GLUT_MULTISAMPLE);
    glutInitWindowSize(STARTX, STARTY);
    glutCreateWindow(title.c_str());
    
    // ========================================================================
    // Initialize GLEW and VBOs after OpenGL context is created
    // ========================================================================
    initVBOs();
    // ========================================================================
    
    zprInit();

    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutSpecialFunc(special);
    glutIdleFunc(idle);
    glScalef(0.25, 0.25, 0.25);

    zprSelectionFunc(display);
    zprPickFunc(pick);

    // OpenGL state setup
    glDepthFunc(GL_LESS);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_NORMALIZE);
    glEnable(GL_COLOR_MATERIAL);
    
    // ========================================================================
    // Additional optimizations for NVIDIA L40S
    // ========================================================================
    
    // Enable backface culling (assumes consistent winding order)
    // Comment out if you see missing faces
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glFrontFace(GL_CCW);
    
    // Enable multisampling anti-aliasing if available
    glEnable(GL_MULTISAMPLE);
    
    // Hint for best quality/performance
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
    glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST);
    
    printf("Mesh size: %zu x %zu = %zu vertices, %zu triangles\n", 
           nrow, ncol, nrow * ncol, num_indices / 3);
    printf("Rendering mode: %s\n", use_vbos ? "VBO (accelerated)" : "Immediate mode (slow)");
    
    // ========================================================================

    glutMainLoop();
    return 0;
}




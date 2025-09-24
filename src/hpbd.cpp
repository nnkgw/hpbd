// hpbd.cpp - Hierarchical Position-Based Dynamics
#if defined(WIN32)
#pragma warning(disable:4996)
#include <GL/freeglut.h>
#elif defined(__APPLE__) || defined(MACOSX)
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#define GL_SILENCE_DEPRECATION
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <fstream>   // <-- for CSV output

using glm::vec2;
using glm::vec3;
using glm::dot;
using glm::length;
using glm::normalize;

struct Constraint {
  int i, j;
  float rest;
  int level;
};

struct Particle {
  vec3 p;                 // position
  vec3 v;                 // velocity
  vec3 q;                 // saved position per level
  float w;                // inverse mass
  int level;              // finest level this particle belongs to
  std::vector<int> parents;
  std::vector<float> wij;
  bool pinned = false;
  vec3 pinPos;
};

static const float dt = 1.0f / 60.0f;
static const vec3 g(0.0f, -9.8f, 0.0f);
static const int clothW = 30, clothH = 20;
static const float spacing = 0.05f;

std::vector<Particle> P;
std::vector<std::vector<int>> levelParticles;
std::vector<Constraint> constraints;
int Lmax = 2;

// Camera
float camDist = 2.0f;
float camYaw = 0.0f;
float camPitch = -30.0f * 3.14159265f / 180.0f;
vec2 camPan(0.0f, -0.3f);
int lastMouseX = 0, lastMouseY = 0;
bool lbtn = false, rbtn = false;

// Toggles
bool g_useHierarchy = true;
int  g_itersHier    =   3;
int  g_itersPlain   =  20;

// Mapping from any particle id to its source finest (level-0) index
std::vector<int> coarseFromFine;

inline int idx(int x, int y) { return y * clothW + x; }

// ---------- Level-0 build ----------
void buildLevel0() {
  P.clear();
  P.resize(clothW * clothH);
  levelParticles.assign(Lmax + 1, {});
  constraints.clear();

  coarseFromFine.assign(clothW * clothH, -1);

  for (int y = 0; y < clothH; ++y) {
    for (int x = 0; x < clothW; ++x) {
      int id = idx(x, y);
      P[id].p = vec3((x - (clothW - 1) * 0.5f) * spacing,
                     (clothH - 1 - y) * spacing, 0);
      P[id].v = vec3(0);
      P[id].w = 1.0f;
      P[id].level = 0;
      if (y == 0 && (x == 0 || x == clothW - 1)) {
        P[id].pinned = true;
        P[id].pinPos = P[id].p;
        P[id].w = 0.0f;
      }
      levelParticles[0].push_back(id);
      coarseFromFine[id] = id;
    }
  }

  auto addEdge = [&](int a, int b) {
    constraints.push_back({a, b, length(P[a].p - P[b].p), 0});
  };
  for (int y = 0; y < clothH; ++y) {
    for (int x = 0; x < clothW; ++x) {
      if (x + 1 < clothW) addEdge(idx(x, y), idx(x + 1, y));
      if (y + 1 < clothH) addEdge(idx(x, y), idx(x, y + 1));
    }
  }
}

// ---------- Hierarchy build ----------
void buildHierarchy() {
  for (int l = 1; l <= Lmax; ++l) {
    int step = 1 << l;
    if ((int)levelParticles.size() <= l) levelParticles.resize(l + 1);
    levelParticles[l].clear();

    std::vector<int> mapL_level(clothW * clothH, -1);

    int coarseW = (clothW + step - 1) / step;
    int coarseH = (clothH + step - 1) / step;

    auto fx_of = [&](int cx) { return (cx == coarseW - 1) ? (clothW - 1) : cx * step; };
    auto fy_of = [&](int cy) { return (cy == coarseH - 1) ? (clothH - 1) : cy * step; };

    for (int cy = 0; cy < coarseH; ++cy) {
      for (int cx = 0; cx < coarseW; ++cx) {
        int fx = fx_of(cx);
        int fy = fy_of(cy);
        int finest = idx(fx, fy);
        if (mapL_level[finest] == -1) {
          Particle parent = P[finest];
          parent.level = l;

          bool isTop = (fy == 0);
          bool isTopLeft  = isTop && (fx == 0);
          bool isTopRight = isTop && (fx == clothW - 1);
          parent.pinned = (isTopLeft || isTopRight);
          parent.w = parent.pinned ? 0.0f : 1.0f;
          if (parent.pinned) parent.pinPos = parent.p;

          int pid = (int)P.size();
          P.push_back(parent);
          levelParticles[l].push_back(pid);
          mapL_level[finest] = pid;

          if ((int)coarseFromFine.size() < (int)P.size())
            coarseFromFine.resize(P.size(), -1);
          coarseFromFine[pid] = finest;
        }
      }
    }

    auto coarseIndex = [&](int cx, int cy) -> int {
      int fx = fx_of(cx);
      int fy = fy_of(cy);
      return mapL_level[idx(fx, fy)];
    };
    for (int cy = 0; cy < coarseH; ++cy) {
      for (int cx = 0; cx < coarseW; ++cx) {
        int a = coarseIndex(cx, cy);
        if (a < 0) continue;
        if (cx + 1 < coarseW) {
          int b = coarseIndex(cx + 1, cy);
          if (b >= 0) constraints.push_back({a, b, length(P[a].p - P[b].p), l});
        }
        if (cy + 1 < coarseH) {
          int b = coarseIndex(cx, cy + 1);
          if (b >= 0) constraints.push_back({a, b, length(P[a].p - P[b].p), l});
        }
      }
    }
  }

  const float eps = 1e-6f;
  for (int l = Lmax; l >= 1; --l) {
    for (int i : levelParticles[l - 1]) {
      P[i].parents.clear();
      P[i].wij.clear();
      std::vector<std::pair<float, int>> cand;
      for (int pj : levelParticles[l]) {
        cand.push_back({length(P[pj].p - P[i].p), pj});
      }
      std::sort(cand.begin(), cand.end(),
                [](auto &a, auto &b) { return a.first < b.first; });
      int k = std::min(2, (int)cand.size());
      float denom = 0.f;
      for (int t = 0; t < k; ++t) {
        float dij = cand[t].first;
        float wij = 1.0f / (dij + eps);
        P[i].parents.push_back(cand[t].second);
        P[i].wij.push_back(wij);
        denom += wij;
      }
      if (denom > 0) for (float &w : P[i].wij) w /= denom;
    }
  }
}

// ---------- Constraint projection ----------
void projectConstraint(const Constraint &Cst) {
  Particle &A = P[Cst.i];
  Particle &B = P[Cst.j];
  vec3 n = A.p - B.p;
  float dist = length(n);
  if (dist < 1e-8f) return;
  n /= dist;
  float Cval = dist - Cst.rest;
  float s = A.w * dot(n, n) + B.w * dot(n, n);
  if (s <= 1e-12f) return;
  float lambda = -Cval / s;
  vec3 dpi = (A.w * lambda) * n;
  vec3 dpj = (B.w * lambda) * (-n);
  if (!A.pinned) A.p += dpi;
  if (!B.pinned) B.p += dpj;
}

void solveLevel(int level, int iterations) {
  for (int it = 0; it < iterations; ++it) {
    for (const auto &c : constraints)
      if (c.level == level) projectConstraint(c);
    for (int i : levelParticles[level])
      if (P[i].pinned) P[i].p = P[i].pinPos;
  }
}

// ---------- Hierarchical solver (with restriction + prolongation) ----------
void hierarchicalSolve(int solverItersPerLevel = 2) {
  for (int l = Lmax; l >= 0; --l) {
    if (l > 0) {
      for (int i : levelParticles[l]) {
        int src = (i < (int)coarseFromFine.size()) ? coarseFromFine[i] : -1;
        if (src >= 0) P[i].p = P[src].p;
      }
    }
    for (int i : levelParticles[l]) P[i].q = P[i].p;
    solveLevel(l, solverItersPerLevel);
    if (l > 0) {
      for (int i : levelParticles[l - 1]) {
        vec3 corr(0);
        for (size_t t = 0; t < P[i].parents.size(); ++t) {
          int pj = P[i].parents[t];
          float wij = P[i].wij[t];
          corr += wij * (P[pj].p - P[pj].q);
        }
        if (!P[i].pinned) P[i].p += corr;
      }
    }
  }
}

// ---------- Metrics/CSV (header-only) ----------
#include "hpbd_metrics.h"

CsvLogger g_logger;

std::vector<vec3> prevX;

void simulate() {
  prevX.resize(P.size());
  for (size_t i = 0; i < P.size(); ++i) prevX[i] = P[i].p;

  // integrate only level-0
  for (auto &a : P)
    if (!a.pinned && a.level == 0) {
      a.v += dt * g;
      a.p += dt * a.v;
    }

  if (g_useHierarchy) {
    hierarchicalSolve(g_itersHier);
  } else {
    solveLevel(0, g_itersPlain);
  }

  for (size_t i = 0; i < P.size(); ++i) {
    vec3 xnew = P[i].p;
    P[i].v = (xnew - prevX[i]) / dt;
  }
}

void drawCloth() {
  glColor3f(0.9f, 0.1f, 0.1f);
  glBegin(GL_LINES);
  for (const auto &c : constraints)
    if (c.level == 0) {
      glVertex3fv(glm::value_ptr(P[c.i].p));
      glVertex3fv(glm::value_ptr(P[c.j].p));
    }
  glEnd();

  if (g_useHierarchy) {
    for (int l = 1; l <= Lmax; ++l) {
      glPointSize(4.0f);
      glBegin(GL_POINTS);
      for (int i : levelParticles[l]) {
        glColor3f(0.2f + 0.3f * l, 0.6f - 0.2f * l, 1.0f - 0.3f * l);
        glVertex3fv(glm::value_ptr(P[i].p));
      }
      glEnd();
    }
  }
}

void display() {
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  glTranslatef(camPan.x, camPan.y, -camDist);
  glRotatef(camPitch * 180.0f / 3.14159265f, 1, 0, 0);
  glRotatef(camYaw   * 180.0f / 3.14159265f, 0, 1, 0);

  drawCloth();

  glutSwapBuffers();
}

void reshape(int w, int h) {
  glViewport(0, 0, w, h);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(45.0, (double)w / h, 0.01, 10.0);
}

void idle(void){
  GLfloat time = (float)glutGet(GLUT_ELAPSED_TIME) / 1000.0f;

  hpbdperf::sim_begin();
  simulate();
  hpbdperf::sim_end();

  glutSetWindowTitle(hpbdperf::title_with_ms("Hierarchical Position-Based Dynamics"));

  while(1) {
    if (((float)glutGet(GLUT_ELAPSED_TIME) / 1000.0f - time) > dt) {
      break; // keep fps
    }
  }
  glutPostRedisplay();
}

void mouseButton(int button, int state, int x, int y) {
  if (button == GLUT_LEFT_BUTTON)  lbtn = (state == GLUT_DOWN);
  if (button == GLUT_RIGHT_BUTTON) rbtn = (state == GLUT_DOWN);
  lastMouseX = x;
  lastMouseY = y;
}

void mouseMotion(int x, int y) {
  int dx = x - lastMouseX;
  int dy = y - lastMouseY;
  lastMouseX = x;
  lastMouseY = y;

  if (lbtn) {
    camYaw   += dx * 0.005f;
    camPitch += dy * 0.005f;
    const float lim = 1.5f;
    if (camPitch >  lim) camPitch =  lim;
    if (camPitch < -lim) camPitch = -lim;
  }
  if (rbtn) {
    float s = 0.002f * camDist;
    camPan.x += dx * s;
    camPan.y -= dy * s;
  }
}

void mouseWheel(int wheel, int direction, int x, int y) {
  camDist *= (direction > 0) ? 0.9f : 1.1f;
  if (camDist < 0.5f) camDist = 0.5f;
  if (camDist > 6.0f) camDist = 6.0f;
}

// ---------- Reset ----------
void resetSim() {
  constraints.clear();
  levelParticles.clear();
  buildLevel0();
  buildHierarchy();
}

// Over-stretch the level-0 mesh by pure gravity prediction (no solves)
static void overstretchTo(double targetPct, int maxSteps = 12) {
  for (int s = 0; s < maxSteps && computeRelStretchLmaxpct() < targetPct; ++s) {
    for (auto &a : P) if (!a.pinned && a.level == 0) {
      a.v += dt * g;
      a.p += dt * a.v;
    }
  }
}

// ---------- Helpers for Figure-3 style sweep ----------
static int runUntilThreshold_L0(double target_pct, int max_iters) {
  for (int it = 1; it <= max_iters; ++it) {
    solveLevel(0, 1);
    if (computeRelStretchLmaxpct() <= target_pct) return it;
  }
  return max_iters + 1;
}

static int runUntilThreshold_HIER(double target_pct, int max_iters, int itersPerLevel) {
  for (int it = 1; it <= max_iters; ++it) {
    hierarchicalSolve(itersPerLevel);  // one V-cycle per iteration
    if (computeRelStretchLmaxpct() <= target_pct) return it;
  }
  return max_iters + 1;
}

// ---------- Benchmark (replaced): write paper_fig3.csv with L0 and L3 ----------
void runBenchmark() {
  std::puts("[paper] Figure-3 style sweep (levels 0 and 3)");

  const int max_iters = 30;        // stop if not reached earlier
  const int itersPerLevel = 2;     // coarse/fine iterations per level in a V-cycle
  const int thresholdsArr[] = {25,23,21,19,17,15,13,11,9,7,5};

  std::ofstream ofs("paper_fig3.csv");
  if (!ofs) { std::perror("paper_fig3.csv"); return; }
  ofs << "threshold_pct,iters_L0,iters_L3\n";

  for (int th : thresholdsArr) {
    // L = 0
    Lmax = 0;
    resetSim();
    overstretchTo(30.0);
    int it0 = runUntilThreshold_L0(th, max_iters);

    // L = 3
    Lmax = 3;
    resetSim();
    overstretchTo(30.0);
    int it3 = runUntilThreshold_HIER(th, max_iters, itersPerLevel);

    ofs << th << "," << it0 << "," << it3 << "\n";
    std::printf("  threshold %2d%% -> L0:%d  L3:%d\n", th, it0, it3);
  }

  ofs.close();
  std::puts("[paper] wrote paper_fig3.csv (threshold_pct,iters_L0,iters_L3)");
}

// ---------- Keyboard ----------
void keyboard(unsigned char key, int, int) {
  auto adjustActiveIters = [&](int delta){
    int &it = (g_useHierarchy ? g_itersHier : g_itersPlain);
    const int lo = 1;
    const int hi = g_useHierarchy ? 10 : 60;
    it = std::clamp(it + delta, lo, hi);
    std::printf("%s=%d\n", g_useHierarchy ? "itersHier" : "itersPlain", it);
  };
  switch (key) {
    case 'h': case 'H':
      g_useHierarchy = !g_useHierarchy;
      std::printf("[Toggle] Hierarchy = %s\n", g_useHierarchy ? "ON" : "OFF");
      break;
    case ']': adjustActiveIters(+1); break;
    case '[': adjustActiveIters(-1); break;
    case 'b': case 'B':
      runBenchmark();
      break;
    case 'r': case 'R':
      resetSim();
      break;
    case '+':
      camDist *= 0.9f; if (camDist < 0.5f) camDist = 0.5f; break;
    case '-':
      camDist *= 1.1f; if (camDist > 6.0f) camDist = 6.0f; break;
    case 27:
#if defined(FREEGLUT)
      glutLeaveMainLoop();
#else
      std::exit(0);
#endif
      break;
  }
}

void usage() {
  std::puts("=== Hierarchical Position-Based Dynamics Controls ===");
  std::printf("Grid: %dx%d, Lmax=%d, dt=%.4f s\n", clothW, clothH, Lmax, dt);
  std::printf("Mode: Hierarchy=%s  (iters: hier=%d, plain=%d)\n",
              g_useHierarchy ? "ON" : "OFF", g_itersHier, g_itersPlain);

  std::puts("\nMouse:");
  std::puts("  Left-drag   : rotate camera (yaw/pitch)");
  std::puts("  Right-drag  : pan");
  std::puts("  Wheel       : zoom");

  std::puts("\nKeyboard:");
  std::puts("  H           : toggle hierarchy ON/OFF");
  std::puts("  [ / ]       : decrement / increment iterations of the ACTIVE mode");
  std::puts("  R           : reset simulation");
  std::puts("  B           : run Figure.3 benchmark (writes fig3.csv)");
  std::puts("  + / -       : zoom in / out");
  std::puts("  ESC         : quit");

  std::puts("\nBenchmark:");
  std::puts("  - X-axis: Relative stretch (%) = {25,23,...,5} (decreasing)");
  std::puts("  - Y-axis: Number of iterations");
  std::puts("  - Series: L=0 vs L=3 (hierarchical V-cycles)");
  std::puts("  Output : fig3.csv");

  std::puts("\nNotes:");
  std::puts("  - Window title shows last simulate() time in milliseconds.");
  std::puts("  - Fixed time step = 1/60 sec;\n");
}

void init() {
  buildLevel0();
  buildHierarchy();
  usage();
}

int main(int argc, char **argv) {
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
  glutInitWindowSize(960, 600);
  glutCreateWindow("Hierarchical Position-Based Dynamics");

  glEnable(GL_DEPTH_TEST);
  glClearColor(0.1f, 0.1f, 0.1f, 1);

  init();

  glutDisplayFunc(display);
  glutReshapeFunc(reshape);
  glutIdleFunc(idle);

  glutMouseFunc(mouseButton);
  glutMotionFunc(mouseMotion);
#if defined(FREEGLUT)
  glutMouseWheelFunc(mouseWheel);
#endif
  glutKeyboardFunc(keyboard);

  glutMainLoop();
  return 0;
}

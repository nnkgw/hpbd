// Hierarchical Position-Based Dynamics
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

using glm::vec2;
using glm::vec3;
using glm::dot;
using glm::length;
using glm::normalize;

struct Constraint {
  int i, j;      // particle indices
  float rest;    // rest length
  int level;     // hierarchy level
};

struct Particle {
  vec3 p;                 // position
  vec3 v;                 // velocity
  vec3 q;                 // saved position q_l (Alg. Sec.7 step 2)
  float w;                // inverse mass (w_i = 1/m_i)
  int level;              // finest level this particle belongs to
  std::vector<int> parents;   // indices of parent particles P(i)
  std::vector<float> wij;     // weights w_ij (Eq.(6))
  bool pinned = false;    // cardinality-1 constraint
  vec3 pinPos;            // pin target
};

static const float dt = 1.0f / 60.0f;
static const vec3 g(0.0f, -9.8f, 0.0f);
static const int clothW = 30, clothH = 20;
static const float spacing = 0.05f;

std::vector<Particle> P;
std::vector<std::vector<int>> levelParticles;
std::vector<Constraint> constraints;
int Lmax = 2;
std::vector<int> coarseFromFine;

// --------- Camera -----------
float camDist = 2.0f;
float camYaw = 0.0f;
float camPitch = -30.0f * 3.14159265f / 180.0f;
vec2 camPan(0.0f, -0.3f);
int lastMouseX = 0, lastMouseY = 0;
bool lbtn = false, rbtn = false;
// ----------------------------

// --------- Toggles ----------
bool g_useHierarchy = true;   // toggle with 'H'
int  g_itersHier    = 3;      // iterations per level when hierarchy ON
int  g_itersPlain   = 3/*20*/;     // iterations for level-0 when hierarchy OFF
// ----------------------------

inline int idx(int x, int y) { return y * clothW + x; }

void buildLevel0() {
  P.clear();
  P.resize(clothW * clothH);
  levelParticles.assign(Lmax + 1, {});
  constraints.clear();

  // NEW: initialize mapping for level-0
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

      // NEW: level-0 particle maps to itself
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

          // NEW: remember which finest index this coarse particle comes from
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

void hierarchicalSolve(int solverItersPerLevel = 2) {
  for (int l = Lmax; l >= 0; --l) {

    // NEW: restriction — sync coarse positions from their source finest indices
    if (l > 0) {
      for (int i : levelParticles[l]) {
        int src = (i < (int)coarseFromFine.size()) ? coarseFromFine[i] : -1;
        if (src >= 0) P[i].p = P[src].p;
      }
    }

    // save q_l (pre-projection positions on this level)
    for (int i : levelParticles[l]) P[i].q = P[i].p;

    // project constraints on this level
    solveLevel(l, solverItersPerLevel);

    // prolongation — propagate corrections (p - q) down to level l-1
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


std::vector<vec3> prevX;

void simulate() {
  prevX.resize(P.size());
  for (size_t i = 0; i < P.size(); ++i) prevX[i] = P[i].p;

  // integrate only finest level (original DoFs)
  for (auto &a : P)
    if (!a.pinned && a.level == 0) {
      a.v += dt * g;
      a.p += dt * a.v;
    }

  // toggle between hierarchical and plain PBD
  if (g_useHierarchy) {
    hierarchicalSolve(g_itersHier);
  } else {
    solveLevel(0, g_itersPlain);
  }

  // reconstruct velocities from displacement
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

  // draw higher levels only when hierarchy toggle is ON
  if (g_useHierarchy) {
    for (int l = 1; l <= Lmax; ++l) {
      float zoff = 0.003f * l;
      glPointSize(4.0f + 2.0f * l);
      glTranslatef(0.f, 0.f, zoff);
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

  simulate();
  drawCloth();

  glutSwapBuffers();
}

void reshape(int w, int h) {
  glViewport(0, 0, w, h);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(45.0, (double)w / h, 0.01, 10.0);
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

// reset simulation to initial state
void resetSim() {
  constraints.clear();
  levelParticles.clear();
  buildLevel0();
  buildHierarchy();
}

// keyboard controls (toggle & params)
void keyboard(unsigned char key, int, int) {
  switch (key) {
    case 'h': case 'H':
      g_useHierarchy = !g_useHierarchy;
      std::printf("[Toggle] Hierarchy = %s\n", g_useHierarchy ? "ON" : "OFF");
      break;
    case ']':
      g_itersHier = std::min(g_itersHier + 1, 10);
      std::printf("itersHier=%d\n", g_itersHier);
      break;
    case '[':
      g_itersHier = std::max(g_itersHier - 1, 1);
      std::printf("itersHier=%d\n", g_itersHier);
      break;
    case '}':
      g_itersPlain = std::min(g_itersPlain + 1, 60);
      std::printf("itersPlain=%d\n", g_itersPlain);
      break;
    case '{':
      g_itersPlain = std::max(g_itersPlain - 1, 1);
      std::printf("itersPlain=%d\n", g_itersPlain);
      break;
    case 'r': case 'R':
      resetSim();
      break;
    case '+':
      camDist *= 0.9f; if (camDist < 0.5f) camDist = 0.5f; break;
    case '-':
      camDist *= 1.1f; if (camDist > 6.0f) camDist = 6.0f; break;
  }
}

int main(int argc, char **argv) {
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
  glutInitWindowSize(960, 600);
  glutCreateWindow("Hierarchical Position-Based Dynamics");

  glEnable(GL_DEPTH_TEST);
  glClearColor(0.1f, 0.1f, 0.1f, 1);

  buildLevel0();
  buildHierarchy();

  glutDisplayFunc(display);
  glutReshapeFunc(reshape);
  glutIdleFunc(display);

  glutMouseFunc(mouseButton);
  glutMotionFunc(mouseMotion);
#if defined(FREEGLUT)
  glutMouseWheelFunc(mouseWheel);
#endif
  glutKeyboardFunc(keyboard);

  glutMainLoop();
  return 0;
}

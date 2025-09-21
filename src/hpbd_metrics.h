// hpbd_metrics.h - header-only metrics & CSV logging for HPBD
#pragma once
#include <cstdio>
#include <chrono>
#include <vector>
#include <cmath>
#include <algorithm>
#include <glm/glm.hpp>

// Externs provided by hpbd.cpp
struct Particle;
struct Constraint;
extern std::vector<Particle> P;
extern std::vector<Constraint> constraints;
extern int Lmax;

// Extern solvers from hpbd.cpp
void solveLevel(int level, int iterations);
void hierarchicalSolve(int solverItersPerLevel);

// ---------- Residuals on level-0 edges ----------
struct Residual {
  double L1{0.0}, RMS{0.0}, Lmax{0.0};
  int n{0};
};

inline Residual computeResidualL0() {
  using glm::length;
  Residual r;
  double s1=0.0, s2=0.0, smax=0.0; int m=0;
  for (const auto& c : constraints) if (c.level == 0) {
    double d = length(P[c.i].p - P[c.j].p);
    double e = std::abs(d - c.rest);
    s1 += e; s2 += e*e; smax = std::max(smax, e); ++m;
  }
  r.n   = m;
  r.L1  = (m ? s1 / m : 0.0);
  r.RMS = (m ? std::sqrt(s2 / m) : 0.0);
  r.Lmax= smax;
  return r;
}

inline int countConstraintsOnLevel(int level) {
  int k=0; for (const auto& c: constraints) if (c.level==level) ++k; return k;
}

// ---------- Relative stretch (%) in paper style ----------
inline double computeRelStretchL1pct() {
  using glm::length;
  double sum = 0.0; int m = 0;
  for (const auto& c : constraints) if (c.level == 0) {
    double d = length(P[c.i].p - P[c.j].p);
    double ext = d > c.rest ? (d - c.rest) / c.rest * 100.0 : 0.0;
    sum += ext; ++m;
  }
  return (m ? sum / m : 0.0);
}

inline double computeRelStretchLmaxpct() {
  using glm::length;
  double smax = 0.0;
  for (const auto& c : constraints) if (c.level == 0) {
    double d = length(P[c.i].p - P[c.j].p);
    double ext = d > c.rest ? (d - c.rest) / c.rest * 100.0 : 0.0;
    if (ext > smax) smax = ext;
  }
  return smax;
}

// ---------- Minimal CSV logger (RAII) ----------
struct CsvLogger {
  std::FILE* f{nullptr};
  void open(const char* path) {
    close();
    f = std::fopen(path, "w");
    if (f) std::fprintf(f, "mode,level,iterG,iterL,cycles,time_ms,res_L1,res_RMS,res_max,proj_accum\n");
  }
  void close() { if (f) { std::fclose(f); f=nullptr; } }
  ~CsvLogger(){ close(); }

  void row(const char* mode, int level, int iterG, int iterL, int cycles,
           long long ms, const Residual& r, int projAccum) {
    if (!f) return;
    std::fprintf(f, "%s,%d,%d,%d,%d,%lld,%.9g,%.9g,%.9g,%d\n",
      mode, level, iterG, iterL, cycles, ms, r.L1, r.RMS, r.Lmax, projAccum);
  }
};

// ---------- Logged wrappers (keep core solvers untouched) ----------
namespace hpbdlog {

  inline void solveLevelLogged(int level, int iterations, CsvLogger& log, const char* modeTag="plain") {
    using clock = std::chrono::steady_clock;
    static int iterG = 0;           // global iteration count across calls
    static int projA = 0;           // accumulated projection count
    auto t0 = clock::now();
    for (int it=0; it<iterations; ++it) {
      solveLevel(level, 1);
      projA += countConstraintsOnLevel(level);
      auto t1 = clock::now();
      auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
      log.row(modeTag, level, ++iterG, it+1, /*cycles*/0, ms, computeResidualL0(), projA);
    }
  }

  inline void hierarchicalCyclesLogged(int cycles, int itersPerLevel, CsvLogger& log, const char* modeTag="hier") {
    using clock = std::chrono::steady_clock;
    static int iterG = 0;
    static int projA = 0;
    auto t0 = clock::now();
    for (int c=1; c<=cycles; ++c) {
      hierarchicalSolve(itersPerLevel);
      int work = 0;
      for (int l=0; l<=Lmax; ++l) work += countConstraintsOnLevel(l) * itersPerLevel;
      projA += work;
      auto t1 = clock::now();
      auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
      log.row(modeTag, /*level*/-1, ++iterG, /*iterL*/itersPerLevel, c, ms, computeResidualL0(), projA);
    }
  }

} // namespace hpbdlog

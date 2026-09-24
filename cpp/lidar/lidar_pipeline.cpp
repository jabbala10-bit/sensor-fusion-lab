// lidar_pipeline.cpp - Lidar obstacle detection from scratch (C++17, no PCL).
// Stages: ray-casting lidar simulator -> voxel grid -> ROI crop -> RANSAC ground plane
//         (+ least-squares refinement) -> KD-tree -> Euclidean clustering -> bounding boxes.
// Build:  g++ -std=c++17 -O2 lidar_pipeline.cpp -o lidar_pipeline
// Run:    ./lidar_pipeline [labels.csv]      (optional CSV: x,y,z,label ; label -1 = ground)
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

struct Point { float x, y, z; };
using Cloud = std::vector<Point>;

struct Box {  // axis-aligned box: ground-truth objects and detected clusters
  std::string name;
  float xmin, ymin, zmin, xmax, ymax, zmax;
};

constexpr float kInf = std::numeric_limits<float>::infinity();
constexpr float kDeg = 3.14159265358979f / 180.f;

// ------------------------------------------------------------------------------------------
// 1. Lidar simulator: one ray per (layer, azimuth), nearest hit wins, Gaussian range noise.
// ------------------------------------------------------------------------------------------
struct LidarSpec {                 // loosely modelled on a 32-beam spinning lidar
  int   layers      = 32;
  float minElevDeg  = -24.8f, maxElevDeg = 2.0f;
  float azResDeg    = 0.4f;        // 900 azimuth steps per revolution
  float minRange    = 0.3f, maxRange = 60.f;
  float mountHeight = 1.73f;       // sensor frame origin, ground plane at z = -mountHeight
  float rangeSigma  = 0.02f;       // 1-sigma range noise [m]
};

// Slab-method ray/AABB intersection for a ray starting at the sensor origin.
float rayBox(const Point& d, const Box& b) {
  float tmin = 0.f, tmax = kInf;
  const float dir[3] = {d.x, d.y, d.z};
  const float lo[3] = {b.xmin, b.ymin, b.zmin}, hi[3] = {b.xmax, b.ymax, b.zmax};
  for (int i = 0; i < 3; ++i) {
    if (std::fabs(dir[i]) < 1e-9f) {             // ray parallel to this slab
      if (0.f < lo[i] || 0.f > hi[i]) return kInf;
      continue;
    }
    float t1 = lo[i] / dir[i], t2 = hi[i] / dir[i];
    if (t1 > t2) std::swap(t1, t2);
    tmin = std::max(tmin, t1);
    tmax = std::min(tmax, t2);
    if (tmin > tmax) return kInf;
  }
  return tmin;
}

Cloud simulateScan(const LidarSpec& s, const std::vector<Box>& scene, std::mt19937& rng) {
  std::normal_distribution<float> noise(0.f, s.rangeSigma);
  const int nAz = static_cast<int>(std::round(360.f / s.azResDeg));
  Cloud cloud;
  cloud.reserve(static_cast<size_t>(s.layers) * nAz);
  for (int l = 0; l < s.layers; ++l) {
    const float elev = (s.minElevDeg + (s.maxElevDeg - s.minElevDeg) * l / (s.layers - 1)) * kDeg;
    for (int a = 0; a < nAz; ++a) {
      const float az = (-180.f + a * s.azResDeg) * kDeg;
      const Point d{std::cos(elev) * std::cos(az), std::cos(elev) * std::sin(az), std::sin(elev)};
      float t = (d.z < 0.f) ? -s.mountHeight / d.z : kInf;  // ground plane z = -h
      for (const Box& b : scene) t = std::min(t, rayBox(d, b));
      if (t < s.minRange || t > s.maxRange) continue;        // no return
      t += noise(rng);
      cloud.push_back({t * d.x, t * d.y, t * d.z});
    }
  }
  return cloud;
}

// ------------------------------------------------------------------------------------------
// 2. Filtering: voxel-grid downsampling (centroid per voxel) and region-of-interest crop.
// ------------------------------------------------------------------------------------------
Cloud voxelGrid(const Cloud& in, float leaf) {
  struct Acc { double x = 0, y = 0, z = 0; int n = 0; };
  std::unordered_map<uint64_t, Acc> grid;
  grid.reserve(in.size());
  const int64_t off = 1 << 20;                   // 21 bits per axis
  for (const Point& p : in) {
    const uint64_t ix = static_cast<uint64_t>(static_cast<int64_t>(std::floor(p.x / leaf)) + off);
    const uint64_t iy = static_cast<uint64_t>(static_cast<int64_t>(std::floor(p.y / leaf)) + off);
    const uint64_t iz = static_cast<uint64_t>(static_cast<int64_t>(std::floor(p.z / leaf)) + off);
    Acc& a = grid[(ix << 42) | (iy << 21) | iz];
    a.x += p.x; a.y += p.y; a.z += p.z; ++a.n;
  }
  Cloud out;
  out.reserve(grid.size());
  for (const auto& kv : grid) {
    const Acc& a = kv.second;
    out.push_back({float(a.x / a.n), float(a.y / a.n), float(a.z / a.n)});
  }
  return out;
}

inline bool inside(const Point& p, const Box& b) {
  return p.x >= b.xmin && p.x <= b.xmax && p.y >= b.ymin && p.y <= b.ymax &&
         p.z >= b.zmin && p.z <= b.zmax;
}

Cloud cropBox(const Cloud& in, const Box& roi, const Box& egoRoof) {
  Cloud out;
  out.reserve(in.size());
  for (const Point& p : in)
    if (inside(p, roi) && !inside(p, egoRoof)) out.push_back(p);
  return out;
}

// ------------------------------------------------------------------------------------------
// 3. Segmentation: RANSAC plane with adaptive iteration count + least-squares refinement.
// ------------------------------------------------------------------------------------------
struct Plane { float a, b, c, d; };              // a x + b y + c z + d = 0, |(a,b,c)| = 1

inline float planeDist(const Plane& pl, const Point& p) {
  return std::fabs(pl.a * p.x + pl.b * p.y + pl.c * p.z + pl.d);
}

std::vector<int> ransacPlane(const Cloud& cloud, int maxIter, float distTol, float maxTiltDeg,
                             std::mt19937& rng, Plane& best, int& itersUsed) {
  std::vector<int> bestInliers;
  itersUsed = 0;
  if (cloud.size() < 3) return bestInliers;
  std::uniform_int_distribution<size_t> pick(0, cloud.size() - 1);
  const float minNz = std::cos(maxTiltDeg * kDeg);
  int needed = maxIter;
  int it = 0;
  for (; it < needed; ++it) {
    const Point& p1 = cloud[pick(rng)];
    const Point& p2 = cloud[pick(rng)];
    const Point& p3 = cloud[pick(rng)];
    const float ux = p2.x - p1.x, uy = p2.y - p1.y, uz = p2.z - p1.z;
    const float vx = p3.x - p1.x, vy = p3.y - p1.y, vz = p3.z - p1.z;
    float a = uy * vz - uz * vy, b = uz * vx - ux * vz, c = ux * vy - uy * vx;  // u x v
    const float n = std::sqrt(a * a + b * b + c * c);
    if (n < 1e-6f) continue;                     // degenerate: repeated or collinear sample
    a /= n; b /= n; c /= n;
    if (std::fabs(c) < minNz) continue;          // ground prior: normal close to vertical
    const Plane pl{a, b, c, -(a * p1.x + b * p1.y + c * p1.z)};
    std::vector<int> inl;
    for (int i = 0; i < static_cast<int>(cloud.size()); ++i)
      if (planeDist(pl, cloud[i]) <= distTol) inl.push_back(i);
    if (inl.size() > bestInliers.size()) {
      bestInliers.swap(inl);
      best = pl;
      // Adaptive stop: N = log(1 - p) / log(1 - w^s), p = 0.99, s = 3 points per sample.
      const double w = double(bestInliers.size()) / cloud.size();
      const double denom = std::log(1.0 - w * w * w);
      if (denom < 0.0) needed = std::min(maxIter, static_cast<int>(std::ceil(std::log(0.01) / denom)));
    }
  }
  itersUsed = it;
  return bestInliers;
}

// Least-squares fit z = alpha x + beta y + gamma on centred inliers (curve fitting step).
Plane refinePlaneLS(const Cloud& cloud, const std::vector<int>& idx) {
  double mx = 0, my = 0, mz = 0;
  for (int i : idx) { mx += cloud[i].x; my += cloud[i].y; mz += cloud[i].z; }
  mx /= idx.size(); my /= idx.size(); mz /= idx.size();
  double sxx = 0, sxy = 0, syy = 0, sxz = 0, syz = 0;
  for (int i : idx) {
    const double dx = cloud[i].x - mx, dy = cloud[i].y - my, dz = cloud[i].z - mz;
    sxx += dx * dx; sxy += dx * dy; syy += dy * dy; sxz += dx * dz; syz += dy * dz;
  }
  const double det = sxx * syy - sxy * sxy;
  const double alpha = (sxz * syy - sxy * syz) / det;
  const double beta  = (sxx * syz - sxy * sxz) / det;
  const double gamma = mz - alpha * mx - beta * my;
  const double norm = std::sqrt(alpha * alpha + beta * beta + 1.0);
  return {float(-alpha / norm), float(-beta / norm), float(1.0 / norm), float(-gamma / norm)};
}

// ------------------------------------------------------------------------------------------
// 4. KD-tree (3D, balanced by median split) with radius search.
// ------------------------------------------------------------------------------------------
class KdTree3 {
 public:
  explicit KdTree3(const Cloud& pts) : pts_(pts) {
    std::vector<int> ids(pts.size());
    std::iota(ids.begin(), ids.end(), 0);
    nodes_.reserve(pts.size());
    root_ = build(ids, 0, static_cast<int>(ids.size()), 0);
  }

  void radiusSearch(const Point& q, float r, std::vector<int>& out) const {
    out.clear();
    search(root_, q, r, 0, out);
  }

 private:
  struct Node { int idx, left, right; };
  static float coord(const Point& p, int axis) { return axis == 0 ? p.x : (axis == 1 ? p.y : p.z); }

  int build(std::vector<int>& ids, int lo, int hi, int depth) {
    if (lo >= hi) return -1;
    const int axis = depth % 3, mid = (lo + hi) / 2;
    std::nth_element(ids.begin() + lo, ids.begin() + mid, ids.begin() + hi,
                     [&](int a, int b) { return coord(pts_[a], axis) < coord(pts_[b], axis); });
    const int n = static_cast<int>(nodes_.size());
    nodes_.push_back({ids[mid], -1, -1});
    const int l = build(ids, lo, mid, depth + 1);
    const int r = build(ids, mid + 1, hi, depth + 1);
    nodes_[n].left = l;                          // index access: safe after reallocation
    nodes_[n].right = r;
    return n;
  }

  void search(int n, const Point& q, float r, int depth, std::vector<int>& out) const {
    if (n < 0) return;
    const Point& p = pts_[nodes_[n].idx];
    const float dx = p.x - q.x, dy = p.y - q.y, dz = p.z - q.z;
    if (std::fabs(dx) <= r && std::fabs(dy) <= r && std::fabs(dz) <= r &&   // cheap box test
        dx * dx + dy * dy + dz * dz <= r * r)                                  // exact test
      out.push_back(nodes_[n].idx);
    const int axis = depth % 3;
    if (coord(q, axis) - r < coord(p, axis)) search(nodes_[n].left, q, r, depth + 1, out);
    if (coord(q, axis) + r > coord(p, axis)) search(nodes_[n].right, q, r, depth + 1, out);
  }

  const Cloud& pts_;
  std::vector<Node> nodes_;
  int root_ = -1;
};

// ------------------------------------------------------------------------------------------
// 5. Euclidean clustering (region growing over the KD-tree) and bounding boxes.
// ------------------------------------------------------------------------------------------
// tol(p) = max(tolMin, tolPerMeter * horizontal range): tolPerMeter = 0 gives the classic fixed
// tolerance; > 0 compensates for beams spreading apart with range (and at grazing incidence).
std::vector<std::vector<int>> euclideanCluster(const Cloud& pts, const KdTree3& tree, float tolMin,
                                               float tolPerMeter, int minSize, int maxSize) {
  std::vector<char> processed(pts.size(), 0);
  std::vector<std::vector<int>> clusters;
  std::vector<int> nbrs, stack;
  for (int i = 0; i < static_cast<int>(pts.size()); ++i) {
    if (processed[i]) continue;
    std::vector<int> cluster;
    stack.push_back(i);
    processed[i] = 1;
    while (!stack.empty()) {
      const int id = stack.back();
      stack.pop_back();
      cluster.push_back(id);
      const float tol = std::max(tolMin, tolPerMeter * std::hypot(pts[id].x, pts[id].y));
      tree.radiusSearch(pts[id], tol, nbrs);
      for (int nb : nbrs)
        if (!processed[nb]) { processed[nb] = 1; stack.push_back(nb); }
    }
    const int sz = static_cast<int>(cluster.size());
    if (sz >= minSize && sz <= maxSize) clusters.push_back(std::move(cluster));
  }
  return clusters;
}

Box boundingBox(const Cloud& pts, const std::vector<int>& idx) {
  Box b{"", kInf, kInf, kInf, -kInf, -kInf, -kInf};
  for (int i : idx) {
    b.xmin = std::min(b.xmin, pts[i].x); b.xmax = std::max(b.xmax, pts[i].x);
    b.ymin = std::min(b.ymin, pts[i].y); b.ymax = std::max(b.ymax, pts[i].y);
    b.zmin = std::min(b.zmin, pts[i].z); b.zmax = std::max(b.zmax, pts[i].z);
  }
  return b;
}

// ------------------------------------------------------------------------------------------
int main(int argc, char** argv) {
  using Clock = std::chrono::steady_clock;
  auto ms = [](Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };
  std::mt19937 rng(42);
  const LidarSpec spec;
  const float g = -spec.mountHeight;             // ground height in the sensor frame

  // Scene in the sensor frame: x forward, y left, z up. Objects stand on the ground.
  const std::vector<Box> objects = {
      {"car_ahead",   12.0f, -0.9f, g, 16.5f,  0.9f, g + 1.5f},
      {"car_left",     6.0f,  2.6f, g, 10.5f,  4.4f, g + 1.5f},
      {"truck_right", 20.0f, -4.7f, g, 28.0f, -2.2f, g + 3.2f},
      {"car_behind", -14.0f, -0.9f, g, -9.5f,  0.9f, g + 1.5f},
      {"pedestrian",   9.0f,  6.5f, g,  9.5f,  7.0f, g + 1.8f},
      {"pole",        15.0f, -6.5f, g, 15.3f, -6.2f, g + 4.0f},
  };
  const Box egoRoof{"ego", -1.5f, -0.85f, g, 1.0f, 0.85f, -0.4f};  // roof 0.4 m below sensor
  std::vector<Box> scene = objects;
  scene.push_back(egoRoof);                      // the sensor sees its own roof

  auto t0 = Clock::now();
  const Cloud raw = simulateScan(spec, scene, rng);
  auto t1 = Clock::now();
  const Cloud vox = voxelGrid(raw, 0.15f);
  const Box roi{"roi", -20.f, -8.f, -2.5f, 40.f, 8.f, 1.5f};
  const Box egoCrop{"ego", -1.7f, -1.0f, -2.0f, 1.2f, 1.0f, 0.0f};
  const Cloud roiCloud = cropBox(vox, roi, egoCrop);
  auto t2 = Clock::now();

  Plane plane{};
  int iters = 0;
  std::vector<int> inl = ransacPlane(roiCloud, 200, 0.20f, 15.f, rng, plane, iters);
  const Plane ransacModel = plane;
  plane = refinePlaneLS(roiCloud, inl);          // refine, then re-select inliers
  std::vector<char> isGround(roiCloud.size(), 0);
  Cloud obstacles;
  std::vector<int> obstacleSrc;
  for (int i = 0; i < static_cast<int>(roiCloud.size()); ++i) {
    if (planeDist(plane, roiCloud[i]) <= 0.20f) { isGround[i] = 1; continue; }
    obstacles.push_back(roiCloud[i]);
    obstacleSrc.push_back(i);
  }
  auto t3 = Clock::now();

  const KdTree3 tree(obstacles);
  const auto fixedClusters = euclideanCluster(obstacles, tree, 0.6f, 0.0f, 5, 5000);
  auto t4 = Clock::now();
  const auto clusters = euclideanCluster(obstacles, tree, 0.5f, 0.09f, 5, 5000);  // adaptive

  std::printf("points  raw %zu | voxel(0.15 m) %zu | ROI + ego crop %zu\n", raw.size(), vox.size(),
              roiCloud.size());
  std::printf("RANSAC  %d iterations (adaptive), plane %.3fx %+.3fy %+.3fz %+.3f = 0\n", iters,
              ransacModel.a, ransacModel.b, ransacModel.c, ransacModel.d);
  std::printf("LS fit  plane %.3fx %+.3fy %+.3fz %+.3f = 0  -> height %.3f m (true %.3f)\n", plane.a,
              plane.b, plane.c, plane.d, -plane.d / plane.c, g);
  std::printf("ground  %zu points | obstacles %zu points\n", roiCloud.size() - obstacles.size(),
              obstacles.size());
  std::printf("cluster fixed tol 0.6 m -> %zu clusters | adaptive tol max(0.5, 0.09 r) -> %zu clusters\n\n",
              fixedClusters.size(), clusters.size());

  std::printf("%-3s %6s  %-26s %-20s %s\n", "id", "points", "box centre x,y,z [m]", "size L,W,H [m]",
              "matched object");
  for (size_t c = 0; c < clusters.size(); ++c) {
    const Box b = boundingBox(obstacles, clusters[c]);
    const float cx = 0.5f * (b.xmin + b.xmax), cy = 0.5f * (b.ymin + b.ymax), cz = 0.5f * (b.zmin + b.zmax);
    std::string match = "-";
    for (const Box& o : objects)
      if (cx >= o.xmin - 0.5f && cx <= o.xmax + 0.5f && cy >= o.ymin - 0.5f && cy <= o.ymax + 0.5f)
        match = o.name;
    std::printf("%-3zu %6zu  %6.2f %6.2f %6.2f       %5.2f %5.2f %5.2f        %s\n", c, clusters[c].size(),
                cx, cy, cz, b.xmax - b.xmin, b.ymax - b.ymin, b.zmax - b.zmin, match.c_str());
  }
  std::printf("\ntiming [ms]  simulate %.1f | filter %.1f | segment %.1f | cluster %.1f\n", ms(t0, t1),
              ms(t1, t2), ms(t2, t3), ms(t3, t4));

  if (argc > 1) {                                // optional export for plotting
    std::vector<int> label(roiCloud.size(), -2);
    for (size_t i = 0; i < roiCloud.size(); ++i) if (isGround[i]) label[i] = -1;
    for (size_t c = 0; c < clusters.size(); ++c)
      for (int k : clusters[c]) label[obstacleSrc[k]] = static_cast<int>(c);
    std::ofstream f(argv[1]);
    f << "x,y,z,label\n";
    for (size_t i = 0; i < roiCloud.size(); ++i)
      f << roiCloud[i].x << ',' << roiCloud[i].y << ',' << roiCloud[i].z << ',' << label[i] << '\n';
    std::printf("wrote %s (label -1 ground, -2 unclustered)\n", argv[1]);
  }
  return 0;
}

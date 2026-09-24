// camera_ttc.cpp - Camera + lidar time-to-collision on a synthetic approach sequence (OpenCV 4).
// Covers: pinhole projection, lidar-to-camera projection, Harris + NMS, OpenCV detectors and
// descriptors, brute-force kNN matching with Lowe's ratio test, ROI association, camera and lidar TTC.
// Build: g++ -std=c++17 -O2 camera_ttc.cpp -o camera_ttc $(pkg-config --cflags --libs opencv4)
// Run:   ./camera_ttc                  detailed AKAZE/AKAZE run + benchmark of all combinations
//        ./camera_ttc FAST BRISK       detailed run for one detector/descriptor pair
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv_modules.hpp>
#ifdef HAVE_OPENCV_XFEATURES2D
#include <opencv2/xfeatures2d.hpp>   // BRIEF, FREAK (opencv_contrib only)
#endif
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

// ------------------------------------------------------------------ camera and lidar geometry
struct Camera {                      // pinhole camera, KITTI-sized image
  double f = 1200.0, cx = 621.0, cy = 187.0;
  int width = 1242, height = 375;
  double zOffset = -0.08;            // camera centre 8 cm below the lidar, same x and y
};

struct LidarPoint { double x, y, z; };  // lidar frame: x forward, y left, z up

// Lidar frame -> camera frame (X right, Y down, Z forward) -> pixels:  u = f X/Z + cx, v = f Y/Z + cy
cv::Point2d project(const Camera& cam, const LidarPoint& p) {
  const double X = -p.y, Y = -(p.z - cam.zOffset), Z = p.x;
  return {cam.f * X / Z + cam.cx, cam.f * Y / Z + cam.cy};
}

// ------------------------------------------------------------------ synthetic scene
// Rear of the preceding car: 720 x 560 texture px = 1.8 m x 1.4 m (400 px per metre).
cv::Mat makeCarTexture(cv::RNG& rng) {
  cv::Mat t(560, 720, CV_8UC1, cv::Scalar(85));
  cv::rectangle(t, {70, 40}, {650, 230}, cv::Scalar(35), cv::FILLED);          // rear window
  for (int i = 0; i < 60; ++i) cv::line(t, {200 + i, 40}, {120 + i, 230}, cv::Scalar(62), 1);
  cv::rectangle(t, {25, 260}, {190, 340}, cv::Scalar(190), cv::FILLED);         // tail lights
  cv::rectangle(t, {530, 260}, {695, 340}, cv::Scalar(190), cv::FILLED);
  cv::rectangle(t, {45, 280}, {110, 320}, cv::Scalar(245), cv::FILLED);
  cv::rectangle(t, {610, 280}, {675, 320}, cv::Scalar(245), cv::FILLED);
  cv::rectangle(t, {250, 345}, {470, 415}, cv::Scalar(235), cv::FILLED);        // licence plate
  cv::putText(t, "KA01 SF26", {262, 397}, cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(20), 4);
  cv::circle(t, {360, 290}, 28, cv::Scalar(220), 5);                            // badge
  cv::rectangle(t, {0, 440}, {720, 560}, cv::Scalar(55), cv::FILLED);           // bumper
  cv::line(t, {0, 470}, {720, 470}, cv::Scalar(125), 4);
  for (int i = 0; i < 120; ++i) {                                                // stickers, dirt
    const cv::Point c(rng.uniform(0, 720), rng.uniform(0, 560));
    const int r = rng.uniform(5, 16);
    cv::rectangle(t, c - cv::Point(r, r), c + cv::Point(r, r / 2), cv::Scalar(rng.uniform(30, 235)), cv::FILLED);
  }
  return t;
}

cv::Mat makeBackground(const Camera& cam, cv::RNG& rng) {
  cv::Mat bg(cam.height, cam.width, CV_8UC1);
  for (int r = 0; r < cam.height; ++r) bg.row(r).setTo(r < cam.cy ? 210 - r / 3 : 105);  // sky, road
  for (int i = 0; i < 40; ++i) {                                                         // buildings
    const int x = rng.uniform(0, cam.width), w = rng.uniform(20, 90), h = rng.uniform(20, 120);
    if (std::abs(x - cam.cx) < 260) continue;                                            // keep lane clear
    cv::rectangle(bg, {x, int(cam.cy) - h}, {x + w, int(cam.cy)}, cv::Scalar(rng.uniform(60, 180)), cv::FILLED);
  }
  for (double y : {-1.8, 1.8})                                                           // lane dashes
    for (double x = 4.0; x < 80.0; x += 6.0) {
      const cv::Point2d a = project(cam, {x, y, -1.73}), b = project(cam, {x + 3.0, y, -1.73});
      cv::line(bg, a, b, cv::Scalar(235), std::max(1, int(12.0 / x)));
    }
  return bg;
}

// Render the car at distance d (lidar x of its rear face) with sub-pixel accurate scaling.
cv::Mat renderFrame(const Camera& cam, const cv::Mat& bg, const cv::Mat& tex, double d, cv::RNG& rng,
                    cv::Rect2d& carBox) {
  const cv::Point2d tl = project(cam, {d, 0.9, -0.03});    // top-left corner of the rear face
  const cv::Point2d br = project(cam, {d, -0.9, -1.43});   // bottom-right corner
  const double s = (br.x - tl.x) / tex.cols;               // texture px -> image px
  cv::Mat blurred;
  cv::GaussianBlur(tex, blurred, cv::Size(0, 0), 0.45 / s); // anti-aliasing before downscaling
  const cv::Mat M = (cv::Mat_<double>(2, 3) << s, 0, tl.x, 0, s, tl.y);
  cv::Mat frame = bg.clone(), f32, noise(bg.size(), CV_32F);
  cv::warpAffine(blurred, frame, M, frame.size(), cv::INTER_LINEAR, cv::BORDER_TRANSPARENT);
  rng.fill(noise, cv::RNG::NORMAL, 0.0, 2.0);               // sensor noise, sigma = 2 grey levels
  frame.convertTo(f32, CV_32F);
  f32 += noise;
  f32.convertTo(frame, CV_8U);
  carBox = cv::Rect2d(tl, br);
  return frame;
}

std::vector<LidarPoint> scanRearFace(double d, cv::RNG& rng, int nOutliers) {
  std::vector<LidarPoint> pts;
  for (double z = -0.10; z >= -1.40; z -= 0.18)             // beam rows ~0.18 m apart at this range
    for (double y = -0.85; y <= 0.851; y += 0.1) pts.push_back({d + rng.gaussian(0.02), y, z});
  for (int i = 0; i < nOutliers; ++i)                       // ghost returns: spray, exhaust, dust
    pts.push_back({d - rng.uniform(0.3, 1.0), rng.uniform(-0.8, 0.8), rng.uniform(-1.3, -0.2)});
  return pts;
}

// ------------------------------------------------------------------ keypoints and descriptors
// Harris: R = det(M) - k trace(M)^2, normalised to 0..255 inside the mask, NMS by max filter.
std::vector<cv::KeyPoint> detectHarris(const cv::Mat& img, const cv::Mat& mask, int blockSize = 2,
                                       int ksize = 3, double k = 0.04, float minResponse = 60.f,
                                       int nmsRadius = 3) {
  cv::Mat resp, dilated;
  cv::cornerHarris(img, resp, blockSize, ksize, k);
  double mn, mx;
  cv::minMaxLoc(resp, &mn, &mx, nullptr, nullptr, mask);
  resp = (resp - mn) * (255.0 / (mx - mn));
  cv::dilate(resp, dilated, cv::getStructuringElement(cv::MORPH_RECT, {2 * nmsRadius + 1, 2 * nmsRadius + 1}));
  std::vector<cv::KeyPoint> kps;
  for (int r = 0; r < resp.rows; ++r)
    for (int c = 0; c < resp.cols; ++c) {
      const float v = resp.at<float>(r, c);
      if (mask.at<uchar>(r, c) && v > minResponse && v >= dilated.at<float>(r, c))  // local maximum
        kps.emplace_back(cv::Point2f(float(c), float(r)), 2.f * ksize, -1.f, v);
    }
  return kps;
}

cv::Ptr<cv::Feature2D> makeDetector(const std::string& type) {  // nullptr for SHITOMASI / HARRIS
  if (type == "FAST") return cv::FastFeatureDetector::create(20, true, cv::FastFeatureDetector::TYPE_9_16);
  if (type == "BRISK") return cv::BRISK::create();
  if (type == "ORB") return cv::ORB::create(1000);
  if (type == "AKAZE") return cv::AKAZE::create();
  if (type == "SIFT") return cv::SIFT::create();
  return nullptr;
}

std::vector<cv::KeyPoint> detectKeypoints(const std::string& type, const cv::Ptr<cv::Feature2D>& det,
                                          const cv::Mat& img, const cv::Mat& mask) {
  std::vector<cv::KeyPoint> kps;
  if (type == "SHITOMASI") {
    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, 500, 0.01, 4.0, mask, 4, false, 0.04);
    for (const auto& c : corners) kps.emplace_back(c, 4.f);
  } else if (type == "HARRIS") {
    kps = detectHarris(img, mask);
  } else {
    det->detect(img, kps, mask);
  }
  return kps;
}

cv::Ptr<cv::Feature2D> makeExtractor(const std::string& type) {
  if (type == "BRISK") return cv::BRISK::create();
  if (type == "ORB") return cv::ORB::create();
  if (type == "AKAZE") return cv::AKAZE::create();
  if (type == "SIFT") return cv::SIFT::create();
#ifdef HAVE_OPENCV_XFEATURES2D
  if (type == "BRIEF") return cv::xfeatures2d::BriefDescriptorExtractor::create();
  if (type == "FREAK") return cv::xfeatures2d::FREAK::create();
#endif
  return nullptr;
}

bool validCombo(const std::string& det, const std::string& desc) {
  if (desc == "AKAZE" && det != "AKAZE") return false;  // AKAZE descriptors need AKAZE keypoints
  if (desc == "ORB" && det == "SIFT") return false;     // SIFT octave encoding breaks ORB
  return static_cast<bool>(makeExtractor(desc));
}

struct Frame {
  cv::Mat img;
  cv::Rect2d box;                    // "detector" box around the preceding car (here: exact)
  std::vector<LidarPoint> lidar;
  std::vector<cv::KeyPoint> kps;
  cv::Mat desc;
  double ms = 0;                     // detection + description time
};

// Detector and extractor objects are created once per combination (BRISK::create builds its
// sampling pattern, so re-creating it per frame would distort the timing).
void extractFeatures(std::vector<Frame>& frames, const std::string& det, const std::string& desc) {
  const cv::Ptr<cv::Feature2D> detector = makeDetector(det), extractor = makeExtractor(desc);
  for (Frame& f : frames) {
    cv::Mat mask = cv::Mat::zeros(f.img.size(), CV_8UC1);
    cv::rectangle(mask, f.box, cv::Scalar(255), cv::FILLED);
    const auto t0 = std::chrono::steady_clock::now();
    f.kps = detectKeypoints(det, detector, f.img, mask);
    extractor->compute(f.img, f.kps, f.desc);           // may drop keypoints near image borders
    f.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  }
}

cv::Rect2d shrink(const cv::Rect2d& r, double factor) {
  return {r.x + r.width * factor / 2, r.y + r.height * factor / 2, r.width * (1 - factor), r.height * (1 - factor)};
}

// kNN (k = 2) brute-force matching, Lowe ratio test, ROI gating, robust displacement filter.
// Convention: queryIdx -> previous frame, trainIdx -> current frame.
std::vector<cv::DMatch> matchFrames(const Frame& prev, const Frame& curr, const std::string& desc,
                                    double ratio = 0.8) {
  if (prev.desc.empty() || curr.desc.empty()) return {};
  const int norm = (desc == "SIFT") ? cv::NORM_L2 : cv::NORM_HAMMING;  // float vs binary descriptors
  cv::BFMatcher matcher(norm, false);
  std::vector<std::vector<cv::DMatch>> knn;
  matcher.knnMatch(prev.desc, curr.desc, knn, 2);
  const cv::Rect2d roiPrev = shrink(prev.box, 0.05), roiCurr = shrink(curr.box, 0.05);
  std::vector<cv::DMatch> good;
  std::vector<double> disp;
  for (const auto& m : knn) {
    if (m.size() < 2 || m[0].distance >= ratio * m[1].distance) continue;  // ambiguous match
    const cv::Point2f& p = prev.kps[m[0].queryIdx].pt;
    const cv::Point2f& c = curr.kps[m[0].trainIdx].pt;
    if (!roiPrev.contains(p) || !roiCurr.contains(c)) continue;
    good.push_back(m[0]);
    disp.push_back(cv::norm(c - p));
  }
  if (good.size() < 3) return good;
  std::vector<double> tmp = disp;                      // median absolute deviation filter
  std::nth_element(tmp.begin(), tmp.begin() + tmp.size() / 2, tmp.end());
  const double med = tmp[tmp.size() / 2];
  for (auto& v : tmp) v = std::fabs(v - med);
  std::nth_element(tmp.begin(), tmp.begin() + tmp.size() / 2, tmp.end());
  const double mad = tmp[tmp.size() / 2];
  std::vector<cv::DMatch> kept;
  for (size_t i = 0; i < good.size(); ++i)
    if (std::fabs(disp[i] - med) <= 3.0 * 1.4826 * mad + 1.0) kept.push_back(good[i]);
  return kept;
}

double median(std::vector<double> v) {
  if (v.empty()) return std::numeric_limits<double>::quiet_NaN();
  const size_t n = v.size() / 2;
  std::nth_element(v.begin(), v.begin() + n, v.end());
  if (v.size() % 2) return v[n];
  const double hi = v[n];
  return 0.5 * (hi + *std::max_element(v.begin(), v.begin() + n));
}

// Camera TTC from scale change: h1/h0 = d0/d1  =>  TTC = -dt / (1 - medianDistRatio)
double ttcCamera(const Frame& prev, const Frame& curr, const std::vector<cv::DMatch>& matches, double dt,
                 double minDist = 40.0) {
  std::vector<double> ratios;
  for (size_t i = 0; i < matches.size(); ++i)
    for (size_t j = i + 1; j < matches.size(); ++j) {
      const double dCurr = cv::norm(curr.kps[matches[i].trainIdx].pt - curr.kps[matches[j].trainIdx].pt);
      const double dPrev = cv::norm(prev.kps[matches[i].queryIdx].pt - prev.kps[matches[j].queryIdx].pt);
      if (dPrev > 1e-6 && dCurr >= minDist) ratios.push_back(dCurr / dPrev);
    }
  return -dt / (1.0 - median(ratios));
}

// Lidar points of the preceding car = points whose projection falls inside the shrunk box.
std::vector<LidarPoint> lidarInBox(const Camera& cam, const Frame& f) {
  const cv::Rect2d roi = shrink(f.box, 0.10);
  std::vector<LidarPoint> in;
  for (const auto& p : f.lidar)
    if (p.x > 0.5 && roi.contains(project(cam, p))) in.push_back(p);
  return in;
}

// Constant-velocity model: TTC = d1 dt / (d0 - d1). 'robust' uses the median x, else the minimum x.
double ttcLidar(const std::vector<LidarPoint>& prev, const std::vector<LidarPoint>& curr, double dt, bool robust) {
  auto closest = [robust](const std::vector<LidarPoint>& pts) {
    std::vector<double> xs;
    for (const auto& p : pts) xs.push_back(p.x);
    return robust ? median(xs) : *std::min_element(xs.begin(), xs.end());
  };
  const double d0 = closest(prev), d1 = closest(curr);
  return d1 * dt / (d0 - d1);
}

void saveDebugImage(const Camera& cam, const Frame& prev, const Frame& curr, const std::vector<cv::DMatch>& m,
                    const std::string& path) {
  const cv::Rect crop = (cv::Rect(prev.box) | cv::Rect(curr.box)) + cv::Size(60, 60) - cv::Point(30, 30);
  auto shift = [&](std::vector<cv::KeyPoint> k) { for (auto& p : k) p.pt -= cv::Point2f(crop.tl()); return k; };
  cv::Mat a, b, out, lid;
  cv::cvtColor(prev.img(crop), a, cv::COLOR_GRAY2BGR);
  cv::cvtColor(curr.img(crop), b, cv::COLOR_GRAY2BGR);
  cv::drawMatches(a, shift(prev.kps), b, shift(curr.kps), m, out, cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255));
  lid = b.clone();
  const auto pts = lidarInBox(cam, curr);
  const double xmin = pts.empty() ? 0 : std::min_element(pts.begin(), pts.end(), [](auto& p, auto& q) { return p.x < q.x; })->x;
  for (const auto& p : curr.lidar) {
    const cv::Point2d uv = project(cam, p) - cv::Point2d(crop.tl());
    const double t = std::clamp((p.x - xmin) / 1.2, 0.0, 1.0);  // red = closest, green = farther
    cv::circle(lid, uv, 2, cv::Scalar(0, 255 * t, 255 * (1 - t)), cv::FILLED);
  }
  cv::rectangle(lid, cv::Rect(shrink(curr.box, 0.10)) - crop.tl(), cv::Scalar(255, 200, 0), 1);
  cv::Mat pad = cv::Mat::zeros(lid.rows, out.cols - lid.cols, CV_8UC3);
  cv::hconcat(lid, pad, lid);
  cv::vconcat(out, lid, out);
  cv::resize(out, out, cv::Size(), 2.0, 2.0, cv::INTER_NEAREST);
  cv::imwrite(path, out);
}

int main(int argc, char** argv) {
  const Camera cam;
  cv::RNG rng(2026);
  const cv::Mat tex = makeCarTexture(rng), bg = makeBackground(cam, rng);
  const double v = 5.0, dt = 0.1, d0 = 15.0;                // closing at 5 m/s, 10 Hz
  const int nFrames = 10;
  std::vector<Frame> frames(nFrames);
  for (int k = 0; k < nFrames; ++k) {
    const double d = d0 - v * dt * k;
    frames[k].img = renderFrame(cam, bg, tex, d, rng, frames[k].box);
    frames[k].lidar = scanRearFace(d, rng, 2);
  }
  auto truth = [&](int k) { return (d0 - v * dt * k) / v; };  // TTC = d / v

  // 1. Detailed run: one detector/descriptor pair, camera vs lidar TTC per frame.
  const std::string det = argc > 2 ? argv[1] : "AKAZE", desc = argc > 2 ? argv[2] : "AKAZE";
  if (!validCombo(det, desc)) { std::printf("invalid combination %s/%s\n", det.c_str(), desc.c_str()); return 1; }
  std::printf("Detailed run %s/%s  (closing speed %.1f m/s, frame rate %.0f Hz)\n", det.c_str(), desc.c_str(), v, 1 / dt);
  std::printf("%-5s %6s %9s %11s %14s %10s %5s %7s\n", "frame", "d[m]", "TTC true", "lidar(min)", "lidar(median)",
              "camera", "kpts", "matches");
  extractFeatures(frames, det, desc);
  std::vector<cv::DMatch> lastMatches;
  for (int k = 1; k < nFrames; ++k) {
    const auto m = matchFrames(frames[k - 1], frames[k], desc);
    const auto lp = lidarInBox(cam, frames[k - 1]), lc = lidarInBox(cam, frames[k]);
    std::printf("%-5d %6.2f %8.2fs %10.2fs %13.2fs %9.2fs %5zu %7zu\n", k, d0 - v * dt * k, truth(k),
                ttcLidar(lp, lc, dt, false), ttcLidar(lp, lc, dt, true), ttcCamera(frames[k - 1], frames[k], m, dt),
                frames[k].kps.size(), m.size());
    lastMatches = m;
  }
  saveDebugImage(cam, frames[nFrames - 2], frames[nFrames - 1], lastMatches, "camera_ttc_matches.png");
  std::printf("wrote camera_ttc_matches.png (matches on top, lidar points projected below)\n\n");

  // 2. Benchmark: every valid detector/descriptor pair over the sequence.
  const std::vector<std::string> dets = {"SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "AKAZE", "SIFT"};
  const std::vector<std::string> descs = {"BRISK", "ORB", "AKAZE", "SIFT", "BRIEF", "FREAK"};
  std::printf("%-10s %-6s %8s %8s %12s %12s %10s\n", "detector", "desc", "kpts/fr", "match/fr", "|TTC err| s",
              "max err s", "ms/frame");
  for (const auto& dn : dets)
    for (const auto& ds : descs) {
      if (!validCombo(dn, ds)) continue;
      std::vector<Frame> fr = frames;
      double kp = 0, mt = 0, err = 0, maxErr = 0, ms = 0;
      extractFeatures(fr, dn, ds);
      for (const auto& f : fr) { kp += f.kps.size(); ms += f.ms; }
      for (int k = 1; k < nFrames; ++k) {
        const auto m = matchFrames(fr[k - 1], fr[k], ds);
        const double e = std::fabs(ttcCamera(fr[k - 1], fr[k], m, dt) - truth(k));
        mt += m.size();
        err += std::isnan(e) ? 99.0 : e;
        maxErr = std::max(maxErr, std::isnan(e) ? 99.0 : e);
      }
      std::printf("%-10s %-6s %8.0f %8.0f %12.3f %12.3f %10.1f\n", dn.c_str(), ds.c_str(), kp / nFrames,
                  mt / (nFrames - 1), err / (nFrames - 1), maxErr, ms / nFrames);
    }
  return 0;
}

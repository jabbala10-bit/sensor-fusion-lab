// fusion_filters.cpp - Kalman, extended Kalman and unscented Kalman filters for lidar + radar
// fusion (C++17, Eigen). Generates a CTRV ground-truth trajectory, simulates alternating lidar and
// radar measurements, runs three filters and reports RMSE and NIS consistency.
// Build: g++ -std=c++17 -O2 -I/usr/include/eigen3 fusion_filters.cpp -o fusion_filters
#include <Eigen/Dense>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using Eigen::MatrixXd;
using Eigen::VectorXd;

constexpr double kPi = 3.14159265358979323846;

double normAngle(double a) {
  while (a > kPi) a -= 2 * kPi;
  while (a < -kPi) a += 2 * kPi;
  return a;
}

struct TruthPt { double t, px, py, vx, vy, v, psi; };
struct Meas { double t; bool radar; VectorXd z; };

// ------------------------------------------------------------------ ground truth (trajectory generation)
// CTRV with a time-varying yaw rate and longitudinal acceleration, integrated in small steps.
std::vector<TruthPt> generateTrajectory(double duration, double dt) {
  double px = -5, py = -15, v = 5, psi = 0;
  std::vector<TruthPt> out;
  for (double t = 0; t <= duration + 1e-9; t += dt) {
    out.push_back({t, px, py, v * std::cos(psi), v * std::sin(psi), v, psi});
    const double acc = 0.6 * std::sin(0.5 * t);          // m/s^2
    const double yawRate = 0.25 + 0.15 * std::sin(0.4 * t);  // rad/s
    if (std::fabs(yawRate) > 1e-6) {
      px += v / yawRate * (std::sin(psi + yawRate * dt) - std::sin(psi));
      py += v / yawRate * (-std::cos(psi + yawRate * dt) + std::cos(psi));
    } else {
      px += v * std::cos(psi) * dt;
      py += v * std::sin(psi) * dt;
    }
    psi = normAngle(psi + yawRate * dt);
    v += acc * dt;
  }
  return out;
}

// ------------------------------------------------------------------ constant-velocity KF / EKF
class KalmanCV {
 public:
  VectorXd x = VectorXd::Zero(4);                        // px, py, vx, vy
  MatrixXd P = MatrixXd::Identity(4, 4);
  double sigmaA2 = 9.0;                                  // acceleration noise [m^2/s^4]

  void predict(double dt) {
    MatrixXd F = MatrixXd::Identity(4, 4);
    F(0, 2) = dt;
    F(1, 3) = dt;
    const double dt2 = dt * dt, dt3 = dt2 * dt / 2, dt4 = dt2 * dt2 / 4;
    MatrixXd Q(4, 4);
    Q << dt4, 0, dt3, 0,                                  // Q = G diag(sigma_a^2) G^T
         0, dt4, 0, dt3,
         dt3, 0, dt2, 0,
         0, dt3, 0, dt2;
    Q *= sigmaA2;
    x = F * x;
    P = F * P * F.transpose() + Q;
  }

  double updateLidar(const VectorXd& z, const MatrixXd& R) {
    MatrixXd H(2, 4);
    H << 1, 0, 0, 0,
         0, 1, 0, 0;
    const VectorXd y = z - H * x;
    const MatrixXd S = H * P * H.transpose() + R;
    const MatrixXd K = P * H.transpose() * S.inverse();
    x += K * y;
    P = (MatrixXd::Identity(4, 4) - K * H) * P;
    return y.transpose() * S.inverse() * y;               // NIS
  }

  double updateRadar(const VectorXd& z, const MatrixXd& R) {   // EKF: linearise h at the prediction
    const double px = x(0), py = x(1), vx = x(2), vy = x(3);
    const double c1 = std::max(1e-6, px * px + py * py), c2 = std::sqrt(c1), c3 = c1 * c2;
    VectorXd h(3);
    h << c2, std::atan2(py, px), (px * vx + py * vy) / c2;
    MatrixXd Hj(3, 4);                                     // Jacobian of h
    Hj << px / c2, py / c2, 0, 0,
          -py / c1, px / c1, 0, 0,
          py * (vx * py - vy * px) / c3, px * (vy * px - vx * py) / c3, px / c2, py / c2;
    VectorXd y = z - h;
    y(1) = normAngle(y(1));                                // angles must wrap, or the filter diverges
    const MatrixXd S = Hj * P * Hj.transpose() + R;
    const MatrixXd K = P * Hj.transpose() * S.inverse();
    x += K * y;
    P = (MatrixXd::Identity(4, 4) - K * Hj) * P;
    return y.transpose() * S.inverse() * y;
  }
};

// ------------------------------------------------------------------ unscented KF with CTRV model
class UKF {
 public:
  static constexpr int n = 5, nAug = 7, nSig = 2 * nAug + 1;   // px, py, v, psi, psi_dot (+ 2 noises)
  VectorXd x = VectorXd::Zero(n);
  MatrixXd P = MatrixXd::Identity(n, n);
  double stdA = 1.0, stdYawdd = 0.5;
  double lambda = 3.0 - nAug;
  VectorXd weights = VectorXd(nSig);
  MatrixXd Xsig = MatrixXd::Zero(n, nSig);

  UKF() {
    weights(0) = lambda / (lambda + nAug);
    for (int i = 1; i < nSig; ++i) weights(i) = 0.5 / (lambda + nAug);
  }

  void predict(double dt) {
    VectorXd xa = VectorXd::Zero(nAug);                    // augment the state with the process noise
    xa.head(n) = x;
    MatrixXd Pa = MatrixXd::Zero(nAug, nAug);
    Pa.topLeftCorner(n, n) = P;
    Pa(5, 5) = stdA * stdA;
    Pa(6, 6) = stdYawdd * stdYawdd;
    const MatrixXd L = Pa.llt().matrixL();                 // Cholesky = matrix square root
    MatrixXd Xa(nAug, nSig);
    Xa.col(0) = xa;
    for (int i = 0; i < nAug; ++i) {
      Xa.col(i + 1) = xa + std::sqrt(lambda + nAug) * L.col(i);
      Xa.col(i + 1 + nAug) = xa - std::sqrt(lambda + nAug) * L.col(i);
    }
    for (int i = 0; i < nSig; ++i) {                        // push each sigma point through CTRV
      const double px = Xa(0, i), py = Xa(1, i), v = Xa(2, i), psi = Xa(3, i), psid = Xa(4, i);
      const double nuA = Xa(5, i), nuPsi = Xa(6, i);
      double ppx, ppy;
      if (std::fabs(psid) > 1e-4) {
        ppx = px + v / psid * (std::sin(psi + psid * dt) - std::sin(psi));
        ppy = py + v / psid * (-std::cos(psi + psid * dt) + std::cos(psi));
      } else {                                              // straight-line limit
        ppx = px + v * std::cos(psi) * dt;
        ppy = py + v * std::sin(psi) * dt;
      }
      Xsig(0, i) = ppx + 0.5 * dt * dt * std::cos(psi) * nuA;
      Xsig(1, i) = ppy + 0.5 * dt * dt * std::sin(psi) * nuA;
      Xsig(2, i) = v + dt * nuA;
      Xsig(3, i) = psi + psid * dt + 0.5 * dt * dt * nuPsi;
      Xsig(4, i) = psid + dt * nuPsi;
    }
    x = Xsig * weights;
    x(3) = normAngle(x(3));
    P.setZero();
    for (int i = 0; i < nSig; ++i) {
      VectorXd d = Xsig.col(i) - x;
      d(3) = normAngle(d(3));
      P += weights(i) * d * d.transpose();
    }
  }

  // One update for any measurement function h; angleIdx names the component that wraps (-1: none).
  template <typename H>
  double update(const VectorXd& z, const MatrixXd& R, H h, int angleIdx) {
    const int nz = int(z.size());
    MatrixXd Zsig(nz, nSig);
    for (int i = 0; i < nSig; ++i) Zsig.col(i) = h(VectorXd(Xsig.col(i)));
    VectorXd zPred = Zsig * weights;
    MatrixXd S = R, T = MatrixXd::Zero(n, nz);
    for (int i = 0; i < nSig; ++i) {
      VectorXd dz = Zsig.col(i) - zPred;
      if (angleIdx >= 0) dz(angleIdx) = normAngle(dz(angleIdx));
      VectorXd dx = Xsig.col(i) - x;
      dx(3) = normAngle(dx(3));
      S += weights(i) * dz * dz.transpose();
      T += weights(i) * dx * dz.transpose();               // cross-correlation state / measurement
    }
    const MatrixXd K = T * S.inverse();
    VectorXd y = z - zPred;
    if (angleIdx >= 0) y(angleIdx) = normAngle(y(angleIdx));
    x += K * y;
    x(3) = normAngle(x(3));
    P -= K * S * K.transpose();
    return y.transpose() * S.inverse() * y;
  }
};

int main() {
  const double dtTruth = 0.002, duration = 25.0, dtMeas = 0.05;   // sensors alternate at 20 Hz
  const auto truth = generateTrajectory(duration, dtTruth);
  const double sLidar = 0.15, sRho = 0.3, sPhi = 0.03, sRhoDot = 0.3;
  MatrixXd Rlidar = MatrixXd::Identity(2, 2) * sLidar * sLidar;
  MatrixXd Rradar = MatrixXd::Zero(3, 3);
  Rradar.diagonal() << sRho * sRho, sPhi * sPhi, sRhoDot * sRhoDot;

  std::mt19937 rng(21);
  std::normal_distribution<double> g(0.0, 1.0);
  std::vector<Meas> meas;
  std::vector<TruthPt> at;
  double minRange = 1e9, maxRange = 0;
  for (int k = 0; k * dtMeas <= duration; ++k) {
    const TruthPt& s = truth[size_t(k * dtMeas / dtTruth)];
    const double rho = std::hypot(s.px, s.py);
    minRange = std::min(minRange, rho);
    maxRange = std::max(maxRange, rho);
    VectorXd z;
    if (k % 2 == 0) {                                     // lidar: cartesian position
      z = VectorXd(2);
      z << s.px + sLidar * g(rng), s.py + sLidar * g(rng);
    } else {                                              // radar: range, bearing, range rate
      z = VectorXd(3);
      z << rho + sRho * g(rng), std::atan2(s.py, s.px) + sPhi * g(rng),
          (s.px * s.vx + s.py * s.vy) / rho + sRhoDot * g(rng);
    }
    meas.push_back({k * dtMeas, k % 2 == 1, z});
    at.push_back(s);
  }
  std::printf("Trajectory: %.0f s, %zu measurements (lidar and radar alternating at 10 Hz each)\n",
              duration, meas.size());
  std::printf("  range from sensor %.1f to %.1f m, speed %.1f to %.1f m/s, yaw rate 0.10 to 0.40 rad/s\n\n",
              minRange, maxRange, 3.5, 6.5);

  KalmanCV kf, ekf;                                        // kf: lidar only; ekf: lidar + radar
  UKF ukf;
  bool init = false;
  double sum[3][4] = {};
  int nRmse = 0, nLidar = 0, nRadar = 0, overLidar = 0, overRadar = 0;
  double tPrev = 0;
  for (size_t i = 0; i < meas.size(); ++i) {
    const Meas& m = meas[i];
    if (!init) {                                           // first measurement is lidar
      kf.x << m.z(0), m.z(1), 0, 0;
      kf.P.diagonal() << sLidar * sLidar, sLidar * sLidar, 100, 100;
      ekf.x = kf.x;
      ekf.P = kf.P;
      ukf.x << m.z(0), m.z(1), 0, 0, 0;
      ukf.P.diagonal() << sLidar * sLidar, sLidar * sLidar, 25, 1, 1;
      init = true;
      tPrev = m.t;
      continue;
    }
    const double dt = m.t - tPrev;
    tPrev = m.t;
    kf.predict(dt);        // every filter predicts to the measurement time; the lidar-only
    ekf.predict(dt);       // filter simply has no update to apply at radar epochs
    ukf.predict(dt);
    if (m.radar) {
      ekf.updateRadar(m.z, Rradar);
      const double nis = ukf.update(m.z, Rradar,
                                    [](const VectorXd& s) {
                                      const double rho = std::max(1e-6, std::hypot(s(0), s(1)));
                                      VectorXd zz(3);
                                      zz << rho, std::atan2(s(1), s(0)),
                                          (s(0) * std::cos(s(3)) * s(2) + s(1) * std::sin(s(3)) * s(2)) / rho;
                                      return zz;
                                    },
                                    1);
      overRadar += nis > 7.815;                            // NIS consistency of the UKF
      ++nRadar;
    } else {
      kf.updateLidar(m.z, Rlidar);
      ekf.updateLidar(m.z, Rlidar);
      const double nis = ukf.update(m.z, Rlidar,
                                    [](const VectorXd& s) {
                                      VectorXd zz(2);
                                      zz << s(0), s(1);
                                      return zz;
                                    },
                                    -1);
      overLidar += nis > 5.991;
      ++nLidar;
    }
    const TruthPt& s = at[i];
    const double est[3][4] = {{kf.x(0), kf.x(1), kf.x(2), kf.x(3)},
                              {ekf.x(0), ekf.x(1), ekf.x(2), ekf.x(3)},
                              {ukf.x(0), ukf.x(1), ukf.x(2) * std::cos(ukf.x(3)), ukf.x(2) * std::sin(ukf.x(3))}};
    const double tru[4] = {s.px, s.py, s.vx, s.vy};
    for (int f = 0; f < 3; ++f)
      for (int c = 0; c < 4; ++c) sum[f][c] += (est[f][c] - tru[c]) * (est[f][c] - tru[c]);
    ++nRmse;
  }

  const char* names[3] = {"KF  (lidar only, CV) ", "EKF (lidar+radar, CV)", "UKF (lidar+radar, CTRV)"};
  std::printf("%-24s %8s %8s %8s %8s\n", "filter", "RMSE px", "py", "vx", "vy");
  for (int f = 0; f < 3; ++f)
    std::printf("%-24s %8.3f %8.3f %8.3f %8.3f\n", names[f], std::sqrt(sum[f][0] / nRmse),
                std::sqrt(sum[f][1] / nRmse), std::sqrt(sum[f][2] / nRmse), std::sqrt(sum[f][3] / nRmse));
  std::printf("\nUKF consistency: lidar NIS above 5.991 in %.1f%% of %d updates, "
              "radar NIS above 7.815 in %.1f%% of %d updates (5%% expected)\n",
              100.0 * overLidar / nLidar, nLidar, 100.0 * overRadar / nRadar, nRadar);
  return 0;
}

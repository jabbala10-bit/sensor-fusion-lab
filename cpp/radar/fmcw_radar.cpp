// fmcw_radar.cpp - FMCW radar signal chain from scratch (C++17, no dependencies).
// Requirements -> waveform design -> beat signal of several targets on an 8-element receive array
// -> own radix-2 FFT -> range FFT -> Doppler FFT (range-Doppler map) -> 2D CA-CFAR
// -> clustering of detected cells -> angle FFT across antennas -> detection list vs ground truth.
// Build: g++ -std=c++17 -O2 fmcw_radar.cpp -o fmcw_radar
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <random>
#include <vector>

using cd = std::complex<double>;
constexpr double kC = 3e8;
constexpr double kPi = 3.14159265358979323846;

// ------------------------------------------------------------------ radix-2 FFT, in place
void fft(std::vector<cd>& a) {
  const size_t n = a.size();                     // must be a power of two
  for (size_t i = 1, j = 0; i < n; ++i) {        // bit-reversal permutation
    size_t bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) std::swap(a[i], a[j]);
  }
  for (size_t len = 2; len <= n; len <<= 1) {    // log2(n) stages of butterflies
    const cd wl = std::polar(1.0, -2.0 * kPi / double(len));
    for (size_t i = 0; i < n; i += len) {
      cd w(1.0);
      for (size_t k = 0; k < len / 2; ++k) {
        const cd u = a[i + k], v = a[i + k + len / 2] * w;
        a[i + k] = u + v;
        a[i + k + len / 2] = u - v;
        w *= wl;
      }
    }
  }
}

double hann(int i, int n) { return 0.5 - 0.5 * std::cos(2.0 * kPi * i / (n - 1)); }

struct Target { const char* name; double range, vel, angleDeg, rcsDbsm; };  // vel > 0: moving away

int main() {
  // ---------------------------------------------------------------- 1. waveform design
  const double fc = 77e9, lambda = kC / fc;
  const double rangeRes = 1.0, maxRange = 200.0;
  const double B = kC / (2.0 * rangeRes);                  // range resolution = c / 2B
  const double Tc = 5.5 * 2.0 * maxRange / kC;             // chirp time: 5.5 x max round trip
  const double S = B / Tc;                                 // chirp slope [Hz/s]
  const int Nr = 256, Nd = 128, Na = 8;                    // samples/chirp, chirps/frame, antennas
  const double Fs = Nr / Tc;                               // complex (IQ) ADC rate
  const double vRes = lambda / (2.0 * Nd * Tc), vMax = lambda / (4.0 * Tc);
  const double dAnt = lambda / 2.0;
  std::printf("Waveform design (77 GHz, lambda %.2f mm)\n", lambda * 1e3);
  std::printf("  range res %.1f m      -> bandwidth B = %.1f MHz\n", rangeRes, B / 1e6);
  std::printf("  max range %.0f m      -> chirp Tc = %.2f us (5.5 x round trip), slope %.2f MHz/us\n",
              maxRange, Tc * 1e6, S / 1e12);
  std::printf("  Nr = %d IQ samples  -> Fs = %.1f MHz, ADC-limited max range %.0f m\n", Nr, Fs / 1e6,
              Fs * kC / (2 * S));
  std::printf("  Nd = %d chirps      -> velocity res %.2f m/s, unambiguous +-%.1f m/s\n", Nd, vRes, vMax);
  std::printf("  Na = %d rx at l/2    -> angle res ~%.1f deg at boresight\n\n", Na, 2.0 / Na * 180 / kPi);

  // ---------------------------------------------------------------- 2. beat signal (IQ mixer output)
  const std::vector<Target> targets = {{"car ahead", 110, -20, 2, 10},
                                       {"car, receding", 60, 8, -15, 10},
                                       {"pedestrian", 25, -1.5, 20, -5},
                                       {"truck", 160, -5, -5, 20}};
  std::mt19937 rng(7);
  std::normal_distribution<double> gauss(0.0, 1.0 / std::sqrt(2.0));  // unit-power complex noise
  // cube[a][chirp][sample]; amplitude ~ sqrt(RCS) / R^2 (radar equation), scaled so that the
  // car at 110 m has a per-sample SNR of -20 dB before any FFT gain.
  const double ampRef = std::pow(10.0, -20.0 / 20.0) * 110.0 * 110.0 / std::pow(10.0, 10.0 / 20.0);
  std::vector<cd> cube(size_t(Na) * Nd * Nr);
  for (int a = 0; a < Na; ++a)
    for (int m = 0; m < Nd; ++m)
      for (int n = 0; n < Nr; ++n) {
        const double tFast = n / Fs, tAbs = m * Tc + tFast;
        cd s(gauss(rng), gauss(rng));
        for (const Target& t : targets) {
          const double tau = 2.0 * (t.range + t.vel * tAbs) / kC;          // round-trip delay
          const double phase = 2 * kPi * (fc * tau + S * tFast * tau - 0.5 * S * tau * tau)  // Tx - Rx
                               + kPi * a * std::sin(t.angleDeg * kPi / 180);  // array phase, d = l/2
          const double amp = ampRef * std::pow(10.0, t.rcsDbsm / 20.0) / (t.range * t.range);
          s += std::polar(amp, phase);
        }
        cube[(size_t(a) * Nd + m) * Nr + n] = s;
      }

  // ---------------------------------------------------------------- 3. range FFT, then Doppler FFT
  std::vector<cd> buf;
  for (int a = 0; a < Na; ++a) {
    for (int m = 0; m < Nd; ++m) {                           // fast time -> range
      buf.assign(Nr, 0);
      for (int n = 0; n < Nr; ++n) buf[n] = cube[(size_t(a) * Nd + m) * Nr + n] * hann(n, Nr);
      fft(buf);
      for (int n = 0; n < Nr; ++n) cube[(size_t(a) * Nd + m) * Nr + n] = buf[n];
    }
    for (int r = 0; r < Nr; ++r) {                           // slow time -> Doppler (fftshifted)
      buf.assign(Nd, 0);
      for (int m = 0; m < Nd; ++m) buf[m] = cube[(size_t(a) * Nd + m) * Nr + r] * hann(m, Nd);
      fft(buf);
      for (int m = 0; m < Nd; ++m) cube[(size_t(a) * Nd + m) * Nr + r] = buf[(m + Nd / 2) % Nd];
    }
  }
  // Range-Doppler map (power) from antenna 0: rdm[d][r]; row d = Doppler bin, velocity (d - Nd/2) vRes
  std::vector<double> rdm(size_t(Nd) * Nr);
  for (int d = 0; d < Nd; ++d)
    for (int r = 0; r < Nr; ++r) rdm[size_t(d) * Nr + r] = std::norm(cube[size_t(d) * Nr + r]);

  // ---------------------------------------------------------------- 4. 2D CA-CFAR
  const int Tr = 8, Td = 4, Gr = 2, Gd = 2;                   // training and guard cells
  const int nTrain = (2 * (Tr + Gr) + 1) * (2 * (Td + Gd) + 1) - (2 * Gr + 1) * (2 * Gd + 1);
  const double pfa = 1e-6;
  const double alpha = nTrain * (std::pow(pfa, -1.0 / nTrain) - 1.0);  // square-law, exponential noise
  std::printf("2D CA-CFAR: %d training cells, Pfa %.0e -> threshold factor %.1f (%.1f dB above local mean)\n",
              nTrain, pfa, alpha, 10 * std::log10(alpha));
  std::vector<char> det(rdm.size(), 0);
  int nDet = 0, nTested = 0;
  for (int d = Td + Gd; d < Nd - Td - Gd; ++d)               // edges are not tested (no full window)
    for (int r = Tr + Gr; r < Nr - Tr - Gr; ++r) {
      double noise = 0;
      for (int i = -(Td + Gd); i <= Td + Gd; ++i)
        for (int j = -(Tr + Gr); j <= Tr + Gr; ++j)
          if (std::abs(i) > Gd || std::abs(j) > Gr) noise += rdm[size_t(d + i) * Nr + (r + j)];  // linear power
      ++nTested;
      if (rdm[size_t(d) * Nr + r] > alpha * noise / nTrain) { det[size_t(d) * Nr + r] = 1; ++nDet; }
    }
  std::vector<double> tmp(rdm.begin(), rdm.end());           // noise floor for SNR reporting
  std::nth_element(tmp.begin(), tmp.begin() + tmp.size() / 2, tmp.end());
  const double noiseMean = tmp[tmp.size() / 2] / std::log(2.0);  // median of exponential = mean ln 2

  // ---------------------------------------------------------------- 5. cluster cells, estimate r, v, angle
  std::vector<char> seen(det.size(), 0);
  std::printf("%d of %d tested cells above threshold; expected noise false alarms per frame %.3f\n\n",
              nDet, nTested, nTested * pfa);
  std::printf("%-3s %9s %13s %10s %8s   %s\n", "id", "range m", "velocity m/s", "angle deg", "SNR dB",
              "matched truth (range, velocity, angle)");
  int id = 0;
  for (int d0 = 0; d0 < Nd; ++d0)
    for (int r0 = 0; r0 < Nr; ++r0) {
      if (!det[size_t(d0) * Nr + r0] || seen[size_t(d0) * Nr + r0]) continue;
      std::vector<std::pair<int, int>> stack{{d0, r0}}, cells;  // 8-connected component
      seen[size_t(d0) * Nr + r0] = 1;
      while (!stack.empty()) {
        const auto [d, r] = stack.back();
        stack.pop_back();
        cells.push_back({d, r});
        for (int i = -1; i <= 1; ++i)
          for (int j = -1; j <= 1; ++j) {
            const int dd = d + i, rr = r + j;
            if (dd < 0 || dd >= Nd || rr < 0 || rr >= Nr) continue;
            const size_t k = size_t(dd) * Nr + rr;
            if (det[k] && !seen[k]) { seen[k] = 1; stack.push_back({dd, rr}); }
          }
      }
      double w = 0, rc = 0, dc = 0, peak = 0;                  // power-weighted centroid
      int pd = d0, pr = r0;
      for (const auto& [d, r] : cells) {
        const double p = rdm[size_t(d) * Nr + r];
        w += p; rc += p * r; dc += p * d;
        if (p > peak) { peak = p; pd = d; pr = r; }
      }
      const double range = (rc / w) * kC / (2.0 * B), vel = (dc / w - Nd / 2) * vRes;
      // angle FFT across the antennas at the peak cell, zero-padded to 64 bins
      const int Nf = 64;
      std::vector<cd> sp(Nf, 0);
      for (int a = 0; a < Na; ++a) sp[a] = cube[(size_t(a) * Nd + pd) * Nr + pr];
      fft(sp);
      int kBest = 0;
      for (int k = 1; k < Nf; ++k) if (std::abs(sp[k]) > std::abs(sp[kBest])) kBest = k;
      const int kSigned = kBest < Nf / 2 ? kBest : kBest - Nf;
      const double angle = std::asin(std::clamp(lambda * kSigned / (dAnt * Nf), -1.0, 1.0)) * 180 / kPi;
      const Target* best = nullptr;                              // truth within 3 m and 3 m/s
      for (const Target& t : targets)
        if (std::fabs(t.range - range) < 3.0 && std::fabs(t.vel - vel) < 3.0) best = &t;
      std::printf("%-3d %9.2f %13.2f %10.1f %8.1f   ", id++, range, vel, angle, 10 * std::log10(peak / noiseMean));
      if (best) std::printf("%-14s (%.0f m, %+.1f m/s, %+.0f deg)\n", best->name, best->range, best->vel, best->angleDeg);
      else std::printf("false alarm (noise)\n");
    }
  return 0;
}

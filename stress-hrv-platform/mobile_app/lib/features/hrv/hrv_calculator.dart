import 'dart:math';

class HrvMetrics {
  final double rmssd;
  final double sdnn;
  final double pnn50;
  final double meanHr;

  const HrvMetrics({
    required this.rmssd,
    required this.sdnn,
    required this.pnn50,
    required this.meanHr,
  });

  Map<String, dynamic> toJson() => {
        'rmssd': rmssd,
        'sdnn': sdnn,
        'pnn50': pnn50,
        'mean_hr': meanHr,
      };
}

class HrvCalculator {
  static HrvMetrics fromSignal(List<double> signal, double samplingHz) {
    final minSamples = max(30, (samplingHz * 8).round());
    if (signal.length < minSamples) {
      return const HrvMetrics(rmssd: 0, sdnn: 0, pnn50: 0, meanHr: 0);
    }

    final detrended = _detrend(signal);
    final peaks = _detectPeaks(detrended, samplingHz);
    if (peaks.length < 3) {
      return const HrvMetrics(rmssd: 0, sdnn: 0, pnn50: 0, meanHr: 0);
    }

    final rr = <double>[];
    for (int i = 1; i < peaks.length; i++) {
      rr.add((peaks[i] - peaks[i - 1]) * 1000 / samplingHz);
    }

    if (rr.isEmpty) {
      return const HrvMetrics(rmssd: 0, sdnn: 0, pnn50: 0, meanHr: 0);
    }

    final meanRr = rr.reduce((a, b) => a + b) / rr.length;
    final meanHr = meanRr > 0 ? 60000.0 / meanRr : 0.0;

    final rrDiff = <double>[];
    for (int i = 1; i < rr.length; i++) {
      rrDiff.add((rr[i] - rr[i - 1]).abs());
    }

    final rmssd = rrDiff.isEmpty
      ? 0.0
      : sqrt(rrDiff.map((d) => d * d).reduce((a, b) => a + b) / rrDiff.length);

    final variance = rr.map((v) => pow(v - meanRr, 2)).reduce((a, b) => a + b) / rr.length;
    final sdnn = sqrt(variance);

    final nn50Count = rrDiff.where((d) => d > 50).length;
    final pnn50 = rrDiff.isEmpty ? 0.0 : (nn50Count / rrDiff.length) * 100.0;

    return HrvMetrics(rmssd: rmssd, sdnn: sdnn, pnn50: pnn50, meanHr: meanHr);
  }

  static List<double> _detrend(List<double> signal) {
    final mean = signal.reduce((a, b) => a + b) / signal.length;
    return signal.map((v) => v - mean).toList(growable: false);
  }

  static List<int> _detectPeaks(List<double> signal, double samplingHz) {
    final peaks = <int>[];
    final minDistance = max(1, (samplingHz * 0.4).floor());
    int lastPeak = -minDistance;

    for (int i = 1; i < signal.length - 1; i++) {
      final isPeak = signal[i] > signal[i - 1] && signal[i] > signal[i + 1] && signal[i] > 0;
      if (isPeak && (i - lastPeak) >= minDistance) {
        peaks.add(i);
        lastPeak = i;
      }
    }

    return peaks;
  }
}

import '../hrv/hrv_calculator.dart';

class StressInference {
  final double probability;
  final bool isHighStress;

  const StressInference({required this.probability, required this.isHighStress});
}

class StressEngine {
  final double threshold;

  const StressEngine({this.threshold = 0.7});

  StressInference infer(HrvMetrics metrics) {
    if (metrics.meanHr == 0) {
      return const StressInference(probability: 0, isHighStress: false);
    }

    final hrvScore = (metrics.rmssd + metrics.sdnn + metrics.pnn50) / 3;
    final normalizedHrv = (hrvScore / 80.0).clamp(0.0, 1.0).toDouble();
    final normalizedHr = (metrics.meanHr / 120.0).clamp(0.0, 1.0).toDouble();

    final probability = (0.65 * (1.0 - normalizedHrv) + 0.35 * normalizedHr).clamp(0.0, 1.0).toDouble();

    return StressInference(
      probability: probability,
      isHighStress: probability >= threshold,
    );
  }
}

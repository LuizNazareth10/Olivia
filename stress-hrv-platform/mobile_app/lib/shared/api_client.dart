import 'dart:convert';

import 'package:http/http.dart' as http;

import '../features/hrv/hrv_calculator.dart';

class ApiClient {
  final String edgeBaseUrl;

  ApiClient({required this.edgeBaseUrl});

  Future<Map<String, dynamic>> predictStress({
    required HrvMetrics metrics,
    required List<double> signal,
    required double samplingHz,
    required String userId,
    required String deviceId,
  }) async {
    final response = await http.post(
      Uri.parse('$edgeBaseUrl/analyze'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'user_id': userId,
        'device_id': deviceId,
        'sampling_hz': samplingHz,
        'signal': signal,
        'metrics': metrics.toJson(),
      }),
    );

    if (response.statusCode >= 200 && response.statusCode < 300) {
      return jsonDecode(response.body) as Map<String, dynamic>;
    }

    throw Exception('Erro ao analisar HRV no edge: ${response.body}');
  }
}

import 'dart:async';

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:image/image.dart' as img;

class CaptureResult {
  final List<double> signal;
  final double samplingHz;

  const CaptureResult({required this.signal, required this.samplingHz});
}

class CaptureController {
  final CameraController cameraController;
  final Duration captureDuration;

  CaptureController({
    required this.cameraController,
    this.captureDuration = const Duration(seconds: 30),
  });

  bool _isCapturing = false;

  Future<CaptureResult> startCapture() async {
    if (_isCapturing) {
      throw Exception('Captura já em andamento');
    }

    if (kIsWeb) {
      return _captureWithSnapshots();
    }

    try {
      return await _captureWithImageStream();
    } on UnimplementedError {
      return _captureWithSnapshots();
    } on CameraException catch (e) {
      final message = (e.description ?? '').toLowerCase();
      final code = e.code.toLowerCase();
      if (message.contains('not implemented') ||
          message.contains('streamedframeavailable') ||
          code.contains('unimplemented')) {
        return _captureWithSnapshots();
      }
      rethrow;
    }
  }

  Future<CaptureResult> _captureWithImageStream() async {
    _isCapturing = true;
    final signal = <double>[];
    final timestamps = <DateTime>[];
    final completer = Completer<CaptureResult>();

    await cameraController.startImageStream((CameraImage image) {
      if (!_isCapturing) {
        return;
      }

      final luminancePlane = image.planes.first;
      final bytes = luminancePlane.bytes;

      double sum = 0;
      int count = 0;
      for (int i = 0; i < bytes.length; i += 10) {
        sum += bytes[i];
        count++;
      }

      signal.add(count == 0 ? 0 : sum / count);
      timestamps.add(DateTime.now());
    });

    Future.delayed(captureDuration, () async {
      _isCapturing = false;
      await cameraController.stopImageStream();
      final samplingHz = _estimateSamplingHz(timestamps);
      completer.complete(CaptureResult(signal: signal, samplingHz: samplingHz));
    });

    return completer.future;
  }

  Future<CaptureResult> _captureWithSnapshots() async {
    _isCapturing = true;
    final signal = <double>[];
    final timestamps = <DateTime>[];

    final startedAt = DateTime.now();
    while (DateTime.now().difference(startedAt) < captureDuration) {
      if (!_isCapturing) {
        break;
      }

      try {
        final shot = await cameraController.takePicture();
        final bytes = await shot.readAsBytes();
        signal.add(_meanLuminanceFromJpeg(bytes));
        timestamps.add(DateTime.now());
      } catch (_) {
      }

      await Future.delayed(const Duration(milliseconds: 250));
    }

    _isCapturing = false;
    final samplingHz = _estimateSamplingHz(timestamps);
    return CaptureResult(signal: signal, samplingHz: samplingHz);
  }

  double _meanLuminanceFromJpeg(Uint8List bytes) {
    final decoded = img.decodeImage(bytes);
    if (decoded == null) {
      return 0.0;
    }

    final stepX = (decoded.width / 48).ceil().clamp(1, decoded.width);
    final stepY = (decoded.height / 48).ceil().clamp(1, decoded.height);

    double sum = 0.0;
    int count = 0;
    for (int y = 0; y < decoded.height; y += stepY) {
      for (int x = 0; x < decoded.width; x += stepX) {
        final pixel = decoded.getPixel(x, y);
        sum += 0.2126 * pixel.r + 0.7152 * pixel.g + 0.0722 * pixel.b;
        count++;
      }
    }

    return count == 0 ? 0.0 : sum / count;
  }

  double _estimateSamplingHz(List<DateTime> timestamps) {
    if (timestamps.length < 2) {
      return 30.0;
    }
    final elapsedMs = timestamps.last.difference(timestamps.first).inMilliseconds;
    if (elapsedMs <= 0) {
      return 30.0;
    }
    return (timestamps.length - 1) / (elapsedMs / 1000.0);
  }
}

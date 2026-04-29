import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:http/http.dart' as http;
import 'package:permission_handler/permission_handler.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:uuid/uuid.dart';

import '../features/capture/capture_controller.dart';
import '../features/hrv/hrv_calculator.dart';
import '../features/stress/stress_engine.dart';
import '../shared/api_client.dart';

class CaptureScreen extends StatefulWidget {
  final String userId;

  const CaptureScreen({super.key, required this.userId});

  @override
  State<CaptureScreen> createState() => _CaptureScreenState();
}

class _CaptureScreenState extends State<CaptureScreen> {
  // Use 192.168.1.103 for physical device access
  final _edgeUrlController = TextEditingController(text: 'http://192.168.1.103:8001');
  final _notifier = FlutterLocalNotificationsPlugin();
  final _stressEngine = const StressEngine();

  CameraController? _cameraController;
  CaptureController? _captureController;
  bool _busy = false;
  bool _torchOn = false;
  String? _deviceId;
  String _status = 'Pronto para iniciar captura orientada';
  HrvMetrics? _metrics;
  double? _stressProb;

  @override
  void initState() {
    super.initState();
    _initNotifications();
    _initSettings();
    _initDeviceId();
    _initCamera();
  }

  Future<void> _initSettings() async {
    // URL Edge Service forced to default for demo purposes
  }

  Future<bool> _validateEdgeEndpoint() async {
    final raw = _edgeUrlController.text.trim();
    final uri = Uri.tryParse(raw);
    if (uri == null || !uri.hasScheme || uri.host.isEmpty) {
      setState(() => _status = 'URL do Edge inválida');
      return false;
    }

    if (kIsWeb && uri.scheme != 'https' && uri.host != '127.0.0.1' && uri.host != 'localhost') {
      setState(() => _status = 'No iPhone/web, use URL HTTPS no campo Edge Service');
      return false;
    }

    try {
      final healthUri = uri.replace(path: '/health', query: '');
      final res = await http.get(healthUri).timeout(const Duration(seconds: 6));
      if (res.statusCode < 200 || res.statusCode >= 300) {
        setState(() => _status = 'Edge indisponível (${res.statusCode})');
        return false;
      }
      return true;
    } catch (_) {
      setState(() => _status = 'Não consegui conectar ao Edge. Verifique URL/Internet.');
      return false;
    }
  }

  Future<void> _initDeviceId() async {
    String id;
    try {
      final prefs = await SharedPreferences.getInstance();
      id = prefs.getString('device_id') ?? '';
      if (id.isEmpty) {
        id = const Uuid().v4();
        await prefs.setString('device_id', id);
      }
    } catch (_) {
      id = 'ephemeral-${const Uuid().v4()}';
    }

    if (!mounted) {
      return;
    }

    setState(() {
      _deviceId = id;
    });
  }

  Future<void> _initNotifications() async {
    const androidSettings = AndroidInitializationSettings('@mipmap/ic_launcher');
    await _notifier.initialize(const InitializationSettings(android: androidSettings));
  }

  Future<void> _initCamera() async {
    try {
      if (kIsWeb) {
        final isSecure = Uri.base.scheme == 'https' ||
            Uri.base.host == 'localhost' ||
            Uri.base.host == '127.0.0.1';
        if (!isSecure) {
          setState(() {
            _status =
                'No iPhone/Safari, a câmera no navegador exige HTTPS. Abra este app via URL https.';
          });
          return;
        }
      } else {
        final cameraPermission = await Permission.camera.request();
        if (!cameraPermission.isGranted) {
          setState(() => _status = 'Permissão da câmera negada');
          return;
        }
      }

      final cameras = await availableCameras();
      if (cameras.isEmpty) {
        setState(() => _status = 'Nenhuma câmera disponível neste dispositivo');
        return;
      }

      final backCamera = cameras.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.back,
        orElse: () => cameras.first,
      );

      final controller = CameraController(
        backCamera,
        ResolutionPreset.medium,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.yuv420,
      );

      await controller.initialize();

      setState(() {
        _cameraController = controller;
        _captureController = CaptureController(cameraController: controller);
        _status = 'Posicione o dedo sobre câmera + flash e inicie';
      });
    } catch (e) {
      setState(() {
        _status = 'Falha ao inicializar câmera: $e';
      });
    }
  }

  Future<void> _startGuidedCapture() async {
    final captureController = _captureController;
    final cameraController = _cameraController;
    if (captureController == null || cameraController == null || !cameraController.value.isInitialized) {
      setState(() => _status = 'Câmera não inicializada');
      return;
    }

    final edgeOk = await _validateEdgeEndpoint();
    if (!edgeOk) {
      return;
    }

    var deviceId = _deviceId;
    if (deviceId == null || deviceId.isEmpty) {
      deviceId = 'ephemeral-${const Uuid().v4()}';
      setState(() {
        _deviceId = deviceId;
        _status = 'Identificação persistente indisponível; usando ID temporário para esta sessão';
      });
    }

    setState(() {
      _busy = true;
      _status = 'Capturando PPG por 30 segundos... mantenha o dedo estável';
    });

    try {
      final capture = await captureController.startCapture();
      final metrics = HrvCalculator.fromSignal(capture.signal, capture.samplingHz);
      final localInference = _stressEngine.infer(metrics);

      final api = ApiClient(edgeBaseUrl: _edgeUrlController.text.trim());
      Map<String, dynamic>? remote;

      try {
        remote = await api.predictStress(
          metrics: metrics,
          signal: capture.signal,
          samplingHz: capture.samplingHz,
          userId: widget.userId,
          deviceId: deviceId,
        );
      } catch (_) {
        remote = null;
      }

      final sentToBackend = remote != null;
      final stressProb = (remote?['stress_probability'] as num?)?.toDouble() ?? localInference.probability;
      final isHighStress = (remote?['high_stress'] as bool?) ?? localInference.isHighStress;

      if (isHighStress) {
        await _notifyStress(stressProb);
      }

      setState(() {
        _metrics = metrics;
        _stressProb = stressProb;
        if (!sentToBackend) {
          _status =
              'Captura concluída localmente, mas envio ao backend falhou. Verifique URL do Edge (HTTPS no iPhone) e conexão.';
        } else {
          _status = isHighStress
              ? 'Alerta: possível estresse detectado'
              : 'Sem alerta crítico no momento';
        }
      });
    } catch (e) {
      setState(() => _status = 'Erro na captura: $e');
    } finally {
      setState(() => _busy = false);
    }
  }

  Future<void> _toggleTorch() async {
    final controller = _cameraController;
    if (controller == null || !controller.value.isInitialized) {
      setState(() => _status = 'Câmera não inicializada');
      return;
    }

    if (kIsWeb) {
      setState(() {
        _status = 'Lanterna não é suportada de forma confiável no navegador do iPhone.';
      });
      return;
    }

    try {
      final next = !_torchOn;
      await controller.setFlashMode(next ? FlashMode.torch : FlashMode.off);
      setState(() {
        _torchOn = next;
        _status = _torchOn
            ? 'Lanterna ligada. Posicione o dedo sobre câmera + flash.'
            : 'Lanterna desligada.';
      });
    } catch (e) {
      setState(() {
        _status = 'Não foi possível controlar a lanterna: $e';
      });
    }
  }

  Future<void> _notifyStress(double probability) async {
    const details = NotificationDetails(
      android: AndroidNotificationDetails(
        'stress-alert',
        'Stress Alerts',
        importance: Importance.high,
        priority: Priority.high,
      ),
    );

    await _notifier.show(
      1,
      'Atenção: possível estresse',
      'Seu HRV mudou bastante (risco ${(probability * 100).toStringAsFixed(1)}%). Considere pausa/respiração.',
      details,
    );
  }

  @override
  void dispose() {
    _cameraController?.dispose();
    _edgeUrlController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Stress HRV Monitor')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Card(
              child: Padding(
                padding: const EdgeInsets.all(12),
                child: Row(
                  children: [
                    const Icon(Icons.verified_user_outlined, color: Color(0xFF8B7BFF)),
                    const SizedBox(width: 8),
                    Expanded(child: Text('Usuário autenticado: ${widget.userId}')),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 8),
            // URL Edge Service exposed for deployment config
            Padding(
              padding: const EdgeInsets.only(top: 8),
              child: TextField(
                controller: _edgeUrlController,
                decoration: const InputDecoration(
                  labelText: 'Edge Service URL',
                  border: OutlineInputBorder(),
                  isDense: true,
                  prefixIcon: Icon(Icons.link),
                ),
                style: const TextStyle(fontSize: 12),
              ),
            ),
            if (_deviceId != null)
              Padding(
                padding: const EdgeInsets.only(top: 8),
                child: Text('Device ID: $_deviceId', style: const TextStyle(fontSize: 10, color: Colors.grey)),
              ),
            const SizedBox(height: 12),
            Expanded(
              child: _cameraController != null && _cameraController!.value.isInitialized
                  ? ClipRRect(
                      borderRadius: BorderRadius.circular(12),
                      child: CameraPreview(_cameraController!),
                    )
                  : const Center(child: CircularProgressIndicator()),
            ),
            const SizedBox(height: 12),
            Text(_status, style: const TextStyle(color: Colors.white70)),
            if (_metrics != null) ...[
              Text('RMSSD: ${_metrics!.rmssd.toStringAsFixed(2)} ms'),
              Text('SDNN: ${_metrics!.sdnn.toStringAsFixed(2)} ms'),
              Text('pNN50: ${_metrics!.pnn50.toStringAsFixed(2)} %'),
              Text('HR médio: ${_metrics!.meanHr.toStringAsFixed(2)} bpm'),
            ],
            if (_stressProb != null) Text('Prob. estresse: ${(_stressProb! * 100).toStringAsFixed(1)}%'),
            const SizedBox(height: 8),
            OutlinedButton(
              onPressed: _busy ? null : _toggleTorch,
              child: Text(_torchOn ? 'Desligar lanterna' : 'Ligar lanterna'),
            ),
            const SizedBox(height: 8),
            FilledButton(
              onPressed: _busy ? null : _startGuidedCapture,
              child: Text(_busy ? 'Processando...' : 'Iniciar captura guiada'),
            ),
          ],
        ),
      ),
    );
  }
}

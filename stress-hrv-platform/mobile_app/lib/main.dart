import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'auth_session.dart';
import 'screens/auth_screen.dart';
import 'screens/dashboard_screen.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const StressHrvApp());
}

class StressHrvApp extends StatelessWidget {
  const StressHrvApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Olivia',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        brightness: Brightness.light, 
        scaffoldBackgroundColor: const Color(0xFFF8FAFC), // Slate-50
        colorScheme: const ColorScheme.light(
          primary: Color(0xFF8B5CF6),
          secondary: Color(0xFF10B981),
          surface: Color(0xFFFFFFFF),
        ),
        textTheme: GoogleFonts.interTextTheme(ThemeData.light().textTheme),
        useMaterial3: true,
      ),
      home: const AuthGateScreen(),
    );
  }
}

class AuthGateScreen extends StatefulWidget {
  const AuthGateScreen({super.key});

  @override
  State<AuthGateScreen> createState() => _AuthGateScreenState();
}

class _AuthGateScreenState extends State<AuthGateScreen> {
  AuthSession? _session;
  bool _loading = true;

  @override
  void initState() {
    super.initState();
    _loadSession();
  }

  Future<void> _loadSession() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final token = prefs.getString('auth_token') ?? '';
      final name = prefs.getString('auth_user_name') ?? '';
      final email = prefs.getString('auth_user_email') ?? '';
      final id = prefs.getInt('auth_user_id') ?? 0;
      if (token.isNotEmpty && email.isNotEmpty) {
        _session = AuthSession(token: token, userName: name, userEmail: email, userId: id);
      }
    } catch (_) {
    }
    if (!mounted) {
      return;
    }
    setState(() {
      _loading = false;
    });
  }

  Future<void> _saveSession(AuthSession session) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('auth_token', session.token);
    await prefs.setString('auth_user_name', session.userName);
    await prefs.setString('auth_user_email', session.userEmail);
    await prefs.setInt('auth_user_id', session.userId);
  }

  Future<void> _clearSession() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove('auth_token');
    await prefs.remove('auth_user_name');
    await prefs.remove('auth_user_email');
    await prefs.remove('auth_user_id');
  }

  @override
  Widget build(BuildContext context) {
    if (_loading) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }

    if (_session == null) {
      return AuthScreen(
        onLoginSuccess: (session) async {
          await _saveSession(session);
          if (!mounted) {
            return;
          }
          setState(() => _session = session);
        },
      );
    }

    return DashboardScreen(
      session: _session!,
      onLogout: () async {
        await _clearSession();
        if (!mounted) {
          return;
        }
        setState(() => _session = null);
      },
    );
  }
}

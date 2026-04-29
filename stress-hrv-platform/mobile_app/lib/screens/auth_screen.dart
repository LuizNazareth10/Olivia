import 'dart:convert';
import 'dart:ui';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';
import '../auth_session.dart'; // for AuthSession

class AuthScreen extends StatefulWidget {
  final Function(AuthSession) onLoginSuccess;
  
  const AuthScreen({super.key, required this.onLoginSuccess});

  @override
  State<AuthScreen> createState() => _AuthScreenState();
}

class _AuthScreenState extends State<AuthScreen> {
  bool _isLogin = true;
  bool _isLoading = false;
  final _formKey = GlobalKey<FormState>();
  
  final _nameController = TextEditingController();
  final _emailController = TextEditingController();
  final _passwordController = TextEditingController();
  // 192.168.1.103 é o seu IP local atual
  final _serverUrlController = TextEditingController(text: 'http://192.168.1.103:8002');

  // Updated Modern Light Theme Colors
  final Color _primaryColor = const Color(0xFF8B5CF6); // Violet-500
  final Color _secondaryColor = const Color(0xFF10B981); // Emerald-500
  final Color _bg = const Color(0xFFF8FAFC);       // Slate-50
  final Color _surface = const Color(0xFFFFFFFF);  // White
  final Color _textMain = const Color(0xFF0F172A); // Slate-900
  final Color _textSub = const Color(0xFF64748B);  // Slate-500

  @override
  void dispose() {
    _nameController.dispose();
    _emailController.dispose();
    _passwordController.dispose();
    _serverUrlController.dispose();
    super.dispose();
  }

  Future<void> _submit() async {
    if (!_formKey.currentState!.validate()) return;
    
    setState(() => _isLoading = true);
    
    // API Call
    // Get URL from user input or default
    final baseUrl = _serverUrlController.text.isNotEmpty 
        ? _serverUrlController.text.trim() 
        : 'http://127.0.0.1:8002';

    final endpoint = _isLogin ? '/auth/login' : '/auth/register';
    
    try {
      final body = <String, dynamic>{
        'email': _emailController.text.trim(), // Server expects 'email', not 'username' for login
        'password': _passwordController.text,
      };
      if (!_isLogin) {
         body['name'] = _nameController.text.trim();
         // Register endpoint might expect strict fields, ensure extra fields don't break validation
         // or match exactly what RegisterPayload expects if different.
         body['username'] = _emailController.text.trim(); // Just in case
         body['full_name'] = _nameController.text.trim();
      }

      final uri = Uri.parse('$baseUrl$endpoint');
      final resp = await http.post(
        uri,
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode(body),
      );

      if (resp.statusCode == 200 || resp.statusCode == 201) {
        final data = jsonDecode(resp.body);
        if (data != null) { 
           // Check for explicit error from backend
           if (data['ok'] == false) {
             _showError("Failed: ${data['reason'] ?? 'Unknown error'}");
             return;
           }

           // Access token might be directly in response or under 'access_token'
           final token = data['access_token'] ?? data['token'];
           final user = data['user'] ?? {};
           
           if (token != null) {
              final session = AuthSession(
                token: token,
                userName: user['name'] ?? user['full_name'] ?? user['username'] ?? _nameController.text.trim(),
                userEmail: user['email'] ?? _emailController.text.trim(),
                userId: user['id'] ?? 0,
              );
              
              // Persist
              final prefs = await SharedPreferences.getInstance();
              await prefs.setString('auth_token', session.token);
              await prefs.setString('auth_user_name', session.userName);
              await prefs.setString('auth_user_email', session.userEmail);
              if (session.userId != 0) await prefs.setInt('auth_user_id', session.userId);

              widget.onLoginSuccess(session);
           } else {
             _showError("Login successful but no token returned.");
           }
        } else {
           _showError("Auth failed: Unknown response");
        }
      } else {
        _showError("Server error: ${resp.statusCode}");
      }
    } catch (e) {
      _showError("Connection error: $e. Target: $baseUrl");
    } finally {
      if (mounted) setState(() => _isLoading = false);
    }
  }

  void _showError(String msg) {
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(
      content: Text(msg, style: GoogleFonts.inter(color: Colors.white)), 
      backgroundColor: Colors.redAccent,
      behavior: SnackBarBehavior.floating,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
      margin: const EdgeInsets.all(16),
    ));
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: _bg,
      body: Stack(
        children: [
          // Background Blobs (Light)
          Positioned(
            top: -100,
            right: -100,
            child: Container(
              width: 300,
              height: 300,
              decoration: BoxDecoration(shape: BoxShape.circle, color: _primaryColor.withOpacity(0.05)),
            ).blurred(blur: 80),
          ),
          Positioned(
            bottom: -50,
            left: -50,
            child: Container(
              width: 250,
              height: 250,
              decoration: BoxDecoration(shape: BoxShape.circle, color: _secondaryColor.withOpacity(0.05)),
            ).blurred(blur: 80),
          ),

          Center(
            child: SingleChildScrollView(
              padding: const EdgeInsets.all(24),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  // Logo
                  Container(
                    width: 70, height: 70,
                    decoration: BoxDecoration(
                      gradient: LinearGradient(colors: [_primaryColor, Colors.purpleAccent]),
                      borderRadius: BorderRadius.circular(20),
                      boxShadow: [BoxShadow(color: _primaryColor.withOpacity(0.3), blurRadius: 20, offset: const Offset(0, 10))]
                    ),
                    child: const Center(child: Text("O", style: TextStyle(fontSize: 32, fontWeight: FontWeight.bold, color: Colors.white))),
                  ),
                  const SizedBox(height: 24),
                  Text(
                    "Bem-vindo à Olivia",
                    style: GoogleFonts.inter(fontSize: 28, fontWeight: FontWeight.bold, color: _textMain),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    _isLogin ? "Entre para continuar monitorando" : "Crie sua conta para começar",
                    style: GoogleFonts.inter(fontSize: 14, color: _textSub),
                  ),
                  const SizedBox(height: 40),

                  // Form Container
                  Container(
                    padding: const EdgeInsets.all(24),
                    decoration: BoxDecoration(
                      color: _surface,
                      borderRadius: BorderRadius.circular(24),
                      boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.05), blurRadius: 20, offset: const Offset(0, 4))],
                      border: Border.all(color: Colors.black.withOpacity(0.02)),
                    ),
                    child: Form(
                      key: _formKey,
                      child: Column(
                        children: [
                          if (!_isLogin) ...[
                            _buildTextField("Nome Completo", _nameController, icon: Icons.person_outline),
                            const SizedBox(height: 16),
                          ],
                          _buildTextField("Email", _emailController, icon: Icons.email_outlined),
                          const SizedBox(height: 16),
                          _buildTextField("Senha", _passwordController, isPass: true, icon: Icons.lock_outline),
                          const SizedBox(height: 16),
                          // Server URL configuration for production/deployment
                          _buildTextField("URL do Servidor", _serverUrlController, icon: Icons.cloud_outlined),
                          const SizedBox(height: 24),
                          
                          SizedBox(
                            width: double.infinity,
                            height: 50,
                            child: ElevatedButton(
                              onPressed: _isLoading ? null : _submit,
                              style: ElevatedButton.styleFrom(
                                backgroundColor: _primaryColor,
                                foregroundColor: Colors.white,
                                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                                elevation: 0,
                              ),
                              child: _isLoading 
                                ? const SizedBox(width: 20, height: 20, child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white)) 
                                : Text(_isLogin ? "Entrar" : "Criar Conta", style: GoogleFonts.inter(fontWeight: FontWeight.w600, fontSize: 16)),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                  
                  const SizedBox(height: 24),
                  TextButton(
                    onPressed: () => setState(() => _isLogin = !_isLogin),
                    child: RichText(
                      text: TextSpan(
                        text: _isLogin ? "Não tem uma conta? " : "Já tem uma conta? ",
                        style: GoogleFonts.inter(color: _textSub),
                        children: [
                          TextSpan(
                            text: _isLogin ? "Cadastre-se" : "Entrar",
                            style: GoogleFonts.inter(color: _primaryColor, fontWeight: FontWeight.bold),
                          )
                        ],
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildTextField(String label, TextEditingController controller, {bool isPass = false, IconData? icon}) {
    return TextFormField(
      controller: controller,
      obscureText: isPass,
      style: GoogleFonts.inter(color: _textMain),
      decoration: InputDecoration(
        labelText: label,
        labelStyle: GoogleFonts.inter(color: _textSub),
        prefixIcon: icon != null ? Icon(icon, color: _textSub.withOpacity(0.5), size: 20) : null,
        filled: true,
        fillColor: _bg, // Light grey fill
        border: OutlineInputBorder(borderRadius: BorderRadius.circular(12), borderSide: BorderSide.none),
        enabledBorder: OutlineInputBorder(borderRadius: BorderRadius.circular(12), borderSide: const BorderSide(color: Colors.transparent)),
        focusedBorder: OutlineInputBorder(borderRadius: BorderRadius.circular(12), borderSide: BorderSide(color: _primaryColor, width: 1.5)),
      ),
      validator: (v) => v!.isEmpty ? "Obrigatório" : null,
    );
  }
}

extension WidgetExt on Widget {
  Widget blurred({double blur = 20}) {
    return ImageFiltered(
      imageFilter: ImageFilter.blur(sigmaX: blur, sigmaY: blur),
      child: this,
    );
  }
}

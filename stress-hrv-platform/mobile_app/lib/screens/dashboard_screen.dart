import 'dart:ui';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:fl_chart/fl_chart.dart';
import 'package:intl/intl.dart';
import '../auth_session.dart';
import 'capture_screen.dart';

class DashboardScreen extends StatefulWidget {
  final AuthSession session;
  final VoidCallback onLogout;
  
  const DashboardScreen({super.key, required this.session, required this.onLogout});

  @override
  State<DashboardScreen> createState() => _DashboardScreenState();
}

class _DashboardScreenState extends State<DashboardScreen> {
  int _currentIndex = 0;
  
  // Feature 4: Chart Data (Dummy)
  final List<FlSpot> _historyData = [
    const FlSpot(0, 0.3), const FlSpot(1, 0.4), const FlSpot(2, 0.35),
    const FlSpot(3, 0.6), const FlSpot(4, 0.2), const FlSpot(5, 0.25),
    const FlSpot(6, 0.3),
  ];

  // Feature 5: Notifications Toggle
  bool _notificationsEnabled = true;

  @override
  Widget build(BuildContext context) {
    // Updated Modern Light Theme
    final bg = const Color(0xFFF8FAFC); // Slate-50
    final surface = const Color(0xFFFFFFFF); // White
    final textMain = const Color(0xFF0F172A); // Slate-900
    final textSub = const Color(0xFF64748B); // Slate-500
    final primary = const Color(0xFF8B5CF6);
    final secondary = const Color(0xFF10B981);
    
    final screens = [
      _buildHome(context, textMain, textSub, surface),
      _buildHistory(context, textMain, textSub, surface),
      _buildTips(context, textMain, textSub, surface), // Feature 3
      _buildProfile(context, textMain, textSub, surface), // Feature 2
    ];

    return Scaffold(
      backgroundColor: bg,
      body: Stack(
        children: [
           // Animated Background Blobs (Lighter)
           Positioned(
             top: -100, left: 0,
             child: Container(width: 300, height: 300, decoration: BoxDecoration(shape: BoxShape.circle, color: primary.withOpacity(0.05))).blurred(blur: 90),
           ),
           Positioned(
             bottom: 100, right: -50,
             child: Container(width: 250, height: 250, decoration: BoxDecoration(shape: BoxShape.circle, color: secondary.withOpacity(0.05))).blurred(blur: 90),
           ),
           
           SafeArea(child: screens[_currentIndex]),
        ],
      ),
      bottomNavigationBar: Container(
        decoration: BoxDecoration(
          color: surface,
          border: Border(top: BorderSide(color: Colors.black.withOpacity(0.05))),
          boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.02), blurRadius: 10, offset: const Offset(0, -2))],
        ),
        child: BottomNavigationBar(
          currentIndex: _currentIndex,
          onTap: (i) => setState(() => _currentIndex = i),
          backgroundColor: Colors.transparent,
          elevation: 0,
          selectedItemColor: primary,
          unselectedItemColor: textSub.withOpacity(0.6),
          showSelectedLabels: true,
          showUnselectedLabels: true,
          type: BottomNavigationBarType.fixed,
          items: const [
            BottomNavigationBarItem(icon: Icon(Icons.grid_view_rounded), label: 'Home'),
            BottomNavigationBarItem(icon: Icon(Icons.insights_rounded), label: 'History'),
            BottomNavigationBarItem(icon: Icon(Icons.lightbulb_outline_rounded), label: 'Dicas'),
            BottomNavigationBarItem(icon: Icon(Icons.person_rounded), label: 'Perfil'),
          ],
        ),
      ),
      floatingActionButton: _currentIndex == 0 ? FloatingActionButton(
        onPressed: () {
            Navigator.of(context).push(MaterialPageRoute(builder: (_) => CaptureScreen(userId: widget.session.userEmail)));
        },
        backgroundColor: secondary,
        foregroundColor: Colors.white,
        elevation: 4,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        child: const Icon(Icons.camera_alt_rounded, size: 28),
      ) : null,
    );
  }

  Widget _buildHome(BuildContext context, Color textMain, Color textSub, Color surface) {
    return ListView(
      padding: const EdgeInsets.all(24),
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('Olá, ${widget.session.userName}', style: GoogleFonts.inter(fontSize: 24, fontWeight: FontWeight.bold, color: textMain)),
                Text('Como você está hoje?', style: GoogleFonts.inter(fontSize: 14, color: textSub)),
              ],
            ),
            IconButton(
              onPressed: widget.onLogout,
              icon: Icon(Icons.logout_rounded, color: textSub),
            )
          ],
        ),
        const SizedBox(height: 24),
        
        // Feature 4: Chart (Visualization)
        Container(
          height: 200,
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            color: surface,
            borderRadius: BorderRadius.circular(20),
            boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.03), blurRadius: 15, offset: const Offset(0, 4))],
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
               Text('Tendência de Estresse', style: GoogleFonts.inter(fontSize: 16, fontWeight: FontWeight.w600, color: textMain)),
               const SizedBox(height: 12),
               Expanded(
                 child: LineChart(
                   LineChartData(
                     gridData: FlGridData(show: false),
                     titlesData: FlTitlesData(show: false),
                     borderData: FlBorderData(show: false),
                     lineBarsData: [
                       LineChartBarData(
                         spots: _historyData,
                         isCurved: true,
                         color: const Color(0xFF8B5CF6),
                         barWidth: 3,
                         dotData: FlDotData(show: false),
                         belowBarData: BarAreaData(show: true, color: const Color(0xFF8B5CF6).withOpacity(0.1)),
                       )
                     ]
                   )
                 ),
               )
            ],
          ),
        ),
        const SizedBox(height: 20),
        Text('Ações Rápidas', style: GoogleFonts.inter(fontSize: 18, fontWeight: FontWeight.w600, color: textMain)),
        const SizedBox(height: 12),
        Row(
          children: [
            Expanded(child: _ActionCard(icon: Icons.play_arrow_rounded, label: 'Iniciar\nCaptura', color: const Color(0xFF10B981), onTap: () => Navigator.of(context).push(MaterialPageRoute(builder: (_) => CaptureScreen(userId: widget.session.userEmail))))),
            const SizedBox(width: 12),
            Expanded(child: _ActionCard(icon: Icons.history_rounded, label: 'Ver\nHistórico', color: const Color(0xFF6366F1), onTap: () => setState(() => _currentIndex = 1))),
          ],
        ),
      ],
    );
  }

  Widget _buildHistory(BuildContext context, Color textMain, Color textSub, Color surface) {
    // Feature 1: History List
    return ListView(
      padding: const EdgeInsets.all(24),
      children: [
        Text('Histórico de Medições', style: GoogleFonts.inter(fontSize: 22, fontWeight: FontWeight.bold, color: textMain)),
        const SizedBox(height: 16),
        for (int i = 0; i < 5; i++)
          Container(
            margin: const EdgeInsets.only(bottom: 12),
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: surface,
              borderRadius: BorderRadius.circular(16),
              boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.02), blurRadius: 8)],
            ),
            child: Row(
              children: [
                Container(
                  width: 48, height: 48,
                  decoration: BoxDecoration(color: (i % 2 == 0 ? Colors.green : Colors.orange).withOpacity(0.1), borderRadius: BorderRadius.circular(12)),
                  child: Icon(i % 2 == 0 ? Icons.sentiment_satisfied_rounded : Icons.sentiment_dissatisfied_rounded, color: i % 2 == 0 ? Colors.green : Colors.orange),
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(DateFormat('dd/MM/yyyy HH:mm').format(DateTime.now().subtract(Duration(days: i))), style: GoogleFonts.inter(fontSize: 14, fontWeight: FontWeight.w600, color: textMain)),
                      Text(i % 2 == 0 ? 'Baixo Estresse' : 'Estresse Moderado', style: GoogleFonts.inter(fontSize: 12, color: textSub)),
                    ],
                  ),
                ),
                Text('${(80 + i * 2)} HR', style: GoogleFonts.inter(fontWeight: FontWeight.bold, color: textMain)),
              ],
            ),
          )
      ],
    );
  }

  Widget _buildTips(BuildContext context, Color textMain, Color textSub, Color surface) {
    // Feature 3: Tips
    final tips = [
      {'title': 'Respire Fundo', 'desc': 'A respiração profunda ativa o sistema parassimpático e reduz o estresse imediato.', 'icon': Icons.air_rounded},
      {'title': 'Hidratação', 'desc': 'A desidratação leve pode aumentar os níveis de cortisol.', 'icon': Icons.water_drop_rounded},
      {'title': 'Pausa Ativa', 'desc': 'Movimente-se por 5 minutos a cada hora de trabalho.', 'icon': Icons.accessibility_new_rounded},
    ];

    return ListView(
      padding: const EdgeInsets.all(24),
      children: [
        Text('Dicas de Bem-estar', style: GoogleFonts.inter(fontSize: 22, fontWeight: FontWeight.bold, color: textMain)),
        const SizedBox(height: 16),
        ...tips.map((t) => Container(
          margin: const EdgeInsets.only(bottom: 12),
          padding: const EdgeInsets.all(20),
          decoration: BoxDecoration(
            color: surface,
            borderRadius: BorderRadius.circular(16),
            border: Border.all(color: Colors.black.withOpacity(0.03)),
          ),
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Icon(t['icon'] as IconData, color: const Color(0xFF8B5CF6), size: 32),
              const SizedBox(width: 16),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(t['title'] as String, style: GoogleFonts.inter(fontSize: 16, fontWeight: FontWeight.w600, color: textMain)),
                    const SizedBox(height: 4),
                    Text(t['desc'] as String, style: GoogleFonts.inter(fontSize: 13, color: textSub, height: 1.4)),
                  ],
                ),
              )
            ],
          ),
        )).toList()
      ],
    );
  }

  Widget _buildProfile(BuildContext context, Color textMain, Color textSub, Color surface) {
    // Feature 2 & 5: Profile + Notifications
    return ListView(
      padding: const EdgeInsets.all(24),
      children: [
        Center(
          child: Column(
            children: [
              CircleAvatar(
                radius: 40, 
                backgroundColor: const Color(0xFF8B5CF6), 
                child: Text(
                  widget.session.userName.isNotEmpty ? widget.session.userName[0].toUpperCase() : "?", 
                  style: const TextStyle(fontSize: 32, color: Colors.white)
                )
              ),
              const SizedBox(height: 16),
              Text(widget.session.userName.isNotEmpty ? widget.session.userName : "Usuário", style: GoogleFonts.inter(fontSize: 20, fontWeight: FontWeight.bold, color: textMain)),
              Text(widget.session.userEmail, style: GoogleFonts.inter(fontSize: 14, color: textSub)),
            ],
          ),
        ),
        const SizedBox(height: 32),
        Container(
          decoration: BoxDecoration(color: surface, borderRadius: BorderRadius.circular(16)),
          child: Column(
            children: [
              SwitchListTile(
                value: _notificationsEnabled,
                onChanged: (v) => setState(() => _notificationsEnabled = v),
                title: Text('Notificações de Estresse', style: GoogleFonts.inter(fontWeight: FontWeight.w500, color: textMain)),
                secondary: Icon(Icons.notifications_outlined, color: textSub),
                activeColor: const Color(0xFF8B5CF6),
              ),
              const Divider(height: 1, indent: 16, endIndent: 16),
              ListTile(
                title: Text('Editar Perfil', style: GoogleFonts.inter(fontWeight: FontWeight.w500, color: textMain)),
                leading: Icon(Icons.edit_outlined, color: textSub),
                onTap: () {},
              )
            ],
          ),
        )
      ],
    );
  }
}

class _ActionCard extends StatelessWidget {
  final IconData icon;
  final String label;
  final Color color;
  final VoidCallback onTap;

  const _ActionCard({required this.icon, required this.label, required this.color, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        height: 110,
        decoration: BoxDecoration(
          color: color,
          borderRadius: BorderRadius.circular(20),
          boxShadow: [BoxShadow(color: color.withOpacity(0.3), blurRadius: 10, offset: const Offset(0, 4))],
        ),
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Icon(icon, color: Colors.white, size: 28),
            Text(label, style: GoogleFonts.inter(color: Colors.white, fontWeight: FontWeight.w600, fontSize: 13)),
          ],
        ),
      ),
    );
  }
}

extension BlurExt on Container {
  Widget blurred({required double blur}) => ImageFiltered(
    imageFilter: ImageFilter.blur(sigmaX: blur, sigmaY: blur),
    child: this,
  );
}

import 'dart:io' show File;

import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';

import 'api.dart';
import 'game_page.dart';
import 'history_page.dart';
import 'main.dart';
import 'stats_page.dart';

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  final api = EngineApi();
  EngineInfo? info;
  String? engineError;
  String level = 'normal';
  String humanColor = 'black';

  @override
  void initState() {
    super.initState();
    _ping();
    // 展示/驗證用鉤子：從 Mac 寫入本 app 容器 Documents/autodemo.txt
    // （內容 game / stats），啟動即自動導頁；讀後即刪。
    WidgetsBinding.instance.addPostFrameCallback((_) async {
      final docs = await getApplicationDocumentsDirectory();
      final f = File('${docs.path}/autodemo.txt');
      if (!f.existsSync()) return;
      final mode = f.readAsStringSync().trim();
      f.deleteSync();
      if (!mounted) return;
      if (mode == 'game') {
        Navigator.push(
          context,
          MaterialPageRoute(
            builder: (_) => const GamePage(
              level: 'normal',
              humanColor: 'black',
              autoDemo: true,
            ),
          ),
        );
      } else if (mode.startsWith('stats')) {
        // 'stats' 或 'stats-scroll:0.45'（捲到頁高的 45%）
        final parts = mode.split(':');
        final frac = parts.length > 1 ? double.tryParse(parts[1]) ?? 0.0 : 0.0;
        Navigator.push(
          context,
          MaterialPageRoute(
            builder: (_) => StatsPage(autoScrollFraction: frac),
          ),
        );
      }
    });
  }

  Future<void> _ping() async {
    setState(() => engineError = null);
    try {
      final i = await api.health();
      setState(() => info = i);
    } catch (_) {
      setState(() => engineError = '連不上對弈引擎，請確認網路連線後重試');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: LayoutBuilder(
          builder: (context, constraints) => SingleChildScrollView(
            padding: const EdgeInsets.symmetric(horizontal: 28),
            child: ConstrainedBox(
              constraints: BoxConstraints(minHeight: constraints.maxHeight),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  const SizedBox(height: 24),
                  // 標題：朱印 + 玄石
                  Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    crossAxisAlignment: CrossAxisAlignment.center,
                    children: [
                      const SealGlyph(
                        '弈',
                        size: 54,
                        fontSize: 34,
                        filled: true,
                      ),
                      const SizedBox(width: 16),
                      const Text(
                        '玄石',
                        style: TextStyle(
                          fontSize: 52,
                          fontWeight: FontWeight.w700,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 6),
                  const Text(
                    'GoZero · 自研 Gumbel-AlphaZero 九路圍棋',
                    textAlign: TextAlign.center,
                    style: TextStyle(fontSize: 14, color: Sumi.paperDim),
                  ),
                  const SizedBox(height: 40),
                  _section('執子', [
                    _choice(
                      '執黑（先行）',
                      'black',
                      humanColor,
                      (v) => setState(() => humanColor = v),
                    ),
                    _choice(
                      '執白',
                      'white',
                      humanColor,
                      (v) => setState(() => humanColor = v),
                    ),
                  ]),
                  const SizedBox(height: 18),
                  _section('棋力', [
                    _choice(
                      '直覺 · 純策略網路',
                      'easy',
                      level,
                      (v) => setState(() => level = v),
                    ),
                    _choice(
                      '均衡 · 32 次搜索',
                      'normal',
                      level,
                      (v) => setState(() => level = v),
                    ),
                    _choice(
                      '深思 · 128 次搜索',
                      'strong',
                      level,
                      (v) => setState(() => level = v),
                    ),
                  ]),
                  const SizedBox(height: 32),
                  FilledButton(
                    style: FilledButton.styleFrom(
                      backgroundColor: Sumi.seal,
                      padding: const EdgeInsets.symmetric(vertical: 16),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(10),
                      ),
                    ),
                    onPressed: info == null
                        ? null
                        : () => Navigator.push(
                            context,
                            MaterialPageRoute(
                              builder: (_) => GamePage(
                                level: level,
                                humanColor: humanColor,
                              ),
                            ),
                          ),
                    child: const Text(
                      '開始對弈',
                      style: TextStyle(
                        fontSize: 20,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                  ),
                  const SizedBox(height: 12),
                  Row(
                    children: [
                      Expanded(
                        child: OutlinedButton.icon(
                          style: _secondaryButtonStyle(),
                          onPressed: () => Navigator.push(
                            context,
                            MaterialPageRoute(
                              builder: (_) => const HistoryPage(),
                            ),
                          ),
                          icon: const SealGlyph('錄', size: 22, fontSize: 13),
                          label: const Text(
                            '對戰紀錄',
                            style: TextStyle(fontSize: 16),
                          ),
                        ),
                      ),
                      const SizedBox(width: 10),
                      Expanded(
                        child: OutlinedButton.icon(
                          style: _secondaryButtonStyle(),
                          onPressed: () => Navigator.push(
                            context,
                            MaterialPageRoute(
                              builder: (_) => const StatsPage(),
                            ),
                          ),
                          icon: const SealGlyph('能', size: 22, fontSize: 13),
                          label: const Text(
                            '模型性能',
                            style: TextStyle(fontSize: 16),
                          ),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 20),
                  if (engineError != null)
                    GestureDetector(
                      onTap: _ping,
                      child: Text(
                        '⚠ $engineError\n（點此重試）',
                        textAlign: TextAlign.center,
                        style: const TextStyle(
                          color: Sumi.danger,
                          fontSize: 12,
                        ),
                      ),
                    )
                  else
                    Text(
                      info == null
                          ? '正在連線引擎…'
                          : '引擎已連線 · ${info!.model} · 迭代 ${info!.iteration}',
                      textAlign: TextAlign.center,
                      style: const TextStyle(
                        color: Sumi.paperDim,
                        fontSize: 12,
                      ),
                    ),
                  const SizedBox(height: 16),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _section(String title, List<Widget> children) => Column(
    crossAxisAlignment: CrossAxisAlignment.start,
    children: [
      Text(
        title,
        style: const TextStyle(
          fontSize: 14,
          color: Sumi.paperDim,
          fontWeight: FontWeight.w600,
          letterSpacing: 4,
        ),
      ),
      const SizedBox(height: 8),
      ...children,
    ],
  );

  ButtonStyle _secondaryButtonStyle() => OutlinedButton.styleFrom(
    foregroundColor: Sumi.paper,
    side: const BorderSide(color: Sumi.paperDim),
    padding: const EdgeInsets.symmetric(vertical: 14),
    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
  );

  Widget _choice(
    String label,
    String value,
    String group,
    ValueChanged<String> onTap,
  ) {
    final selected = value == group;
    return GestureDetector(
      onTap: () => onTap(value),
      child: Container(
        margin: const EdgeInsets.only(bottom: 8),
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
        decoration: BoxDecoration(
          color: selected ? Sumi.panel : Colors.transparent,
          borderRadius: BorderRadius.circular(10),
          border: Border.all(
            color: selected ? Sumi.seal : Sumi.panel,
            width: selected ? 1.5 : 1,
          ),
        ),
        child: Row(
          children: [
            Icon(
              selected ? Icons.circle : Icons.circle_outlined,
              size: 14,
              color: selected ? Sumi.seal : Sumi.paperDim,
            ),
            const SizedBox(width: 12),
            Text(
              label,
              style: TextStyle(
                fontSize: 16,
                color: selected ? Sumi.paper : Sumi.paperDim,
                fontWeight: selected ? FontWeight.w600 : FontWeight.w400,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

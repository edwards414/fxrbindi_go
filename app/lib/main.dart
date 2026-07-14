import 'package:flutter/material.dart';

import 'home_page.dart';

/// 玄石 GoZero — 與自研 Gumbel-AlphaZero 模型對弈的 9 路圍棋 app。
/// 引擎後端: gozero/server.py (http://127.0.0.1:8765，模擬器與主機共用網路)
void main() => runApp(const GoZeroApp());

/// 墨 × 原木 × 宣紙 配色
class Sumi {
  static const bg = Color(0xFF17130F); // 深墨
  static const panel = Color(0xFF241E18); // 硯台
  static const paper = Color(0xFFEDE3D2); // 宣紙米白
  static const paperDim = Color(0xFFB8A78F);
  static const seal = Color(0xFFB03A2E); // 朱印紅（品牌強調／選中／正向）
  static const danger = Color(0xFFE2544A); // 警示紅（錯誤／不可逆動作，故意比朱印紅更亮更暖，一眼可辨）
  static const woodHi = Color(0xFFDCA968); // 榧木
  static const woodLo = Color(0xFFB07E42);
  static const line = Color(0xFF3A2F24); // 棋盤線
}

/// 朱印方章造型的字符標記——實心＝主品牌（弈），外框＝次要功能（錄／能）。
/// 讓 app 內的「圖示」跟品牌識別共用同一套視覺語言，而不是外掛 Material 圖示
/// （History/Insights 那種圓角線性圖示跟水墨宣紙的東方書法質感明顯是兩個世界）。
class SealGlyph extends StatelessWidget {
  final String char;
  final double size;
  final double fontSize;
  final bool filled;

  const SealGlyph(
    this.char, {
    super.key,
    required this.size,
    required this.fontSize,
    this.filled = false,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      width: size,
      height: size,
      alignment: Alignment.center,
      decoration: BoxDecoration(
        color: filled ? Sumi.seal : Colors.transparent,
        borderRadius: BorderRadius.circular(filled ? 6 : size * 0.22),
        border: filled ? null : Border.all(color: Sumi.paperDim, width: 1),
      ),
      child: Text(
        char,
        style: TextStyle(
          fontSize: fontSize,
          fontWeight: FontWeight.w800,
          color: filled ? Sumi.paper : Sumi.paperDim,
          height: 1.0,
        ),
      ),
    );
  }
}

class GoZeroApp extends StatelessWidget {
  const GoZeroApp({super.key});

  @override
  Widget build(BuildContext context) {
    const serif = ['Songti TC', 'Songti SC', 'PingFang TC'];
    return MaterialApp(
      title: '玄石 GoZero',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        brightness: Brightness.dark,
        scaffoldBackgroundColor: Sumi.bg,
        colorScheme: const ColorScheme.dark(
          primary: Sumi.seal,
          surface: Sumi.panel,
          onSurface: Sumi.paper,
        ),
        fontFamilyFallback: serif,
        textTheme: Typography.whiteMountainView.apply(
          bodyColor: Sumi.paper,
          displayColor: Sumi.paper,
          fontFamilyFallback: serif,
        ),
        useMaterial3: true,
      ),
      home: const HomePage(),
    );
  }
}

import 'package:flutter/material.dart';

import 'game_page.dart';
import 'main.dart';

/// 貼目說明裡的一條：朱印方章字符 + 一段文字。
class _KomiNote extends StatelessWidget {
  final String glyph;
  final String text;

  const _KomiNote(this.glyph, this.text);

  @override
  Widget build(BuildContext context) => Padding(
    padding: const EdgeInsets.only(bottom: 14),
    child: Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        SealGlyph(glyph, size: 22, fontSize: 14),
        const SizedBox(width: 12),
        Expanded(
          child: Text(
            text,
            style: const TextStyle(
              fontSize: 14,
              color: Sumi.paperDim,
              height: 1.6,
            ),
          ),
        ),
      ],
    ),
  );
}

/// 對弈設定頁：從首頁按「開始對弈」進入，選好執子與棋力後才進棋盤。
class GameSetupPage extends StatefulWidget {
  const GameSetupPage({super.key});

  @override
  State<GameSetupPage> createState() => _GameSetupPageState();
}

class _GameSetupPageState extends State<GameSetupPage> {
  // 記住本次 app 存活期間最後一次的選擇，重進設定頁不用重選。
  static String _lastLevel = 'normal';
  static String _lastColor = 'black';
  static double _lastKomi = 7.5;
  static int _lastHandicap = 0;

  static const komiPresets = [7.5, 6.5, 5.5, 0.5];

  late String level = _lastLevel;
  late String humanColor = _lastColor;
  late int handicap = _lastHandicap;
  late bool customKomi = !komiPresets.contains(_lastKomi);
  late double komi = _lastKomi;
  late final komiCtrl = TextEditingController(
    text: customKomi ? _fmtKomi(_lastKomi) : '',
  );
  String? komiError;

  static String _fmtKomi(double k) =>
      k == k.roundToDouble() ? k.toStringAsFixed(0) : k.toStringAsFixed(1);

  @override
  void dispose() {
    komiCtrl.dispose();
    super.dispose();
  }

  /// 回傳最終貼目；自訂輸入不合法時回 null 並設定錯誤訊息。
  double? _resolveKomi() {
    if (!customKomi) return komi;
    final v = double.tryParse(komiCtrl.text.trim());
    if (v == null ||
        (v * 2) != (v * 2).roundToDouble() ||
        v < -361 ||
        v > 361) {
      setState(() => komiError = '請輸入 -361 到 361 之間、以 0.5 為單位的貼目');
      return null;
    }
    return v;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Sumi.bg,
        title: const Text(
          '對弈設定',
          style: TextStyle(fontSize: 20, fontWeight: FontWeight.w600),
        ),
      ),
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
                  const SizedBox(height: 16),
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
                  const SizedBox(height: 18),
                  _section('貼目', [
                    Wrap(
                      spacing: 8,
                      runSpacing: 8,
                      children: [
                        for (final k in komiPresets)
                          _chip(
                            _fmtKomi(k),
                            !customKomi && komi == k,
                            () => setState(() {
                              customKomi = false;
                              komi = k;
                              komiError = null;
                            }),
                          ),
                        _chip(
                          '自訂',
                          customKomi,
                          () => setState(() => customKomi = true),
                        ),
                      ],
                    ),
                    if (customKomi) ...[
                      const SizedBox(height: 8),
                      TextField(
                        controller: komiCtrl,
                        keyboardType: const TextInputType.numberWithOptions(
                          decimal: true,
                          signed: true,
                        ),
                        style: const TextStyle(color: Sumi.paper, fontSize: 16),
                        decoration: InputDecoration(
                          hintText: '如 0、6.5、13.5',
                          hintStyle: const TextStyle(color: Sumi.paperDim),
                          errorText: komiError,
                          isDense: true,
                          filled: true,
                          fillColor: Sumi.panel,
                          border: OutlineInputBorder(
                            borderRadius: BorderRadius.circular(10),
                            borderSide: BorderSide.none,
                          ),
                        ),
                        onChanged: (_) => setState(() => komiError = null),
                      ),
                    ],
                    const Padding(
                      padding: EdgeInsets.only(top: 8),
                      child: Text(
                        '貼目是終局時加給白方的補償，用來抵銷黑棋先行之利',
                        style: TextStyle(fontSize: 12, color: Sumi.paperDim),
                      ),
                    ),
                  ], trailing: _helpLink(_showKomiHelp)),
                  const SizedBox(height: 18),
                  _section('讓子', [
                    Wrap(
                      spacing: 8,
                      runSpacing: 8,
                      children: [
                        _chip(
                          '無',
                          handicap == 0,
                          () => setState(() => handicap = 0),
                        ),
                        for (final n in const [2, 3, 4])
                          _chip(
                            '讓 $n 子',
                            handicap == n,
                            () => setState(() => handicap = n),
                          ),
                      ],
                    ),
                    if (handicap > 0)
                      const Padding(
                        padding: EdgeInsets.only(top: 8),
                        child: Text(
                          '讓子由黑方先擺，白方接著行棋；慣例貼目 0.5',
                          style: TextStyle(fontSize: 12, color: Sumi.paperDim),
                        ),
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
                    onPressed: () {
                      final resolvedKomi = _resolveKomi();
                      if (resolvedKomi == null) return;
                      _lastLevel = level;
                      _lastColor = humanColor;
                      _lastKomi = resolvedKomi;
                      _lastHandicap = handicap;
                      Navigator.push(
                        context,
                        MaterialPageRoute(
                          builder: (_) => GamePage(
                            level: level,
                            humanColor: humanColor,
                            komi: resolvedKomi,
                            handicap: handicap,
                          ),
                        ),
                      );
                    },
                    child: const Text(
                      '開始對弈',
                      style: TextStyle(
                        fontSize: 20,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                  ),
                  const SizedBox(height: 24),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _section(String title, List<Widget> children, {Widget? trailing}) =>
      Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
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
              if (trailing != null) ...[const Spacer(), trailing],
            ],
          ),
          const SizedBox(height: 8),
          ...children,
        ],
      );

  /// 章節標題右側的「說明」連結。用朱印方章字符當圖示，跟 SealGlyph 的
  /// 品牌視覺語言一致（見 main.dart 為何不用 Material 圖示）。
  Widget _helpLink(VoidCallback onTap) => GestureDetector(
    behavior: HitTestBehavior.opaque,
    onTap: onTap,
    child: const Padding(
      padding: EdgeInsets.symmetric(vertical: 4, horizontal: 2),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          SealGlyph('問', size: 16, fontSize: 11),
          SizedBox(width: 6),
          Text(
            '說明',
            style: TextStyle(
              fontSize: 13,
              color: Sumi.paperDim,
              decoration: TextDecoration.underline,
              decorationColor: Sumi.paperDim,
            ),
          ),
        ],
      ),
    ),
  );

  void _showKomiHelp() => showDialog<void>(
    context: context,
    builder: (_) => AlertDialog(
      backgroundColor: Sumi.panel,
      title: const Text(
        '關於貼目',
        style: TextStyle(
          color: Sumi.paper,
          fontSize: 22,
          fontWeight: FontWeight.w700,
        ),
      ),
      content: SingleChildScrollView(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: const [
            _KomiNote('先', '黑棋先落子，天生佔了先行之利；貼目用來平衡這項優勢。'),
            _KomiNote(
              '補',
              '終局計點時，把貼目加到白方，用來抵銷這個先行之利。'
                  '玄石採中國規則（Tromp-Taylor 計點），預設貼目 7.5 目。',
            ),
            _KomiNote('半', '貼目帶半目，是為了讓雙方總分不可能相等，確保一定分出勝負、不會和棋。'),
            _KomiNote('讓', '讓子局中黑方開局已先佔數子，先行之利不再存在，因此慣例把貼目降到 0.5。'),
            _KomiNote('調', '調高貼目對白方有利，調低則對黑方有利。想加重讓步就往上調。'),
          ],
        ),
      ),
      actions: [
        FilledButton(
          style: FilledButton.styleFrom(backgroundColor: Sumi.seal),
          onPressed: () => Navigator.pop(context),
          child: const Text('知道了'),
        ),
      ],
    ),
  );

  Widget _chip(String label, bool selected, VoidCallback onTap) =>
      GestureDetector(
        onTap: onTap,
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
          decoration: BoxDecoration(
            color: selected ? Sumi.panel : Colors.transparent,
            borderRadius: BorderRadius.circular(20),
            border: Border.all(
              color: selected ? Sumi.seal : Sumi.panel,
              width: selected ? 1.5 : 1,
            ),
          ),
          child: Text(
            label,
            style: TextStyle(
              fontSize: 15,
              color: selected ? Sumi.paper : Sumi.paperDim,
              fontWeight: selected ? FontWeight.w600 : FontWeight.w400,
            ),
          ),
        ),
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

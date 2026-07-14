import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;

import 'main.dart';

/// 模型性能頁：架構 / 訓練 / 棋力評測 / 推理延遲 / 曲線圖。
/// 數據由 scripts 從 runs/ 的 metrics.jsonl 與評測結果生成 (assets/model_stats.json)。
class StatsPage extends StatefulWidget {
  final double autoScrollFraction; // 展示模式：自動捲動到頁高比例
  const StatsPage({super.key, this.autoScrollFraction = 0});

  @override
  State<StatsPage> createState() => _StatsPageState();
}

class _StatsPageState extends State<StatsPage> {
  Map<String, dynamic>? stats;
  final _scroll = ScrollController();

  @override
  void initState() {
    super.initState();
    rootBundle.loadString('assets/model_stats.json').then((s) {
      if (!mounted) return;
      setState(() => stats = jsonDecode(s));
      if (widget.autoScrollFraction > 0) {
        WidgetsBinding.instance.addPostFrameCallback(
          (_) => _scroll.animateTo(
            _scroll.position.maxScrollExtent * widget.autoScrollFraction,
            duration: const Duration(seconds: 1),
            curve: Curves.easeOut,
          ),
        );
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    final s = stats;
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Sumi.bg,
        title: const Text(
          '模型性能',
          style: TextStyle(fontSize: 20, fontWeight: FontWeight.w600),
        ),
      ),
      body: s == null
          ? const Center(child: CircularProgressIndicator(color: Sumi.seal))
          : ListView(
              controller: _scroll,
              padding: const EdgeInsets.fromLTRB(20, 8, 20, 32),
              children: [
                _hero(s),
                const SizedBox(height: 14),
                _card('架構 · AZNet', [
                  for (final row in s['architecture']) _kv(row[0], row[1]),
                ]),
                _card('推理延遲 · 每手（Apple Silicon CPU）', [
                  for (final row in s['latency']) _kv(row[0], row[1]),
                ]),
                _chartCard('訓練損失（前 ${s['iters_logged']} 迭代紀錄）', [
                  _Series(
                    '策略',
                    const Color(0xFFD8A657),
                    _pts(s['loss_curve'], 'iter', 'policy'),
                  ),
                  _Series(
                    '價值',
                    const Color(0xFF7FA8C9),
                    _pts(s['loss_curve'], 'iter', 'value'),
                  ),
                ]),
                _chartCard('對 pgx AlphaZero baseline 勝率爬升', [
                  _Series(
                    '勝率 %',
                    Sumi.seal,
                    _pts(s['baseline_curve'], 'iter', 'win'),
                  ),
                ], yMax: 100),
                _techDetails(s),
              ],
            ),
    );
  }

  /// 最重要的那個數字放大當主角：對 pgx baseline 的勝率。其餘評測數字降為次要清單。
  Widget _hero(Map<String, dynamic> s) {
    final evals = (s['evals'] as List).cast<Map<String, dynamic>>();
    final hero = evals.firstWhere(
      (e) => (e['opponent'] as String).contains('baseline'),
      orElse: () => evals.first,
    );
    final others = evals.where((e) => e != hero).toList();
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [Sumi.seal, Sumi.seal.withValues(alpha: 0.72)],
        ),
        borderRadius: BorderRadius.circular(14),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            '這隻 AI 有多強',
            style: TextStyle(
              fontSize: 13,
              fontWeight: FontWeight.w600,
              letterSpacing: 3,
              color: Sumi.paper.withValues(alpha: 0.85),
            ),
          ),
          const SizedBox(height: 8),
          Row(
            crossAxisAlignment: CrossAxisAlignment.baseline,
            textBaseline: TextBaseline.alphabetic,
            children: [
              Text(
                '${hero['winrate']}',
                style: const TextStyle(
                  fontSize: 56,
                  fontWeight: FontWeight.w800,
                  color: Sumi.paper,
                  height: 1.0,
                ),
              ),
              const Text(
                '%',
                style: TextStyle(
                  fontSize: 24,
                  fontWeight: FontWeight.w700,
                  color: Sumi.paper,
                ),
              ),
            ],
          ),
          const SizedBox(height: 4),
          Text(
            '勝率 · 對戰${hero['opponent']}（${hero['detail']}）',
            style: TextStyle(
              fontSize: 13,
              color: Sumi.paper.withValues(alpha: 0.9),
            ),
          ),
          if (others.isNotEmpty) ...[
            const SizedBox(height: 14),
            Divider(color: Sumi.paper.withValues(alpha: 0.25), height: 1),
            const SizedBox(height: 10),
            for (final e in others) _heroSecondaryRow(e),
          ],
          const SizedBox(height: 4),
          Text(
            s['evals_note'],
            style: TextStyle(
              fontSize: 11,
              color: Sumi.paper.withValues(alpha: 0.75),
            ),
          ),
        ],
      ),
    );
  }

  Widget _heroSecondaryRow(Map<String, dynamic> e) => Padding(
    padding: const EdgeInsets.only(bottom: 6),
    child: Row(
      children: [
        Expanded(
          child: Text(
            '${e['opponent']}',
            style: TextStyle(
              fontSize: 12,
              color: Sumi.paper.withValues(alpha: 0.85),
            ),
          ),
        ),
        Text(
          '${e['winrate']}%',
          style: const TextStyle(
            fontSize: 13,
            fontWeight: FontWeight.w700,
            color: Sumi.paper,
          ),
        ),
      ],
    ),
  );

  /// 超參數等工程細節預設收合，感興趣的人才展開看。
  Widget _techDetails(Map<String, dynamic> s) => Theme(
    data: Theme.of(context).copyWith(dividerColor: Colors.transparent),
    child: Container(
      margin: const EdgeInsets.only(bottom: 14),
      decoration: BoxDecoration(
        color: Sumi.panel,
        borderRadius: BorderRadius.circular(12),
      ),
      child: ExpansionTile(
        tilePadding: const EdgeInsets.symmetric(horizontal: 16),
        childrenPadding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
        iconColor: Sumi.paperDim,
        collapsedIconColor: Sumi.paperDim,
        title: const Text(
          '訓練細節（技術向）',
          style: TextStyle(
            fontSize: 15,
            fontWeight: FontWeight.w700,
            color: Sumi.paper,
          ),
        ),
        children: [for (final row in s['training']) _kv(row[0], row[1])],
      ),
    ),
  );

  List<Offset> _pts(List<dynamic> rows, String x, String y) => [
    for (final r in rows)
      Offset((r[x] as num).toDouble(), (r[y] as num).toDouble()),
  ];

  Widget _card(String title, List<Widget> children) => Container(
    margin: const EdgeInsets.only(bottom: 14),
    padding: const EdgeInsets.all(16),
    decoration: BoxDecoration(
      color: Sumi.panel,
      borderRadius: BorderRadius.circular(12),
    ),
    child: Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          title,
          style: const TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.w700,
            color: Sumi.paper,
          ),
        ),
        const SizedBox(height: 10),
        ...children,
      ],
    ),
  );

  Widget _kv(String k, String v) => Padding(
    padding: const EdgeInsets.only(bottom: 6),
    child: Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        SizedBox(
          width: 96,
          child: Text(
            k,
            style: const TextStyle(fontSize: 13, color: Sumi.paperDim),
          ),
        ),
        Expanded(
          child: Text(
            v,
            style: const TextStyle(fontSize: 13, color: Sumi.paper),
          ),
        ),
      ],
    ),
  );

  Widget _chartCard(String title, List<_Series> series, {double? yMax}) =>
      _card(title, [
        SizedBox(
          height: 160,
          child: CustomPaint(
            painter: _ChartPainter(series, yMaxOverride: yMax),
            size: const Size(double.infinity, 160),
          ),
        ),
        const SizedBox(height: 8),
        Row(
          children: [
            for (final s in series) ...[
              Container(width: 10, height: 3, color: s.color),
              const SizedBox(width: 4),
              Text(
                s.name,
                style: const TextStyle(fontSize: 12, color: Sumi.paperDim),
              ),
              const SizedBox(width: 14),
            ],
          ],
        ),
      ]);
}

class _Series {
  final String name;
  final Color color;
  final List<Offset> points; // x = iter, y = value
  _Series(this.name, this.color, this.points);
}

class _ChartPainter extends CustomPainter {
  final List<_Series> series;
  final double? yMaxOverride;
  _ChartPainter(this.series, {this.yMaxOverride});

  @override
  void paint(Canvas canvas, Size size) {
    final all = series.expand((s) => s.points).toList();
    if (all.isEmpty) return;
    final xMin = all.map((p) => p.dx).reduce((a, b) => a < b ? a : b);
    final xMax = all.map((p) => p.dx).reduce((a, b) => a > b ? a : b);
    final yMax =
        yMaxOverride ??
        all.map((p) => p.dy).reduce((a, b) => a > b ? a : b) * 1.1;

    const padL = 34.0, padB = 18.0, padT = 6.0;
    final plotW = size.width - padL, plotH = size.height - padB - padT;
    Offset map(Offset p) => Offset(
      padL + (p.dx - xMin) / (xMax - xMin + 1e-9) * plotW,
      padT + (1 - p.dy / yMax) * plotH,
    );

    final grid = Paint()
      ..color = Sumi.paperDim.withValues(alpha: 0.18)
      ..strokeWidth = 0.7;
    final tp = TextPainter(textDirection: TextDirection.ltr);
    for (var i = 0; i <= 4; i++) {
      final y = padT + plotH * i / 4;
      canvas.drawLine(Offset(padL, y), Offset(size.width, y), grid);
      tp.text = TextSpan(
        text: (yMax * (1 - i / 4)).toStringAsFixed(yMax >= 10 ? 0 : 1),
        style: const TextStyle(fontSize: 9, color: Sumi.paperDim),
      );
      tp.layout();
      tp.paint(canvas, Offset(padL - tp.width - 4, y - tp.height / 2));
    }
    for (final frac in [0.0, 0.5, 1.0]) {
      tp.text = TextSpan(
        text: (xMin + (xMax - xMin) * frac).round().toString(),
        style: const TextStyle(fontSize: 9, color: Sumi.paperDim),
      );
      tp.layout();
      tp.paint(
        canvas,
        Offset(padL + plotW * frac - tp.width * frac, size.height - tp.height),
      );
    }

    for (final s in series) {
      final paint = Paint()
        ..color = s.color
        ..strokeWidth = 1.8
        ..style = PaintingStyle.stroke
        ..strokeJoin = StrokeJoin.round;
      final path = Path();
      for (var i = 0; i < s.points.length; i++) {
        final p = map(s.points[i]);
        i == 0 ? path.moveTo(p.dx, p.dy) : path.lineTo(p.dx, p.dy);
      }
      canvas.drawPath(path, paint);
      if (s.points.length <= 12) {
        for (final p in s.points) {
          canvas.drawCircle(map(p), 3, Paint()..color = s.color);
        }
      }
    }
  }

  @override
  bool shouldRepaint(_ChartPainter old) => old.series != series;
}

import 'package:flutter/material.dart';

import 'main.dart';

/// 對局勝率曲線：黑勝率隨手數的折線（上 = 黑優，下 = 白優）。
class WinrateCurvePainter extends CustomPainter {
  final List<double> winrates; // winrates[i] = 第 i 手後的黑勝率
  WinrateCurvePainter(this.winrates);

  @override
  void paint(Canvas canvas, Size s) {
    const pad = 2.0;
    final plot = Rect.fromLTRB(pad, pad, s.width - pad, s.height - pad);
    Offset map(int i, double w) => Offset(
      plot.left +
          (winrates.length < 2 ? 0 : i / (winrates.length - 1)) * plot.width,
      plot.top + (1 - w) * plot.height,
    );

    // 50% 基準虛線
    final mid = plot.top + plot.height / 2;
    final dash = Paint()
      ..color = Sumi.paperDim.withValues(alpha: 0.35)
      ..strokeWidth = 0.8;
    for (double x = plot.left; x < plot.right; x += 7) {
      canvas.drawLine(Offset(x, mid), Offset(x + 3.5, mid), dash);
    }

    // 「黑／白」側標
    final tp = TextPainter(textDirection: TextDirection.ltr);
    for (final side in [('黑', plot.top), ('白', plot.bottom - 11)]) {
      tp.text = TextSpan(
        text: side.$1,
        style: TextStyle(
          fontSize: 10,
          color: Sumi.paperDim.withValues(alpha: 0.7),
        ),
      );
      tp.layout();
      tp.paint(canvas, Offset(plot.left, side.$2));
    }

    if (winrates.length < 2) return;

    // 曲線下（黑方勢力）淡墨填充
    final line = Path()..moveTo(map(0, winrates[0]).dx, map(0, winrates[0]).dy);
    for (var i = 1; i < winrates.length; i++) {
      final p = map(i, winrates[i]);
      line.lineTo(p.dx, p.dy);
    }
    final fill = Path.from(line)
      ..lineTo(plot.right, plot.bottom)
      ..lineTo(plot.left, plot.bottom)
      ..close();
    canvas.drawPath(
      fill,
      Paint()..color = Colors.black.withValues(alpha: 0.28),
    );
    canvas.drawPath(
      line,
      Paint()
        ..style = PaintingStyle.stroke
        ..strokeWidth = 1.8
        ..strokeJoin = StrokeJoin.round
        ..color = Sumi.paper,
    );

    // 最新一手
    final last = map(winrates.length - 1, winrates.last);
    canvas.drawCircle(last, 3.2, Paint()..color = Sumi.seal);
  }

  @override
  bool shouldRepaint(WinrateCurvePainter old) => old.winrates != winrates;
}

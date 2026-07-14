import 'dart:math' as math;
import 'dart:ui' as ui;

import 'package:flutter/material.dart';

import 'main.dart';

/// 榧木棋盤 + 雲子質感棋子。板面含座標 (A–J / 1–9)、星位、最後一手標記。
class BoardPainter extends CustomPainter {
  final int size; // 路數
  final List<int> board; // 0 空 1 黑 2 白
  final int? lastMove;
  final int? hover; // 觸點預覽
  final Set<int> entering; // 本回合新落的子（進場縮放）
  final Map<int, int> exiting; // 本回合被提掉的子：index -> 原本顏色（淡出）
  final Animation<double>? anim; // 0→1，entering/exiting 共用的進度

  BoardPainter({
    required this.size,
    required this.board,
    this.lastMove,
    this.hover,
    this.entering = const {},
    this.exiting = const {},
    this.anim,
  }) : super(repaint: anim);

  static const _gtpCols = 'ABCDEFGHJKLMNOPQRST'; // GTP 座標，跳過 I
  String get _cols => _gtpCols.substring(0, size);

  /// 星位（尺寸通用：9 路 2-2、13 路以上 3-3，奇數路含天元）
  List<List<int>> get _hoshi {
    final e = size >= 13 ? 3 : 2;
    final far = size - 1 - e;
    final pts = [
      [e, e],
      [e, far],
      [far, e],
      [far, far],
    ];
    if (size.isOdd) {
      final c = size ~/ 2;
      pts.add([c, c]);
      if (size >= 13) {
        pts.addAll([
          [e, c],
          [c, e],
          [c, far],
          [far, c],
        ]);
      }
    }
    return pts;
  }

  @override
  void paint(Canvas canvas, Size s) {
    final w = s.width;
    final margin = w * 0.085;
    final grid = (w - 2 * margin) / (size - 1);
    Offset pt(int r, int c) => Offset(margin + c * grid, margin + r * grid);

    _paintWood(canvas, s);

    // 棋盤線
    final line = Paint()
      ..color = Sumi.line
      ..strokeWidth = 1.1;
    final edge = Paint()
      ..color = Sumi.line
      ..strokeWidth = 2.0;
    for (var i = 0; i < size; i++) {
      canvas.drawLine(
        pt(i, 0),
        pt(i, size - 1),
        i == 0 || i == size - 1 ? edge : line,
      );
      canvas.drawLine(
        pt(0, i),
        pt(size - 1, i),
        i == 0 || i == size - 1 ? edge : line,
      );
    }

    // 星位
    final hoshi = Paint()..color = Sumi.line;
    for (final p in _hoshi) {
      canvas.drawCircle(pt(p[0], p[1]), grid * 0.09, hoshi);
    }

    // 座標
    final tp = TextPainter(textDirection: TextDirection.ltr);
    for (var i = 0; i < size; i++) {
      _label(canvas, tp, _cols[i], Offset(pt(0, i).dx, margin * 0.38), grid);
      _label(
        canvas,
        tp,
        _cols[i],
        Offset(pt(0, i).dx, w - margin * 0.38),
        grid,
      );
      _label(
        canvas,
        tp,
        '${size - i}',
        Offset(margin * 0.38, pt(i, 0).dy),
        grid,
      );
      _label(
        canvas,
        tp,
        '${size - i}',
        Offset(w - margin * 0.38, pt(i, 0).dy),
        grid,
      );
    }

    // 棋子
    final stoneR = grid * 0.47;
    final t = (anim?.value ?? 1.0).clamp(0.0, 1.0);
    final enterScale = Curves.easeOutBack.transform(t).clamp(0.0, 1.15);
    for (var i = 0; i < size * size; i++) {
      if (board[i] == 0) continue;
      final c = pt(i ~/ size, i % size);
      final isEntering = entering.contains(i);
      if (isEntering && enterScale <= 0.02) continue; // 剛開始，先不畫避免閃一下
      _paintStone(
        canvas,
        c,
        isEntering ? stoneR * enterScale : stoneR,
        black: board[i] == 1,
      );
    }

    // 剛被提掉的子：原地淡出，不會憑空消失
    if (exiting.isNotEmpty && t < 1.0) {
      for (final entry in exiting.entries) {
        final c = pt(entry.key ~/ size, entry.key % size);
        _paintStone(canvas, c, stoneR, black: entry.value == 1, opacity: 1 - t);
      }
    }

    // 觸點預覽（半透明）
    if (hover != null && hover! < size * size && board[hover!] == 0) {
      final c = pt(hover! ~/ size, hover! % size);
      canvas.drawCircle(
        c,
        stoneR,
        Paint()..color = Sumi.seal.withValues(alpha: 0.45),
      );
    }

    // 最後一手標記
    if (lastMove != null && lastMove! < size * size && board[lastMove!] != 0) {
      final c = pt(lastMove! ~/ size, lastMove! % size);
      final mark = Paint()
        ..style = PaintingStyle.stroke
        ..strokeWidth = 1.8
        ..color = board[lastMove!] == 1 ? Sumi.paper : Sumi.bg;
      canvas.drawCircle(c, stoneR * 0.45, mark);
    }
  }

  void _label(
    Canvas canvas,
    TextPainter tp,
    String text,
    Offset center,
    double grid,
  ) {
    tp.text = TextSpan(
      text: text,
      style: TextStyle(
        color: Sumi.line.withValues(alpha: 0.75),
        fontSize: grid * 0.30,
        fontWeight: FontWeight.w600,
      ),
    );
    tp.layout();
    tp.paint(canvas, center - Offset(tp.width / 2, tp.height / 2));
  }

  void _paintWood(Canvas canvas, Size s) {
    final rect = Offset.zero & s;
    final rrect = RRect.fromRectAndRadius(rect, const Radius.circular(10));
    canvas.save();
    canvas.clipRRect(rrect);
    canvas.drawRect(
      rect,
      Paint()
        ..shader = ui.Gradient.linear(
          rect.topLeft,
          rect.bottomRight,
          [Sumi.woodHi, const Color(0xFFCE9857), Sumi.woodLo],
          [0.0, 0.55, 1.0],
        ),
    );
    // 木紋：固定種子的細波紋
    final rnd = math.Random(9);
    final grain = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 0.8;
    for (var i = 0; i < 22; i++) {
      final y = rnd.nextDouble() * s.height;
      final amp = 1.5 + rnd.nextDouble() * 3.0;
      final phase = rnd.nextDouble() * math.pi * 2;
      final alpha = 0.05 + rnd.nextDouble() * 0.08;
      grain.color = const Color(0xFF6B4A22).withValues(alpha: alpha);
      final path = Path()..moveTo(0, y);
      for (double x = 0; x <= s.width; x += 8) {
        path.lineTo(x, y + math.sin(x / 34 + phase) * amp);
      }
      canvas.drawPath(path, grain);
    }
    canvas.restore();
  }

  void _paintStone(
    Canvas canvas,
    Offset c,
    double r, {
    required bool black,
    double opacity = 1.0,
  }) {
    if (opacity <= 0 || r <= 0) return;
    final fading = opacity < 1.0;
    if (fading) {
      canvas.saveLayer(
        Rect.fromCircle(center: c, radius: r * 1.4),
        Paint()..color = Color.fromRGBO(0, 0, 0, opacity),
      );
    }
    canvas.drawCircle(
      c + Offset(r * 0.08, r * 0.12),
      r,
      Paint()
        ..color = Colors.black.withValues(alpha: 0.35)
        ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 2.5),
    );
    final hi = c - Offset(r * 0.35, r * 0.4);
    final shader = black
        ? ui.Gradient.radial(
            hi,
            r * 2.1,
            [
              const Color(0xFF5A5A58),
              const Color(0xFF23211E),
              const Color(0xFF0D0C0A),
            ],
            [0.0, 0.45, 1.0],
          )
        : ui.Gradient.radial(
            hi,
            r * 2.1,
            [
              const Color(0xFFFFFEF9),
              const Color(0xFFEDE8DA),
              const Color(0xFFC9C2B0),
            ],
            [0.0, 0.5, 1.0],
          );
    canvas.drawCircle(c, r, Paint()..shader = shader);
    if (!black) {
      canvas.drawCircle(
        c,
        r,
        Paint()
          ..style = PaintingStyle.stroke
          ..strokeWidth = 0.6
          ..color = const Color(0xFF9A917E).withValues(alpha: 0.6),
      );
    }
    if (fading) canvas.restore();
  }

  @override
  bool shouldRepaint(BoardPainter old) =>
      old.board != board ||
      old.lastMove != lastMove ||
      old.hover != hover ||
      old.entering != entering ||
      old.exiting != exiting ||
      old.anim != anim;
}

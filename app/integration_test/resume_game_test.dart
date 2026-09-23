import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

import 'package:gozero_go/board_painter.dart';
import 'package:gozero_go/main.dart' as app;

/// 中途離開後從首頁「繼續對局」接回，且不扣第二次體力。
/// 需要 scripts/fake_engine_server.py（預設體力 10）在 ENGINE_BASE。
Future<void> waitFor(
  WidgetTester tester,
  Finder finder, {
  Duration timeout = const Duration(seconds: 20),
}) async {
  final end = DateTime.now().add(timeout);
  while (DateTime.now().isBefore(end)) {
    await tester.pump(const Duration(milliseconds: 200));
    if (finder.evaluate().isNotEmpty) return;
  }
  fail('等不到 $finder');
}

bool _enabled(WidgetTester tester, String label) {
  final button = tester.widget<ButtonStyleButton>(
    find.ancestor(of: find.text(label), matching: find.byType(OutlinedButton)),
  );
  return button.onPressed != null;
}

Future<void> playOneMove(WidgetTester tester) async {
  final board = find.byWidgetPredicate(
    (w) => w is CustomPaint && w.painter is BoardPainter,
  );
  await tester.tapAt(tester.getCenter(board));
  final end = DateTime.now().add(const Duration(seconds: 20));
  while (DateTime.now().isBefore(end)) {
    await tester.pump(const Duration(milliseconds: 200));
    if (_enabled(tester, '虛手')) return;
  }
  fail('AI 沒有回手');
}

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('中途離開 → 繼續對局 → 不再扣體力', (tester) async {
    app.main();
    await tester.pump(const Duration(seconds: 1));
    await waitFor(tester, find.textContaining('引擎已連線'));
    await waitFor(tester, find.textContaining('體力 10／10'));
    expect(find.text('繼續對局'), findsNothing);

    await tester.tap(find.text('開始對弈'));
    await tester.pumpAndSettle();
    await tester.tap(find.text('9 路 · 快速對弈'));
    await tester.pump();
    await tester.ensureVisible(find.text('開始對弈'));
    await tester.pumpAndSettle();
    await tester.tap(find.text('開始對弈'));
    await waitFor(tester, find.text('悔棋 3/3'));
    await playOneMove(tester); // 黑一手 + AI 回手 = 第 2 手

    // 中途離開：棋盤頁 → 設定頁 → 首頁
    await tester.tap(find.byTooltip('Back').last);
    await tester.pumpAndSettle();
    await tester.tap(find.byTooltip('Back').last);
    await tester.pumpAndSettle();
    await waitFor(tester, find.text('繼續對局'));
    expect(find.textContaining('9 路 · 均衡 · 第 2 手'), findsOneWidget);
    await waitFor(tester, find.textContaining('體力 9／10'));

    // 接回：盤面還在（第 2 手後輪到黑，悔棋鈕可用），體力不變
    await tester.tap(find.text('繼續對局'));
    await waitFor(tester, find.text('悔棋 3/3'));
    await waitFor(tester, find.textContaining('對弈 · 9路'));
    expect(_enabled(tester, '悔棋 3/3'), isTrue, reason: '接回的局已有兩手，可悔');
    await playOneMove(tester);

    // 認輸結束 → 回首頁後「繼續對局」消失，體力仍是 9
    for (var attempt = 0; attempt < 4; attempt++) {
      await tester.tap(find.text('認輸').first, warnIfMissed: false);
      final end = DateTime.now().add(const Duration(seconds: 3));
      while (DateTime.now().isBefore(end)) {
        await tester.pump(const Duration(milliseconds: 200));
        if (find.text('確定認輸？').evaluate().isNotEmpty) break;
      }
      if (find.text('確定認輸？').evaluate().isNotEmpty) break;
    }
    await tester.tap(find.widgetWithText(FilledButton, '認輸'));
    await waitFor(tester, find.text('檢視棋盤'));
    await tester.tap(find.text('檢視棋盤'));
    await tester.pump(const Duration(milliseconds: 400));
    await tester.pump(const Duration(milliseconds: 400));
    await tester.tap(find.byTooltip('Back').last);
    await tester.pumpAndSettle();
    expect(find.text('繼續對局'), findsNothing);
    await waitFor(tester, find.textContaining('體力 9／10'));
  });
}

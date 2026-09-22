import 'dart:io' show File;

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:path_provider/path_provider.dart';

import 'package:gozero_go/board_painter.dart';
import 'package:gozero_go/main.dart' as app;

/// 體力與悔棋上限的端到端流程。需要 scripts/fake_engine_server.py（或真引擎）
/// 以 --stamina-max 3 啟動，且此裝置的玩家體力為滿。
///
/// 截圖：binding.takeScreenshot 在 iOS 模擬器會卡住，所以改成把
/// Documents/shot_NAME.req 寫出去，由外面的 shell 迴圈用 simctl 截圖。
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

Future<void> shot(WidgetTester tester, String name) async {
  final docs = await getApplicationDocumentsDirectory();
  final req = File('${docs.path}/shot_$name.req');
  req.writeAsStringSync(name);
  final end = DateTime.now().add(const Duration(seconds: 8));
  while (req.existsSync() && DateTime.now().isBefore(end)) {
    await tester.pump(const Duration(milliseconds: 250));
  }
}

bool _enabled(WidgetTester tester, String label) {
  final button = tester.widget<ButtonStyleButton>(
    find.ancestor(of: find.text(label), matching: find.byType(OutlinedButton)),
  );
  return button.onPressed != null;
}

Future<void> tapCenter(WidgetTester tester) async {
  final board = find.byWidgetPredicate(
    (w) => w is CustomPaint && w.painter is BoardPainter,
  );
  await tester.tapAt(tester.getCenter(board));
}

Future<void> playOneMove(WidgetTester tester, String undoLabelAfter) async {
  await tapCenter(tester);
  // 等 AI 回手：悔棋鈕重新啟用
  final end = DateTime.now().add(const Duration(seconds: 20));
  while (DateTime.now().isBefore(end)) {
    await tester.pump(const Duration(milliseconds: 200));
    if (find.text(undoLabelAfter).evaluate().isNotEmpty &&
        _enabled(tester, undoLabelAfter)) {
      return;
    }
  }
  fail('AI 沒有回手');
}

Future<void> resignAndExpectResult(WidgetTester tester) async {
  // 棋盤頁的「思考中」呼吸動畫永遠在跑，pumpAndSettle 不會靜止，只能定量 pump
  // 上一個對話框的退場動畫可能還沒結束、遮罩會吃掉第一下，所以按不到就再按
  for (var attempt = 0; attempt < 4; attempt++) {
    await tester.tap(find.text('認輸').first, warnIfMissed: false);
    final end = DateTime.now().add(const Duration(seconds: 3));
    while (DateTime.now().isBefore(end)) {
      await tester.pump(const Duration(milliseconds: 200));
      if (find.text('確定認輸？').evaluate().isNotEmpty) break;
    }
    if (find.text('確定認輸？').evaluate().isNotEmpty) break;
  }
  expect(find.text('確定認輸？'), findsOneWidget);
  await tester.tap(find.widgetWithText(FilledButton, '認輸'));
  await waitFor(tester, find.text('檢視棋盤'));
}

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('體力 3 點用完會擋開局；每局只能悔棋 3 次', (tester) async {
    app.main();
    await tester.pump(const Duration(seconds: 1));
    await waitFor(tester, find.textContaining('引擎已連線'));
    await waitFor(tester, find.textContaining('體力 3／3'));
    await shot(tester, '10_home_full');

    await tester.tap(find.text('開始對弈'));
    await tester.pumpAndSettle();
    expect(find.text('對弈設定'), findsOneWidget);
    await tester.tap(find.text('9 路 · 快速對弈'));
    await tester.pump();
    // 設定頁比一屏長，按鈕在體力列上方、畫面底部之外
    await tester.ensureVisible(find.text('開始對弈'));
    await tester.pumpAndSettle();
    await tester.tap(find.text('開始對弈'));

    // 第一局：開局扣 1 點，悔棋額度 3
    await waitFor(tester, find.text('悔棋 3/3'));
    await waitFor(tester, find.textContaining('對弈 · 9路'));
    expect(_enabled(tester, '悔棋 3/3'), isFalse, reason: '還沒下子不能悔');

    await playOneMove(tester, '悔棋 3/3');
    await tester.tap(find.text('悔棋 3/3'));
    await waitFor(tester, find.text('悔棋 2/3'));
    await shot(tester, '11_game_undo_2_left');

    await playOneMove(tester, '悔棋 2/3');
    await tester.tap(find.text('悔棋 2/3'));
    await waitFor(tester, find.text('悔棋 1/3'));

    await playOneMove(tester, '悔棋 1/3');
    await tester.tap(find.text('悔棋 1/3'));
    await waitFor(tester, find.text('悔棋 0/3'));

    await playOneMove(tester, '虛手'); // 只等回手；悔棋鈕此時應永遠停用
    await waitFor(tester, find.text('悔棋 0/3'));
    expect(_enabled(tester, '悔棋 0/3'), isFalse, reason: '第 4 次悔棋要被擋');
    await shot(tester, '12_game_undo_exhausted');

    // 認輸 → 再來一局（第二局，剩 1 點）
    await resignAndExpectResult(tester);
    await tester.tap(find.text('再來一局'));
    await waitFor(tester, find.text('悔棋 3/3'));
    await tester.pump(const Duration(milliseconds: 500));

    // 認輸 → 再來一局（第三局，剩 0 點）
    await resignAndExpectResult(tester);
    await tester.tap(find.text('再來一局'));
    await waitFor(tester, find.text('悔棋 3/3'));
    await tester.pump(const Duration(milliseconds: 500));

    // 認輸 → 對話框的「再來一局」變成停用的「體力不足」
    await resignAndExpectResult(tester);
    expect(find.text('再來一局'), findsNothing);
    expect(find.widgetWithText(FilledButton, '體力不足'), findsOneWidget);
    await shot(tester, '13_result_no_stamina');
    await tester.tap(find.text('檢視棋盤'));
    await tester.pump(const Duration(milliseconds: 400));
    await tester.pump(const Duration(milliseconds: 400));

    // 回設定頁：開始鈕停用、顯示倒數
    await tester.tap(find.byTooltip('Back').last);
    await tester.pumpAndSettle();
    expect(find.text('對弈設定'), findsOneWidget);
    await waitFor(tester, find.textContaining('體力 0／3'));
    expect(find.textContaining('後回復 1 點'), findsOneWidget);
    final start = tester.widget<FilledButton>(
      find.widgetWithText(FilledButton, '體力不足'),
    );
    expect(start.onPressed, isNull);
    await shot(tester, '14_setup_no_stamina');

    // 回首頁：同樣 0/3；按開始只會出現提示，不會進設定頁
    await tester.tap(find.byTooltip('Back').last);
    await tester.pumpAndSettle();
    await waitFor(tester, find.textContaining('體力 0／3'));
    await tester.tap(find.text('開始對弈'));
    await tester.pumpAndSettle();
    expect(find.text('對弈設定'), findsNothing);
    expect(find.textContaining('體力用完了'), findsOneWidget);
    await shot(tester, '15_home_no_stamina');
  });
}

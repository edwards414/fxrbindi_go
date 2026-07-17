import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

import 'package:gozero_go/main.dart' as app;

/// 導航流程測試：首頁（無執子/棋力）→ 開始對弈 → 對弈設定頁 → 進棋盤。
/// 需要引擎伺服器在線（api.dart 的 engineBase），與真機使用情境相同。
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

void main() {
  final binding = IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  // flutter test 直接跑時沒有 driver 收圖，截圖會失敗——靜默跳過即可。
  Future<void> shot(WidgetTester tester, String name) async {
    try {
      await binding.takeScreenshot(name);
    } catch (_) {}
    await tester.pump();
  }

  testWidgets('首頁 → 對弈設定 → 棋盤', (tester) async {
    app.main();
    await tester.pump(const Duration(seconds: 1));

    // 首頁：不再有執子/棋力選項，保留引擎連線狀態列。
    expect(find.text('玄石'), findsOneWidget);
    expect(find.text('執子'), findsNothing);
    expect(find.text('棋力'), findsNothing);
    expect(find.text('執黑（先行）'), findsNothing);

    // 等引擎連上（開始對弈按鈕才會啟用）。
    await waitFor(tester, find.textContaining('引擎已連線'));
    await shot(tester, '01_home');
    await tester.tap(find.text('開始對弈'));
    await tester.pumpAndSettle();

    // 對弈設定頁：執子與棋力選擇搬到這裡。
    expect(find.text('對弈設定'), findsOneWidget);
    expect(find.text('執子'), findsOneWidget);
    expect(find.text('棋力'), findsOneWidget);
    expect(find.text('執黑（先行）'), findsOneWidget);
    expect(find.text('深思 · 128 次搜索'), findsOneWidget);
    await shot(tester, '02_setup');

    // 選深思後開始，應進入棋盤頁。
    await tester.tap(find.text('深思 · 128 次搜索'));
    await tester.pump();
    await tester.tap(find.text('開始對弈'));
    await waitFor(tester, find.text('對弈 · 深思'));
    expect(find.text('虛手'), findsOneWidget);
    expect(find.text('認輸'), findsOneWidget);
    await shot(tester, '03_game');

    // 返回設定頁應記住剛才的選擇（堆疊裡設定頁的返回鈕仍在樹上，點最上層那顆）。
    await tester.tap(find.byTooltip('Back').last);
    await tester.pumpAndSettle();
    expect(find.text('對弈設定'), findsOneWidget);
  });
}

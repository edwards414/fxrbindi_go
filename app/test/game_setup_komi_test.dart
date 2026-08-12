import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:gozero_go/game_setup_page.dart';

void main() {
  Future<void> pumpSetup(WidgetTester tester) => tester.pumpWidget(
    const MaterialApp(home: GameSetupPage()),
  );

  testWidgets('貼目區塊常駐一行規則說明', (tester) async {
    await pumpSetup(tester);
    expect(
      find.text('貼目是終局時加給白方的補償，用來抵銷黑棋先行之利'),
      findsOneWidget,
    );
  });

  testWidgets('點「說明」開啟貼目說明對話框，五條都在', (tester) async {
    await pumpSetup(tester);
    expect(find.text('關於貼目'), findsNothing);

    await tester.tap(find.text('說明'));
    await tester.pumpAndSettle();

    expect(find.text('關於貼目'), findsOneWidget);
    // 五條說明各自的朱印字符圖示
    for (final glyph in ['先', '補', '半', '讓', '調']) {
      expect(find.text(glyph), findsOneWidget, reason: '缺少「$glyph」這條');
    }
    expect(find.textContaining('Tromp-Taylor'), findsOneWidget);

    await tester.tap(find.text('知道了'));
    await tester.pumpAndSettle();
    expect(find.text('關於貼目'), findsNothing);
  });
}

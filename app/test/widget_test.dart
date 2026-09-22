import 'package:flutter_test/flutter_test.dart';

import 'package:gozero_go/main.dart';

void main() {
  testWidgets('app boots to the home screen', (WidgetTester tester) async {
    await tester.pumpWidget(const GoZeroApp());
    expect(find.text('玄石'), findsOneWidget);
    expect(find.text('開始對弈'), findsOneWidget);
  });
}

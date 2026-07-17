import 'dart:io';

import 'package:integration_test/integration_test_driver_extended.dart';

/// flutter drive 用的 driver：把測試裡 binding.takeScreenshot() 的影像存到
/// --dart-define=SCREENSHOT_DIR 指定的資料夾（預設 build/verify_screenshots）。
Future<void> main() async {
  final dir =
      Platform.environment['SCREENSHOT_DIR'] ?? 'build/verify_screenshots';
  await integrationDriver(
    onScreenshot: (name, bytes, [args]) async {
      final f = File('$dir/$name.png');
      f.createSync(recursive: true);
      f.writeAsBytesSync(bytes);
      return true;
    },
  );
}

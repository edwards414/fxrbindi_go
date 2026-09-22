import 'dart:io' show File;
import 'dart:math';

import 'package:path_provider/path_provider.dart';

/// 這台裝置上的匿名玩家識別碼，第一次啟動時隨機產生、存在 app 的 Documents 裡。
///
/// 沒有帳號系統；伺服器只靠它記體力。重裝 app 會拿到新的 ID（體力重置），
/// 這是刻意接受的小漏洞——目的只是讓同時進行的對局數有個上限，不是防作弊。
class PlayerId {
  static const _fileName = 'player_id.txt';
  static final _random = Random.secure();
  static String? _cached;

  static Future<String> get() async {
    final cached = _cached;
    if (cached != null) return cached;
    final docs = await getApplicationDocumentsDirectory();
    final file = File('${docs.path}/$_fileName');
    String? id;
    try {
      if (file.existsSync()) {
        final text = file.readAsStringSync().trim();
        if (RegExp(r'^[A-Za-z0-9._:-]{8,64}$').hasMatch(text)) id = text;
      }
    } catch (_) {
      /* 讀不到就重新產生 */
    }
    if (id == null) {
      id = List.generate(
        16,
        (_) => _random.nextInt(256).toRadixString(16).padLeft(2, '0'),
      ).join();
      try {
        file.writeAsStringSync(id);
      } catch (_) {
        /* 寫不進去也照用；下次啟動會再產生一個 */
      }
    }
    return _cached = id;
  }
}

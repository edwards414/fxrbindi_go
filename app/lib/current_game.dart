import 'dart:convert';
import 'dart:io' show Directory, File;

import 'package:path_provider/path_provider.dart';

import 'api.dart';

/// 進行中的那一局。中途離開（返回首頁、切到別的 app、被系統殺掉）都不會丟：
/// 伺服器端棋局本來就會保留一段時間，這裡只是記住 game_id 與設定，
/// 首頁才能顯示「繼續對局」，重新進入用 /state 拉回盤面，不用再開新局、不扣體力。
class CurrentGame {
  final String gameId;
  final String level;
  final String humanColor;
  final int boardSize;
  final double komi;
  final int handicap;
  final int moves;
  final DateTime savedAt;

  const CurrentGame({
    required this.gameId,
    required this.level,
    required this.humanColor,
    required this.boardSize,
    required this.komi,
    required this.handicap,
    required this.moves,
    required this.savedAt,
  });

  factory CurrentGame.fromGame(GameState g, {required String level}) =>
      CurrentGame(
        gameId: g.gameId,
        level: level,
        humanColor: g.humanColor,
        boardSize: g.size,
        komi: g.komi,
        handicap: g.handicap,
        moves: g.moves,
        savedAt: DateTime.now(),
      );

  factory CurrentGame.fromJson(Map<String, dynamic> j) => CurrentGame(
    gameId: j['game_id'] as String,
    level: j['level'] as String,
    humanColor: j['human_color'] as String,
    boardSize: (j['board_size'] as num).toInt(),
    komi: (j['komi'] as num).toDouble(),
    handicap: (j['handicap'] as num?)?.toInt() ?? 0,
    moves: (j['moves'] as num?)?.toInt() ?? 0,
    savedAt: DateTime.parse(j['saved_at'] as String),
  );

  Map<String, dynamic> toJson() => {
    'game_id': gameId,
    'level': level,
    'human_color': humanColor,
    'board_size': boardSize,
    'komi': komi,
    'handicap': handicap,
    'moves': moves,
    'saved_at': savedAt.toIso8601String(),
  };
}

class CurrentGameStore {
  static const _fileName = 'current_game.json';

  /// 測試用：不走 path_provider。
  static Directory? directoryOverride;

  static Future<File> _file() async {
    final dir = directoryOverride ?? await getApplicationDocumentsDirectory();
    return File('${dir.path}/$_fileName');
  }

  static Future<CurrentGame?> load() async {
    try {
      final file = await _file();
      if (!file.existsSync()) return null;
      final decoded = jsonDecode(await file.readAsString());
      if (decoded is! Map<String, dynamic>) return null;
      return CurrentGame.fromJson(decoded);
    } catch (_) {
      return null; // 壞掉的檔案當作沒有
    }
  }

  static Future<void> save(CurrentGame game) async {
    try {
      final file = await _file();
      await file.writeAsString(jsonEncode(game.toJson()));
    } catch (_) {
      /* 存不進去頂多是首頁少一顆「繼續對局」，不影響下棋 */
    }
  }

  static Future<void> clear() async {
    try {
      final file = await _file();
      if (file.existsSync()) await file.delete();
    } catch (_) {}
  }
}

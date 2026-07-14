import 'dart:convert';

import 'package:http/http.dart' as http;

/// 引擎伺服器 (gozero/server.py)，跑在使用者家中 server，路由器把 8765 轉發到公網固定 IP。
/// ⚠️ 明文 HTTP（沒有網域就沒有正規 TLS 憑證）。裸 IP 沒法用 NSExceptionDomains 精準例外
/// （Apple 不保證對 IP 字串生效），Info.plist 用 NSAllowsArbitraryLoads 整個 app 放行明文。
/// 本機開發要直連時改回 'http://127.0.0.1:8765'。
const engineBase = 'http://59.125.215.63:8765';

class GameState {
  final String gameId;
  final List<int> board; // 0 空, 1 黑, 2 白（列優先，0 = 左上）
  final int size;
  final String toMove; // black | white
  final String humanColor;
  final int moves;
  final List<int> history;
  final int? lastMove;
  final int? aiMove;
  final List<int> legal;
  final double blackWinrate;
  final List<double> winrates; // 每手後的黑勝率（index 0 = 空盤）
  final int capturedBlack; // 被提掉的黑子數
  final int capturedWhite;
  final bool gameOver;
  final String? winner;
  final String? winReason; // score | resign
  final double? margin;
  final double komi;

  GameState.fromJson(Map<String, dynamic> j)
    : gameId = j['game_id'],
      board = List<int>.from(j['board']),
      size = j['size'],
      toMove = j['to_move'],
      humanColor = j['human_color'],
      moves = j['moves'],
      history = List<int>.from(j['history'] ?? const []),
      lastMove = j['last_move'],
      aiMove = j['ai_move'],
      legal = List<int>.from(j['legal']),
      blackWinrate = (j['black_winrate'] as num).toDouble(),
      winrates = [
        for (final w in (j['winrates'] ?? const [])) (w as num).toDouble(),
      ],
      capturedBlack = j['captures']['black'],
      capturedWhite = j['captures']['white'],
      gameOver = j['game_over'],
      winner = j['result']?['winner'],
      winReason = j['result']?['reason'],
      margin = (j['result']?['margin'] as num?)?.toDouble(),
      komi = (j['komi'] as num).toDouble();

  int get passAction => size * size;
}

class EngineInfo {
  final String model;
  final int iteration;
  EngineInfo(this.model, this.iteration);
}

class EngineApi {
  Future<Map<String, dynamic>> _post(
    String path,
    Map<String, dynamic> body,
  ) async {
    final r = await http
        .post(Uri.parse('$engineBase$path'), body: jsonEncode(body))
        .timeout(const Duration(seconds: 60));
    final j = jsonDecode(r.body) as Map<String, dynamic>;
    if (r.statusCode != 200) {
      throw EngineError(j['error'] ?? 'HTTP ${r.statusCode}');
    }
    return j;
  }

  Future<EngineInfo> health() async {
    final r = await http
        .get(Uri.parse('$engineBase/health'))
        .timeout(const Duration(seconds: 3));
    final j = jsonDecode(r.body);
    return EngineInfo(j['model'], j['iteration']);
  }

  Future<GameState> newGame({
    required String level,
    required String humanColor,
  }) async => GameState.fromJson(
    await _post('/new', {'level': level, 'human_color': humanColor}),
  );

  /// 唯讀狀態（timeout 後重新同步用）
  Future<GameState> state(String gameId) async {
    final r = await http
        .get(Uri.parse('$engineBase/state?game_id=$gameId'))
        .timeout(const Duration(seconds: 5));
    final j = jsonDecode(r.body) as Map<String, dynamic>;
    if (r.statusCode != 200) {
      throw EngineError(j['error'] ?? 'HTTP ${r.statusCode}');
    }
    return GameState.fromJson(j);
  }

  Future<GameState> move(String gameId, int action) async => GameState.fromJson(
    await _post('/move', {'game_id': gameId, 'action': action}),
  );

  Future<GameState> undo(String gameId) async =>
      GameState.fromJson(await _post('/undo', {'game_id': gameId}));

  Future<GameState> resign(String gameId) async =>
      GameState.fromJson(await _post('/resign', {'game_id': gameId}));
}

class EngineError implements Exception {
  final String message;
  EngineError(this.message);
  @override
  String toString() => message;
}

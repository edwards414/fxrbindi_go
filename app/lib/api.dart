import 'dart:convert';

import 'package:http/http.dart' as http;

/// 引擎伺服器 (gozero/server.py)。跑在家中主機，但不對外開 port——
/// 由同一台機器上的 cloudflared 建立 Cloudflare Tunnel 對外，
/// 所以公網位址是正規 TLS 的 https://go.fxrbindi.com（憑證由 Cloudflare 簽發）。
/// 家用 IP 不外露，IP 變動也不影響 app；ATS 因此不需要任何例外。
///
/// 本機開發直連引擎：
///   flutter run --dart-define=ENGINE_BASE=http://127.0.0.1:8765
/// （模擬器連 localhost 需要 Info.plist 的 NSAllowsLocalNetworking，已保留。）
const engineBase = String.fromEnvironment(
  'ENGINE_BASE',
  defaultValue: 'https://go.fxrbindi.com',
);

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
  final int handicap;

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
      komi = (j['komi'] as num).toDouble(),
      handicap = (j['handicap'] as num?)?.toInt() ?? 0;

  int get passAction => size * size;
}

class EngineInfo {
  final String model;
  final int iteration;
  EngineInfo(this.model, this.iteration);
}

class EngineApi {
  /// 明確標示自己是誰。Cloudflare 會依 User-Agent 判斷 bot——預設的
  /// `Dart/x.y (dart:io)` 目前放行，但那是別人的清單說了算；自報名號比較穩。
  static const _headers = {
    'Content-Type': 'application/json',
    'User-Agent': 'XuanShiGoZero/1.0 (+https://fxrbindi.com)',
  };

  Future<Map<String, dynamic>> _post(
    String path,
    Map<String, dynamic> body,
  ) async {
    final r = await http
        .post(
          Uri.parse('$engineBase$path'),
          headers: _headers,
          body: jsonEncode(body),
        )
        .timeout(const Duration(seconds: 60));
    final j = jsonDecode(r.body) as Map<String, dynamic>;
    if (r.statusCode != 200) {
      throw EngineError(j['error'] ?? 'HTTP ${r.statusCode}');
    }
    return j;
  }

  Future<EngineInfo> health() async {
    final r = await http
        .get(Uri.parse('$engineBase/health'), headers: _headers)
        // 冷啟動的 TLS 握手 + 跨海往返，3 秒太緊（審查員在美國）
        .timeout(const Duration(seconds: 10));
    final j = jsonDecode(r.body);
    return EngineInfo(j['model'], j['iteration']);
  }

  Future<GameState> newGame({
    required String level,
    required String humanColor,
    double komi = 7.5,
    int handicap = 0,
  }) async => GameState.fromJson(
    await _post('/new', {
      'level': level,
      'human_color': humanColor,
      'komi': komi,
      'handicap': handicap,
    }),
  );

  /// 唯讀狀態（timeout 後重新同步用）
  Future<GameState> state(String gameId) async {
    final r = await http
        .get(
          Uri.parse('$engineBase/state?game_id=$gameId'),
          headers: _headers,
        )
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

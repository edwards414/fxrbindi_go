import 'dart:async';
import 'dart:convert';
import 'dart:math';

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
  final int undoLimit; // 每局可悔棋次數（伺服器決定）
  final int undosLeft;
  final StaminaInfo? stamina; // 只有 /new 的回應會帶

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
      handicap = (j['handicap'] as num?)?.toInt() ?? 0,
      // 舊版伺服器沒有這兩個欄位：視為不限次
      undoLimit = (j['undo_limit'] as num?)?.toInt() ?? 1 << 30,
      undosLeft =
          (j['undos_left'] as num?)?.toInt() ??
          (j['undo_limit'] as num?)?.toInt() ??
          1 << 30,
      stamina = j['stamina'] is Map<String, dynamic>
          ? StaminaInfo.fromJson(j['stamina'] as Map<String, dynamic>)
          : null;

  int get passAction => size * size;
}

/// 伺服器回報的體力：開局扣 1 點，每 [regenSeconds] 回 1 點，上限 [max]。
/// 帶著取得時間，畫面倒數可以在本機推算，不必一直打伺服器。
class StaminaInfo {
  final int points;
  final int max;
  final int regenSeconds;
  final int? nextInSeconds; // 滿格時為 null
  final DateTime fetchedAt;

  StaminaInfo({
    required this.points,
    required this.max,
    required this.regenSeconds,
    required this.nextInSeconds,
    DateTime? fetchedAt,
  }) : fetchedAt = fetchedAt ?? DateTime.now();

  factory StaminaInfo.fromJson(Map<String, dynamic> j) => StaminaInfo(
    points: (j['points'] as num?)?.toInt() ?? 0,
    max: (j['max'] as num?)?.toInt() ?? 10,
    regenSeconds: (j['regen_seconds'] as num?)?.toInt() ?? 3600,
    nextInSeconds: (j['next_in_seconds'] as num?)?.toInt(),
  );

  /// 依取得後經過的時間推算目前點數（伺服器仍是真相，這只是畫面用）。
  int pointsAt(DateTime now) {
    final first = nextInSeconds;
    if (first == null) return max;
    final elapsed = now.difference(fetchedAt).inSeconds;
    if (elapsed < first) return points;
    final gained = 1 + (elapsed - first) ~/ regenSeconds;
    return (points + gained).clamp(0, max);
  }

  /// 距離下一點回復的時間；滿格為 null。
  Duration? nextInAt(DateTime now) {
    final first = nextInSeconds;
    if (first == null) return null;
    if (pointsAt(now) >= max) return null;
    final elapsed = now.difference(fetchedAt).inSeconds;
    if (elapsed < first) return Duration(seconds: first - elapsed);
    final rem = regenSeconds - (elapsed - first) % regenSeconds;
    return Duration(seconds: rem);
  }

  bool isEmptyAt(DateTime now) => pointsAt(now) <= 0;
}

class EngineInfo {
  final String model;
  final int iteration;
  final List<int> boardSizes;
  EngineInfo(this.model, this.iteration, this.boardSizes);
}

class QueueProgress {
  final String status; // queued | running | completed
  final int position; // 1-based while queued, 0 while running
  final int estimatedWaitSeconds;

  const QueueProgress({
    required this.status,
    required this.position,
    required this.estimatedWaitSeconds,
  });

  factory QueueProgress.fromJson(Map<String, dynamic> j) => QueueProgress(
    status: j['status'] as String? ?? 'queued',
    position: (j['queue_position'] as num?)?.toInt() ?? 0,
    estimatedWaitSeconds: (j['estimated_wait_seconds'] as num?)?.toInt() ?? 0,
  );
}

typedef QueueProgressCallback = void Function(QueueProgress progress);

class EngineApi {
  /// 明確標示自己是誰。Cloudflare 會依 User-Agent 判斷 bot——預設的
  /// `Dart/x.y (dart:io)` 目前放行，但那是別人的清單說了算；自報名號比較穩。
  static const _headers = {
    'Content-Type': 'application/json',
    'User-Agent': 'XuanShiGoZero/1.0 (+https://fxrbindi.com)',
  };
  static final _random = Random.secure();
  final http.Client _client;

  EngineApi({http.Client? client}) : _client = client ?? http.Client();

  void close() => _client.close();

  String _newRequestId() {
    final time = DateTime.now().microsecondsSinceEpoch.toRadixString(36);
    final random = List.generate(
      16,
      (_) => _random.nextInt(256).toRadixString(16).padLeft(2, '0'),
    ).join();
    return '$time-$random';
  }

  Map<String, dynamic> _decode(http.Response response) {
    try {
      return jsonDecode(response.body) as Map<String, dynamic>;
    } catch (_) {
      throw EngineError('引擎回應格式錯誤', response.statusCode);
    }
  }

  Future<Map<String, dynamic>> _post(
    String path,
    Map<String, dynamic> body, {
    QueueProgressCallback? onQueueProgress,
  }) async {
    final requestId = _newRequestId();
    final payload = Map<String, dynamic>.from(body)..['request_id'] = requestId;

    Future<http.Response> send() => _client
        .post(
          Uri.parse('$engineBase$path'),
          headers: {..._headers, 'Idempotency-Key': requestId},
          body: jsonEncode(payload),
        )
        .timeout(const Duration(seconds: 15));

    // 入列請求具有 idempotency key；第一次若在回程斷線，重送不會執行兩次。
    http.Response r;
    try {
      r = await send();
    } catch (_) {
      await Future<void>.delayed(const Duration(milliseconds: 500));
      r = await send();
    }
    final j = _decode(r);
    if (r.statusCode == 202) {
      return _waitForJob(j, onQueueProgress: onQueueProgress);
    }
    if (r.statusCode != 200) {
      throw EngineError(
        j['error'] ?? 'HTTP ${r.statusCode}',
        r.statusCode,
        j['stamina'] is Map<String, dynamic>
            ? StaminaInfo.fromJson(j['stamina'] as Map<String, dynamic>)
            : null,
      );
    }
    return j;
  }

  Future<Map<String, dynamic>> _waitForJob(
    Map<String, dynamic> initial, {
    QueueProgressCallback? onQueueProgress,
  }) async {
    var job = initial;
    final jobId = job['job_id'] as String?;
    if (jobId == null || jobId.isEmpty) {
      throw EngineError('引擎沒有回傳排隊編號');
    }
    final deadline = DateTime.now().add(const Duration(minutes: 10));
    var transientFailures = 0;

    while (true) {
      final status = job['status'] as String?;
      if (status == 'completed') {
        onQueueProgress?.call(QueueProgress.fromJson(job));
        final result = job['result'];
        if (result is! Map<String, dynamic>) {
          throw EngineError('引擎任務缺少結果');
        }
        return result;
      }
      if (status == 'failed') {
        throw EngineError(
          job['error'] as String? ?? '引擎任務失敗',
          (job['http_status'] as num?)?.toInt(),
        );
      }
      if (status != 'queued' && status != 'running') {
        throw EngineError('未知的引擎任務狀態');
      }
      onQueueProgress?.call(QueueProgress.fromJson(job));
      if (DateTime.now().isAfter(deadline)) {
        throw EngineError('排隊等候逾時，請稍後重試');
      }

      final position = (job['queue_position'] as num?)?.toInt() ?? 0;
      final delay = position > 8
          ? const Duration(seconds: 2)
          : position > 2
          ? const Duration(seconds: 1)
          : const Duration(milliseconds: 500);
      await Future<void>.delayed(delay);

      try {
        final r = await _client
            .get(Uri.parse('$engineBase/jobs/$jobId'), headers: _headers)
            .timeout(const Duration(seconds: 10));
        final next = _decode(r);
        if (r.statusCode != 200) {
          throw EngineError(
            next['error'] ?? 'HTTP ${r.statusCode}',
            r.statusCode,
          );
        }
        job = next;
        transientFailures = 0;
      } on EngineError {
        rethrow;
      } catch (_) {
        transientFailures++;
        if (transientFailures >= 4) {
          throw EngineError('排隊狀態暫時無法取得，請確認網路連線');
        }
      }
    }
  }

  Future<EngineInfo> health() async {
    final r = await _client
        .get(Uri.parse('$engineBase/health'), headers: _headers)
        // 冷啟動的 TLS 握手 + 跨海往返，3 秒太緊（審查員在美國）
        .timeout(const Duration(seconds: 10));
    final j = jsonDecode(r.body);
    final sizes = j['board_sizes'] is List
        ? List<int>.from(j['board_sizes'])
        : <int>[(j['board_size'] as num?)?.toInt() ?? 19];
    return EngineInfo(j['model'], j['iteration'], sizes);
  }

  Future<GameState> newGame({
    required String level,
    required String humanColor,
    int boardSize = 19,
    double komi = 7.5,
    int handicap = 0,
    String? playerId,
    QueueProgressCallback? onQueueProgress,
  }) async => GameState.fromJson(
    await _post('/new', {
      'level': level,
      'human_color': humanColor,
      'board_size': boardSize,
      'komi': komi,
      'handicap': handicap,
      'player_id': ?playerId,
    }, onQueueProgress: onQueueProgress),
  );

  /// 目前體力（不扣點）。舊版伺服器沒有這個端點時回 null。
  Future<StaminaInfo?> stamina(String playerId) async {
    final r = await _client
        .get(_url('/stamina', {'player_id': playerId}), headers: _headers)
        .timeout(const Duration(seconds: 10));
    if (r.statusCode == 404) return null;
    final j = _decode(r);
    if (r.statusCode != 200) {
      throw EngineError(j['error'] ?? 'HTTP ${r.statusCode}', r.statusCode);
    }
    return StaminaInfo.fromJson(j);
  }

  /// 組 GET 網址：query 一律經過編碼，避免字串拼接把參數值帶進路徑或其他參數
  static Uri _url(String path, [Map<String, String>? query]) =>
      Uri.parse(engineBase).replace(path: path, queryParameters: query);

  /// 唯讀狀態（timeout 後重新同步用）
  Future<GameState> state(String gameId) async {
    final r = await _client
        .get(_url('/state', {'game_id': gameId}), headers: _headers)
        .timeout(const Duration(seconds: 5));
    final j = jsonDecode(r.body) as Map<String, dynamic>;
    if (r.statusCode != 200) {
      throw EngineError(j['error'] ?? 'HTTP ${r.statusCode}');
    }
    return GameState.fromJson(j);
  }

  Future<GameState> move(
    String gameId,
    int action, {
    required int expectedMoves,
    QueueProgressCallback? onQueueProgress,
  }) async => GameState.fromJson(
    await _post('/move', {
      'game_id': gameId,
      'action': action,
      'expected_moves': expectedMoves,
    }, onQueueProgress: onQueueProgress),
  );

  Future<GameState> undo(
    String gameId, {
    required int expectedMoves,
    QueueProgressCallback? onQueueProgress,
  }) async => GameState.fromJson(
    await _post('/undo', {
      'game_id': gameId,
      'expected_moves': expectedMoves,
    }, onQueueProgress: onQueueProgress),
  );

  Future<GameState> resign(String gameId) async =>
      GameState.fromJson(await _post('/resign', {'game_id': gameId}));
}

class EngineError implements Exception {
  final String message;
  final int? statusCode;
  final StaminaInfo? stamina; // 429 體力不足時伺服器附上的倒數
  EngineError(this.message, [this.statusCode, this.stamina]);
  bool get staminaExhausted => statusCode == 429;
  @override
  String toString() => message;
}

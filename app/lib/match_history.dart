import 'dart:convert';
import 'dart:io' show File;

import 'package:path_provider/path_provider.dart';

import 'api.dart';

class MatchRecord {
  final String id;
  final DateTime playedAt;
  final String level;
  final String humanColor;
  final int boardSize;
  final List<int> actions;
  final List<double> winrates;
  final String? winner;
  final String? reason;
  final double? margin;
  final double komi;

  const MatchRecord({
    required this.id,
    required this.playedAt,
    required this.level,
    required this.humanColor,
    required this.boardSize,
    required this.actions,
    required this.winrates,
    required this.winner,
    required this.reason,
    required this.margin,
    required this.komi,
  });

  factory MatchRecord.fromGame(GameState game, {required String level}) =>
      MatchRecord(
        id: game.gameId,
        playedAt: DateTime.now(),
        level: level,
        humanColor: game.humanColor,
        boardSize: game.size,
        actions: List<int>.from(game.history),
        winrates: List<double>.from(game.winrates),
        winner: game.winner,
        reason: game.winReason,
        margin: game.margin,
        komi: game.komi,
      );

  factory MatchRecord.fromJson(Map<String, dynamic> json) => MatchRecord(
    id: json['id'] as String,
    playedAt: DateTime.parse(json['played_at'] as String),
    level: json['level'] as String,
    humanColor: json['human_color'] as String,
    boardSize: (json['board_size'] as num).toInt(),
    actions: [
      for (final action in json['actions'] as List<dynamic>)
        (action as num).toInt(),
    ],
    winrates: [
      for (final winrate in json['winrates'] as List<dynamic>)
        (winrate as num).toDouble(),
    ],
    winner: json['winner'] as String?,
    reason: json['reason'] as String?,
    margin: (json['margin'] as num?)?.toDouble(),
    komi: (json['komi'] as num).toDouble(),
  );

  Map<String, dynamic> toJson() => {
    'id': id,
    'played_at': playedAt.toIso8601String(),
    'level': level,
    'human_color': humanColor,
    'board_size': boardSize,
    'actions': actions,
    'winrates': winrates,
    'winner': winner,
    'reason': reason,
    'margin': margin,
    'komi': komi,
  };
}

class MatchHistoryStore {
  static const _fileName = 'match_history.json';
  static const _maxRecords = 80;

  static Future<File> _file() async {
    final docs = await getApplicationDocumentsDirectory();
    return File('${docs.path}/$_fileName');
  }

  static Future<List<MatchRecord>> load() async {
    final file = await _file();
    if (!file.existsSync()) return [];

    try {
      final decoded = jsonDecode(await file.readAsString());
      if (decoded is! List<dynamic>) return [];

      final records = <MatchRecord>[];
      for (final row in decoded) {
        if (row is Map<String, dynamic>) {
          records.add(MatchRecord.fromJson(row));
        }
      }
      records.sort((a, b) => b.playedAt.compareTo(a.playedAt));
      return records;
    } on FormatException {
      return [];
    } on TypeError {
      return [];
    }
  }

  static Future<void> save(MatchRecord record) async {
    final records = await load();
    records.removeWhere((r) => r.id == record.id);
    records.insert(0, record);
    final kept = records.take(_maxRecords).toList();
    await _write(kept);
  }

  static Future<void> delete(String id) async {
    final records = await load();
    records.removeWhere((r) => r.id == id);
    await _write(records);
  }

  static Future<void> _write(List<MatchRecord> records) async {
    final file = await _file();
    const encoder = JsonEncoder.withIndent('  ');
    await file.writeAsString(
      encoder.convert([for (final record in records) record.toJson()]),
    );
  }
}

class ReviewFrame {
  final int moveNumber;
  final int? action;
  final int? stoneColor;
  final List<int> board;
  final double? blackWinrate;

  const ReviewFrame({
    required this.moveNumber,
    required this.action,
    required this.stoneColor,
    required this.board,
    required this.blackWinrate,
  });
}

class MatchReplay {
  static List<ReviewFrame> frames(MatchRecord record) {
    final board = List<int>.filled(record.boardSize * record.boardSize, 0);
    final frames = <ReviewFrame>[
      ReviewFrame(
        moveNumber: 0,
        action: null,
        stoneColor: null,
        board: List<int>.from(board),
        blackWinrate: record.winrates.isEmpty ? null : record.winrates.first,
      ),
    ];

    for (var ply = 0; ply < record.actions.length; ply++) {
      final action = record.actions[ply];
      final color = ply.isEven ? 1 : 2;
      if (action >= 0 && action < board.length) {
        _play(board, record.boardSize, action, color);
      }
      frames.add(
        ReviewFrame(
          moveNumber: ply + 1,
          action: action,
          stoneColor: color,
          board: List<int>.from(board),
          blackWinrate: ply + 1 < record.winrates.length
              ? record.winrates[ply + 1]
              : null,
        ),
      );
    }
    return frames;
  }

  static void _play(List<int> board, int size, int action, int color) {
    final opponent = color == 1 ? 2 : 1;
    board[action] = color;

    for (final point in _adjacent(action, size)) {
      if (board[point] == opponent && !_hasLiberty(board, size, point)) {
        _removeGroup(board, size, point);
      }
    }

    if (!_hasLiberty(board, size, action)) {
      _removeGroup(board, size, action);
    }
  }

  static Iterable<int> _adjacent(int index, int size) sync* {
    final row = index ~/ size;
    final col = index % size;
    if (row > 0) yield index - size;
    if (row < size - 1) yield index + size;
    if (col > 0) yield index - 1;
    if (col < size - 1) yield index + 1;
  }

  static bool _hasLiberty(List<int> board, int size, int start) {
    final color = board[start];
    final seen = <int>{start};
    final stack = [start];

    while (stack.isNotEmpty) {
      final point = stack.removeLast();
      for (final next in _adjacent(point, size)) {
        if (board[next] == 0) return true;
        if (board[next] == color && seen.add(next)) {
          stack.add(next);
        }
      }
    }
    return false;
  }

  static void _removeGroup(List<int> board, int size, int start) {
    final color = board[start];
    final stack = [start];
    board[start] = 0;

    while (stack.isNotEmpty) {
      final point = stack.removeLast();
      for (final next in _adjacent(point, size)) {
        if (board[next] == color) {
          board[next] = 0;
          stack.add(next);
        }
      }
    }
  }
}

String matchLevelLabel(String level) => switch (level) {
  'easy' => '直覺',
  'normal' => '均衡',
  'strong' => '深思',
  _ => level,
};

String colorLabel(String color) => switch (color) {
  'black' => '黑',
  'white' => '白',
  _ => color,
};

String formatMatchDate(DateTime date) {
  String two(int value) => value.toString().padLeft(2, '0');
  return '${date.year}/${two(date.month)}/${two(date.day)} '
      '${two(date.hour)}:${two(date.minute)}';
}

String resultLabel(MatchRecord record) {
  final winner = record.winner;
  if (winner == null) return '未完局';
  if (winner == 'draw') return '和棋';

  final winnerText = colorLabel(winner);
  if (record.reason == 'resign') return '$winnerText中盤勝';
  if (record.reason == 'rule') return '$winnerText勝（犯規）';
  final margin = record.margin;
  return margin == null
      ? '$winnerText勝'
      : '$winnerText勝 ${margin.toStringAsFixed(1)} 目';
}

String humanResultLabel(MatchRecord record) {
  if (record.winner == 'draw') return '和棋';
  if (record.winner == record.humanColor) return '你勝';
  if (record.winner == null) return '未完局';
  return '玄石勝';
}

String moveLabel(int? action, int size) {
  if (action == null) return '空盤';
  if (action < 0 || action >= size * size) return '虛手';

  const cols = 'ABCDEFGHJKLMNOPQRST';
  final row = action ~/ size;
  final col = action % size;
  return '${cols[col]}${size - row}';
}

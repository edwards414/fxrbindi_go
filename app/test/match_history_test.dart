import 'package:flutter_test/flutter_test.dart';
import 'package:gozero_go/match_history.dart';

void main() {
  test('replay frames rebuild captures from actions', () {
    final record = MatchRecord(
      id: 'capture',
      playedAt: DateTime(2026, 7, 8, 12),
      level: 'normal',
      humanColor: 'black',
      boardSize: 5,
      actions: const [7, 12, 11, 0, 13, 24, 17],
      winrates: const [0.5, 0.51, 0.48, 0.52, 0.49, 0.53, 0.5, 0.55],
      winner: 'black',
      reason: 'score',
      margin: 3.5,
      komi: 7.5,
    );

    final frames = MatchReplay.frames(record);

    expect(frames, hasLength(8));
    expect(frames[2].board[12], 2);
    expect(frames.last.board[12], 0);
    expect(frames.last.board[7], 1);
    expect(frames.last.board[11], 1);
    expect(frames.last.board[13], 1);
    expect(frames.last.board[17], 1);
  });

  test('result labels summarize human and board result', () {
    final record = MatchRecord(
      id: 'result',
      playedAt: DateTime(2026, 7, 8, 12),
      level: 'strong',
      humanColor: 'white',
      boardSize: 9,
      actions: const [40, 30],
      winrates: const [0.5, 0.6, 0.45],
      winner: 'white',
      reason: 'resign',
      margin: null,
      komi: 7.5,
    );

    expect(humanResultLabel(record), '你勝');
    expect(resultLabel(record), '白中盤勝');
    expect(matchLevelLabel(record.level), '深思');
  });
}

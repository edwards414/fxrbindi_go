import 'dart:convert';

import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';

import 'package:gozero_go/api.dart';

Map<String, dynamic> gameJson({int moves = 2, int size = 19}) => {
  'game_id': 'game123',
  'board': List<int>.filled(size * size, 0),
  'size': size,
  'to_move': 'black',
  'human_color': 'black',
  'moves': moves,
  'history': List<int>.generate(moves, (i) => i),
  'last_move': moves == 0 ? null : moves - 1,
  'ai_move': moves == 0 ? null : moves - 1,
  'legal': List<int>.filled(size * size + 1, 1),
  'black_winrate': 0.5,
  'winrates': List<double>.filled(moves + 1, 0.5),
  'captures': {'black': 0, 'white': 0},
  'game_over': false,
  'result': null,
  'komi': 7.5,
  'handicap': 0,
};

void main() {
  test('new game sends the selected 9x9 board size', () async {
    final client = MockClient((request) async {
      expect(request.method, 'POST');
      expect(request.url.path, '/new');
      final body = jsonDecode(request.body) as Map<String, dynamic>;
      expect(body['board_size'], 9);
      expect(body['level'], 'normal');
      expect(body['human_color'], 'black');
      return http.Response(jsonEncode(gameJson(moves: 0, size: 9)), 200);
    });
    final api = EngineApi(client: client);

    final game = await api.newGame(
      level: 'normal',
      humanColor: 'black',
      boardSize: 9,
    );

    expect(game.size, 9);
    expect(game.board, hasLength(81));
    expect(game.legal, hasLength(82));
    api.close();
  });

  test('202 job is polled and reports queue progress', () async {
    var pollCount = 0;
    final jobId = List.filled(32, 'a').join();
    final client = MockClient((request) async {
      if (request.method == 'POST') {
        final body = jsonDecode(request.body) as Map<String, dynamic>;
        expect(body['expected_moves'], 0);
        expect(body['request_id'], isNotEmpty);
        expect(request.headers['idempotency-key'], body['request_id']);
        return http.Response(
          jsonEncode({
            'job_id': jobId,
            'operation': 'move',
            'status': 'queued',
            'queue_position': 2,
            'estimated_wait_seconds': 4,
          }),
          202,
        );
      }
      pollCount++;
      expect(request.url.path, '/jobs/$jobId');
      return http.Response(
        jsonEncode({
          'job_id': jobId,
          'operation': 'move',
          'status': 'completed',
          'result': gameJson(),
        }),
        200,
      );
    });
    final api = EngineApi(client: client);
    final progress = <QueueProgress>[];

    final game = await api.move(
      'game123',
      10,
      expectedMoves: 0,
      onQueueProgress: progress.add,
    );

    expect(game.gameId, 'game123');
    expect(game.size, 19);
    expect(game.board, hasLength(361));
    expect(game.legal, hasLength(362));
    expect(pollCount, 1);
    expect(progress.first.status, 'queued');
    expect(progress.first.position, 2);
    expect(progress.first.estimatedWaitSeconds, 4);
    expect(progress.last.status, 'completed');
    api.close();
  });

  test('failed background job preserves the API status', () async {
    final jobId = List.filled(32, 'b').join();
    final client = MockClient(
      (_) async => http.Response(
        jsonEncode({
          'job_id': jobId,
          'operation': 'move',
          'status': 'failed',
          'error': 'game state changed; refresh and try again',
          'http_status': 409,
        }),
        202,
      ),
    );
    final api = EngineApi(client: client);

    try {
      await api.move('game123', 10, expectedMoves: 0);
      fail('expected EngineError');
    } on EngineError catch (error) {
      expect(error.statusCode, 409);
      expect(error.message, contains('game state changed'));
    } finally {
      api.close();
    }
  });

  test('legacy synchronous server response remains supported', () async {
    final client = MockClient(
      (_) async => http.Response(jsonEncode(gameJson(moves: 4)), 200),
    );
    final api = EngineApi(client: client);

    final game = await api.undo('game123', expectedMoves: 6);

    expect(game.moves, 4);
    api.close();
  });
}

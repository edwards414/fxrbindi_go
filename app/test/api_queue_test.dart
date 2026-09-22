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
    // 舊版伺服器沒有悔棋上限欄位：視為不限次
    expect(game.undosLeft, greaterThan(99));
    api.close();
  });

  test('new game sends player_id and parses stamina and undo budget', () async {
    final client = MockClient((request) async {
      final body = jsonDecode(request.body) as Map<String, dynamic>;
      expect(body['player_id'], 'abcdef0123456789');
      return http.Response(
        jsonEncode({
          ...gameJson(moves: 0),
          'undo_limit': 3,
          'undos_used': 0,
          'undos_left': 3,
          'stamina': {
            'points': 23,
            'max': 24,
            'regen_seconds': 3600,
            'next_in_seconds': 3600,
          },
        }),
        200,
      );
    });
    final api = EngineApi(client: client);

    final game = await api.newGame(
      level: 'normal',
      humanColor: 'black',
      playerId: 'abcdef0123456789',
    );

    expect(game.undoLimit, 3);
    expect(game.undosLeft, 3);
    final s = game.stamina!;
    expect(s.points, 23);
    expect(s.pointsAt(s.fetchedAt), 23);
    expect(s.nextInAt(s.fetchedAt), const Duration(seconds: 3600));
    // 本機推算：2 小時 10 分後回了 2 點，距下一點 50 分鐘
    final later = s.fetchedAt.add(const Duration(hours: 2, minutes: 10));
    expect(s.pointsAt(later), 24);
    expect(s.nextInAt(later), isNull);
    api.close();
  });

  test('stamina regen estimate keeps the carry-over', () {
    final s = StaminaInfo(
      points: 5,
      max: 24,
      regenSeconds: 3600,
      nextInSeconds: 600,
      fetchedAt: DateTime(2026, 1, 1),
    );
    // 10 分鐘後 +1、再 60 分鐘 +1；75 分鐘時共 7 點，距下一點 55 分鐘
    final later = s.fetchedAt.add(const Duration(minutes: 75));
    expect(s.pointsAt(later), 7);
    expect(s.nextInAt(later), const Duration(minutes: 55));
    expect(s.isEmptyAt(later), isFalse);
  });

  test('429 from /new surfaces stamina countdown', () async {
    final client = MockClient(
      (_) async => http.Response(
        jsonEncode({
          'error': 'stamina exhausted',
          'stamina': {
            'points': 0,
            'max': 24,
            'regen_seconds': 3600,
            'next_in_seconds': 1234,
          },
        }),
        429,
      ),
    );
    final api = EngineApi(client: client);

    try {
      await api.newGame(level: 'easy', humanColor: 'black', playerId: 'p' * 16);
      fail('expected EngineError');
    } on EngineError catch (error) {
      expect(error.staminaExhausted, isTrue);
      expect(error.stamina!.points, 0);
      expect(error.stamina!.nextInSeconds, 1234);
    } finally {
      api.close();
    }
  });

  test('GET /stamina returns null on a server without the endpoint', () async {
    final client = MockClient(
      (_) async => http.Response(jsonEncode({'error': 'not found'}), 404),
    );
    final api = EngineApi(client: client);
    expect(await api.stamina('p' * 16), isNull);
    api.close();
  });
}

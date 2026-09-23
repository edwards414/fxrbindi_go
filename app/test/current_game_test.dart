import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:gozero_go/current_game.dart';

void main() {
  late Directory dir;
  setUp(() {
    dir = Directory.systemTemp.createTempSync('current_game');
    CurrentGameStore.directoryOverride = dir;
  });
  tearDown(() => dir.deleteSync(recursive: true));

  test('記住、讀回、清除進行中的對局', () async {
    expect(await CurrentGameStore.load(), isNull);
    await CurrentGameStore.save(
      CurrentGame(
        gameId: 'abc123def456',
        level: 'normal',
        humanColor: 'white',
        boardSize: 9,
        komi: 5.5,
        handicap: 2,
        moves: 17,
        savedAt: DateTime(2026, 9, 23, 12),
      ),
    );
    final saved = await CurrentGameStore.load();
    expect(saved, isNotNull);
    expect(saved!.gameId, 'abc123def456');
    expect(saved.boardSize, 9);
    expect(saved.komi, 5.5);
    expect(saved.handicap, 2);
    expect(saved.moves, 17);
    await CurrentGameStore.clear();
    expect(await CurrentGameStore.load(), isNull);
  });

  test('壞掉的檔案當作沒有', () async {
    File('${dir.path}/current_game.json').writeAsStringSync('{not json');
    expect(await CurrentGameStore.load(), isNull);
  });
}

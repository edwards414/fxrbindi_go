import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'api.dart';
import 'board_painter.dart';
import 'main.dart';
import 'match_history.dart';
import 'review_page.dart';
import 'winrate_curve.dart';

class GamePage extends StatefulWidget {
  final String level;
  final String humanColor;
  final double komi;
  final int handicap;
  final bool autoDemo; // 展示模式：自動下幾手（僅 autodemo 鉤子使用）
  const GamePage({
    super.key,
    required this.level,
    required this.humanColor,
    this.komi = 7.5,
    this.handicap = 0,
    this.autoDemo = false,
  });

  @override
  State<GamePage> createState() => GamePageState();
}

class GamePageState extends State<GamePage> with TickerProviderStateMixin {
  final api = EngineApi();
  GameState? game;
  bool busy = true; // 等待引擎中
  int? pending; // 樂觀渲染：已送出、等 AI 回手的那手（81 = 虛手）
  String? error;
  final _finishedRecords = <String, MatchRecord>{};

  // 落子進場縮放 / 提子淡出，共用同一個 220ms 進度
  late final AnimationController _placeAnim = AnimationController(
    vsync: this,
    duration: const Duration(milliseconds: 220),
  );
  Set<int> _entering = {};
  Map<int, int> _exiting = {};

  // 「思考中」呼吸動畫——等待引擎回手時用，取代泛用轉圈圈
  late final AnimationController _thinkAnim = AnimationController(
    vsync: this,
    duration: const Duration(milliseconds: 900),
  )..repeat(reverse: true);

  static const levelNames = {'easy': '直覺', 'normal': '均衡', 'strong': '深思'};

  @override
  void initState() {
    super.initState();
    _newGame();
  }

  @override
  void dispose() {
    _placeAnim.dispose();
    _thinkAnim.dispose();
    super.dispose();
  }

  /// 目前畫面上實際顯示的盤面（含尚未確認的樂觀落子），供動效 diff 用。
  List<int>? _displayedBoard() {
    final g = game;
    if (g == null) return null;
    if (pending != null && pending! < g.size * g.size) {
      return _resolveCaptures(g, pending!);
    }
    return List<int>.from(g.board);
  }

  void _animateBoardDiff(List<int> prev, List<int> next) {
    final entering = <int>{};
    final exiting = <int, int>{};
    for (var i = 0; i < next.length; i++) {
      if (prev[i] == 0 && next[i] != 0) entering.add(i);
      if (prev[i] != 0 && next[i] == 0) exiting[i] = prev[i];
    }
    if (entering.isEmpty && exiting.isEmpty) return;
    setState(() {
      _entering = entering;
      _exiting = exiting;
    });
    _placeAnim.forward(from: 0).whenComplete(() {
      if (mounted) setState(() => _exiting = {});
    });
  }

  Future<void> _run(
    Future<GameState> Function() op, {
    bool animateDiff = false,
  }) async {
    final prevBoard = animateDiff ? _displayedBoard() : null;
    setState(() {
      busy = true;
      error = null;
    });
    try {
      final g = await op();
      if (!mounted) return;
      setState(() => game = g);
      if (prevBoard != null && prevBoard.length == g.board.length) {
        _animateBoardDiff(prevBoard, g.board);
      }
      if (g.gameOver) {
        HapticFeedback.mediumImpact();
        final record = await _recordFinishedGame(g);
        if (mounted) _showResult(g, record);
      } else if (g.aiMove != null) {
        HapticFeedback.selectionClick();
      }
      // 展示鉤子：自動下幾手（黑走 天元附近 → 引擎回應），供端到端驗證
      if (widget.autoDemo && g.moves < 6 && !g.gameOver) {
        Future.delayed(const Duration(milliseconds: 800), () {
          if (!mounted || !humanTurn) return;
          final prefs = [30, 50, 24, 56, 40];
          final a = prefs.firstWhere(
            (i) => game!.legal[i] == 1,
            orElse: () => game!.legal.indexOf(1),
          );
          _tap(a);
        });
      }
    } on EngineError catch (e) {
      if (!mounted) return;
      setState(() => error = e.message);
      await _resync();
    } catch (_) {
      if (!mounted) return;
      setState(() => error = '無法連線引擎伺服器，請確認網路連線後重試');
      await _resync();
    } finally {
      if (mounted) {
        setState(() {
          busy = false;
          pending = null;
        });
      }
    }
  }

  void _newGame() => _run(
    () => api.newGame(
      level: widget.level,
      humanColor: widget.humanColor,
      komi: widget.komi,
      handicap: widget.handicap,
    ),
  );

  /// 錯誤（如 timeout 後伺服器已落子）後向伺服器拉回真實狀態
  Future<void> _resync() async {
    final gid = game?.gameId;
    if (gid == null) return;
    try {
      final g = await api.state(gid);
      if (mounted) setState(() => game = g);
    } catch (_) {
      /* 連線仍斷；保留原錯誤訊息 */
    }
  }

  bool get humanTurn =>
      game != null &&
      !game!.gameOver &&
      game!.toMove == game!.humanColor &&
      !busy;

  void _tap(int action) {
    if (!humanTurn || game!.legal[action] == 0) return;
    HapticFeedback.lightImpact();
    // 立刻把自己的子畫上（樂觀渲染），AI 回手到達後以伺服器狀態為準
    setState(() => pending = action);
    _run(() => api.move(game!.gameId, action), animateDiff: true);
  }

  /// 等待回應期間，畫面上顯示的手番（送出落子後立即輪到 AI）
  String _displayToMove(GameState g) => pending != null
      ? (g.humanColor == 'black' ? 'white' : 'black')
      : g.toMove;

  Future<MatchRecord> _recordFinishedGame(GameState g) async {
    final existing = _finishedRecords[g.gameId];
    if (existing != null) return existing;

    final record = MatchRecord.fromGame(g, level: widget.level);
    _finishedRecords[g.gameId] = record;
    try {
      await MatchHistoryStore.save(record);
    } catch (e) {
      if (mounted) setState(() => error = '對戰已結束，但紀錄儲存失敗：$e');
    }
    return record;
  }

  void _showResult(GameState g, MatchRecord record) {
    final human = g.humanColor;
    final won = g.winner == human;
    final colorName = g.winner == 'black' ? '黑' : '白';
    final title = g.winner == 'draw'
        ? '和棋'
        : won
        ? '執${human == 'black' ? '黑' : '白'}勝'
        : '$colorName勝';
    final detail = g.winReason == 'resign'
        ? '$colorName中盤勝（對方認輸）'
        : g.winReason == 'rule'
        ? '$colorName勝（對方犯規：全同型禁著）'
        : g.winner == 'draw'
        ? '雙方同目'
        : '$colorName勝 ${g.margin!.toStringAsFixed(1)} 目（Tromp-Taylor，貼目 ${g.komi}）';
    showDialog(
      context: context,
      builder: (_) => AlertDialog(
        backgroundColor: Sumi.panel,
        title: Text(
          title,
          style: TextStyle(
            color: won ? Sumi.seal : Sumi.paper,
            fontSize: 30,
            fontWeight: FontWeight.w700,
          ),
        ),
        content: Text(
          detail,
          style: const TextStyle(color: Sumi.paperDim, fontSize: 16),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('檢視棋盤', style: TextStyle(color: Sumi.paperDim)),
          ),
          TextButton(
            onPressed: () {
              Navigator.pop(context);
              Navigator.push(
                context,
                MaterialPageRoute(builder: (_) => ReviewPage(record: record)),
              );
            },
            child: const Text('回顧此局', style: TextStyle(color: Sumi.paperDim)),
          ),
          FilledButton(
            style: FilledButton.styleFrom(backgroundColor: Sumi.seal),
            onPressed: () {
              Navigator.pop(context);
              _newGame();
            },
            child: const Text('再來一局'),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final g = game;
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Sumi.bg,
        title: Text(
          '對弈 · ${levelNames[widget.level]}'
          '${widget.handicap > 0 ? ' · 讓${widget.handicap}子' : ''}',
          style: const TextStyle(fontSize: 20, fontWeight: FontWeight.w600),
        ),
        actions: [
          // 任何請求進行中（悔棋/認輸/新局）都以此提示忙碌
          if (busy)
            const Padding(
              padding: EdgeInsets.only(right: 16),
              child: Center(
                child: SizedBox(
                  width: 16,
                  height: 16,
                  child: CircularProgressIndicator(
                    strokeWidth: 2,
                    color: Sumi.paperDim,
                  ),
                ),
              ),
            )
          else
            IconButton(
              tooltip: '新對局',
              icon: const Icon(Icons.refresh, color: Sumi.paperDim),
              onPressed: _newGame,
            ),
        ],
      ),
      body: g == null
          ? Center(
              child: error != null
                  ? _errorBox()
                  : const CircularProgressIndicator(color: Sumi.seal),
            )
          : Column(
              children: [
                _playerBar(g),
                _winratePanel(g),
                const Spacer(),
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 12),
                  child: AspectRatio(
                    aspectRatio: 1,
                    child: LayoutBuilder(
                      builder: (context, box) => GestureDetector(
                        onTapUp: (d) {
                          final w = box.maxWidth;
                          final margin = w * 0.085;
                          final grid = (w - 2 * margin) / (g.size - 1);
                          final c = ((d.localPosition.dx - margin) / grid)
                              .round();
                          final r = ((d.localPosition.dy - margin) / grid)
                              .round();
                          if (r >= 0 && r < g.size && c >= 0 && c < g.size) {
                            _tap(r * g.size + c);
                          }
                        },
                        child: Stack(
                          children: [
                            RepaintBoundary(
                              child: CustomPaint(
                                painter: _boardPainter(g),
                                size: Size.square(box.maxWidth),
                              ),
                            ),
                            // VoiceOver：81 個交叉點各自可讀（座標＋黑/白/空），
                            // 合法且輪到你時可 double-tap 直接下子。視覺上完全不可見。
                            ..._boardSemantics(g, box.maxWidth),
                          ],
                        ),
                      ),
                    ),
                  ),
                ),
                if (error != null)
                  Padding(
                    padding: const EdgeInsets.only(top: 8),
                    child: _errorBox(),
                  ),
                const Spacer(),
                _controls(g),
                const SizedBox(height: 24),
              ],
            ),
    );
  }

  /// 含樂觀渲染：pending 的那手先畫上（虛手 81 不畫子），
  /// 並就地結算提子，避免等待期間短暫顯示無氣的死子。
  BoardPainter _boardPainter(GameState g) {
    var board = g.board;
    var last = g.lastMove;
    if (pending != null && pending! < g.size * g.size) {
      board = _resolveCaptures(g, pending!);
      last = pending;
    }
    return BoardPainter(
      size: g.size,
      board: board,
      lastMove: last,
      entering: _entering,
      exiting: _exiting,
      anim: _placeAnim,
    );
  }

  /// 每個交叉點一個透明的 Semantics 節點：VoiceOver 可逐格讀出座標＋黑/白/空，
  /// 合法且輪到你時可 double-tap 下子。棋盤本身是 CustomPaint，沒有這層的話
  /// VoiceOver 使用者完全碰不到這個 app 唯一的核心互動。
  List<Widget> _boardSemantics(GameState g, double w) {
    final margin = w * 0.085;
    final grid = (w - 2 * margin) / (g.size - 1);
    return [
      for (var i = 0; i < g.size * g.size; i++)
        Positioned(
          left: margin + (i % g.size) * grid - grid / 2,
          top: margin + (i ~/ g.size) * grid - grid / 2,
          width: grid,
          height: grid,
          child: Semantics(
            label:
                '${moveLabel(i, g.size)}，'
                '${g.board[i] == 1
                    ? '黑子'
                    : g.board[i] == 2
                    ? '白子'
                    : '空點'}',
            button: humanTurn && g.legal[i] == 1,
            enabled: humanTurn && g.legal[i] == 1,
            onTap: humanTurn && g.legal[i] == 1 ? () => _tap(i) : null,
            child: const SizedBox.expand(),
          ),
        ),
    ];
  }

  List<int> _resolveCaptures(GameState g, int at) {
    final n = g.size;
    final b = List<int>.from(g.board);
    final me = g.humanColor == 'black' ? 1 : 2;
    b[at] = me;

    Iterable<int> adj(int i) sync* {
      final r = i ~/ n, c = i % n;
      if (r > 0) yield i - n;
      if (r < n - 1) yield i + n;
      if (c > 0) yield i - 1;
      if (c < n - 1) yield i + 1;
    }

    // 回傳整串同色棋；無氣時清掉並回報 true
    bool removeIfDead(int start) {
      final color = b[start];
      final group = <int>{start};
      final stack = [start];
      while (stack.isNotEmpty) {
        for (final j in adj(stack.removeLast())) {
          if (b[j] == 0) return false; // 有氣
          if (b[j] == color && group.add(j)) stack.add(j);
        }
      }
      for (final j in group) {
        b[j] = 0;
      }
      return true;
    }

    for (final j in adj(at)) {
      if (b[j] != 0 && b[j] != me) removeIfDead(j);
    }
    removeIfDead(at); // 自填最後一氣的極端情況（伺服器合法手才會送到這）
    return b;
  }

  Widget _errorBox() => Container(
    padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
    decoration: BoxDecoration(
      color: Sumi.danger.withValues(alpha: 0.15),
      borderRadius: BorderRadius.circular(8),
    ),
    child: Text(error!, style: const TextStyle(color: Sumi.danger)),
  );

  Widget _playerBar(GameState g) {
    Widget side(String color) {
      final isHuman = g.humanColor == color;
      final captured = color == 'black' ? g.capturedWhite : g.capturedBlack;
      final active = !g.gameOver && _displayToMove(g) == color;
      return Expanded(
        child: Container(
          margin: const EdgeInsets.symmetric(horizontal: 8),
          padding: const EdgeInsets.symmetric(vertical: 10, horizontal: 14),
          decoration: BoxDecoration(
            color: Sumi.panel,
            borderRadius: BorderRadius.circular(10),
            border: Border.all(
              color: active ? Sumi.seal : Colors.transparent,
              width: 1.5,
            ),
          ),
          child: Row(
            children: [
              _stoneDot(color == 'black'),
              const SizedBox(width: 10),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      isHuman ? '你' : '玄石',
                      style: const TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                    Text(
                      '提子 $captured',
                      style: const TextStyle(
                        fontSize: 12,
                        color: Sumi.paperDim,
                      ),
                    ),
                  ],
                ),
              ),
              if (busy && !isHuman && (pending != null || g.toMove == color))
                _thinkingIndicator(black: color == 'black'),
            ],
          ),
        ),
      );
    }

    return Padding(
      padding: const EdgeInsets.only(top: 8),
      child: Row(children: [side('black'), side('white')]),
    );
  }

  /// 等待引擎回手時的「思考中」指示：以對方將要落下的那顆雲子呼吸縮放淡入淡出，
  /// 取代泛用轉圈圈——尤其「深思」128 sims 現在還要疊加真實網路延遲，等待感更明顯。
  Widget _thinkingIndicator({required bool black}) => AnimatedBuilder(
    animation: _thinkAnim,
    builder: (_, _) {
      final t = Curves.easeInOut.transform(_thinkAnim.value);
      return Opacity(
        opacity: 0.35 + 0.55 * t,
        child: Transform.scale(
          scale: 0.78 + 0.22 * t,
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              _stoneDot(black, size: 13),
              const SizedBox(width: 5),
              const Text(
                '思考中',
                style: TextStyle(fontSize: 11, color: Sumi.paperDim),
              ),
            ],
          ),
        ),
      );
    },
  );

  Widget _stoneDot(bool black, {double size = 22}) => Container(
    width: size,
    height: size,
    decoration: BoxDecoration(
      shape: BoxShape.circle,
      gradient: RadialGradient(
        center: const Alignment(-0.4, -0.4),
        colors: black
            ? [const Color(0xFF56534E), const Color(0xFF111010)]
            : [const Color(0xFFFFFEF9), const Color(0xFFC9C2B0)],
      ),
    ),
  );

  /// 勝率現況＋走勢合併成一張卡片（原本 bar 和 curve 各畫一次同一個數字）。
  Widget _winratePanel(GameState g) {
    final pct = (g.blackWinrate * 100).round();
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 10, 16, 0),
      child: Container(
        padding: const EdgeInsets.fromLTRB(12, 10, 12, 8),
        decoration: BoxDecoration(
          color: Sumi.panel,
          borderRadius: BorderRadius.circular(8),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                const Text(
                  '模型評估',
                  style: TextStyle(fontSize: 12, color: Sumi.paperDim),
                ),
                Text(
                  '黑勝率 $pct%',
                  style: const TextStyle(
                    fontSize: 13,
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 6),
            SizedBox(
              height: 44,
              width: double.infinity,
              child: CustomPaint(painter: WinrateCurvePainter(g.winrates)),
            ),
          ],
        ),
      ),
    );
  }

  Widget _controls(GameState g) {
    ButtonStyle style = OutlinedButton.styleFrom(
      foregroundColor: Sumi.paper,
      side: const BorderSide(color: Sumi.paperDim),
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
    );
    return Row(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        OutlinedButton(
          style: style,
          onPressed: humanTurn ? () => _tap(g.passAction) : null,
          child: const Text('虛手'),
        ),
        const SizedBox(width: 12),
        OutlinedButton(
          style: style,
          // 執白時 AI 先行的開局手不算可悔的自著
          onPressed:
              !busy &&
                  g.moves > (g.humanColor == 'white' ? 1 : 0) &&
                  !g.gameOver
              ? () => _run(() => api.undo(g.gameId))
              : null,
          child: const Text('悔棋'),
        ),
        const SizedBox(width: 12),
        OutlinedButton(
          style: style.copyWith(
            foregroundColor: const WidgetStatePropertyAll(Sumi.danger),
            side: const WidgetStatePropertyAll(BorderSide(color: Sumi.danger)),
          ),
          onPressed: !busy && !g.gameOver
              ? () => _confirmResign(g.gameId)
              : null,
          child: const Text('認輸'),
        ),
      ],
    );
  }

  Future<void> _confirmResign(String gameId) async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (_) => AlertDialog(
        backgroundColor: Sumi.panel,
        title: const Text('確定認輸？'),
        content: const Text(
          '這局會立即結束，無法復原。',
          style: TextStyle(color: Sumi.paperDim),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context, false),
            child: const Text('取消', style: TextStyle(color: Sumi.paperDim)),
          ),
          FilledButton(
            style: FilledButton.styleFrom(backgroundColor: Sumi.danger),
            onPressed: () => Navigator.pop(context, true),
            child: const Text('認輸'),
          ),
        ],
      ),
    );
    if (confirmed == true) _run(() => api.resign(gameId));
  }
}

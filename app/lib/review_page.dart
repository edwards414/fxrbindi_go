import 'package:flutter/material.dart';

import 'board_painter.dart';
import 'main.dart';
import 'match_history.dart';

class ReviewPage extends StatefulWidget {
  final MatchRecord record;

  const ReviewPage({super.key, required this.record});

  @override
  State<ReviewPage> createState() => _ReviewPageState();
}

class _ReviewPageState extends State<ReviewPage> {
  late final List<ReviewFrame> _frames;
  var _index = 0;

  @override
  void initState() {
    super.initState();
    _frames = MatchReplay.frames(widget.record);
    _index = _frames.length - 1;
  }

  ReviewFrame get _frame => _frames[_index];

  void _jump(int index) {
    setState(() => _index = index.clamp(0, _frames.length - 1));
  }

  @override
  Widget build(BuildContext context) {
    final record = widget.record;
    final frame = _frame;
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Sumi.bg,
        title: const Text(
          '對局回顧',
          style: TextStyle(fontSize: 20, fontWeight: FontWeight.w600),
        ),
      ),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.fromLTRB(16, 8, 16, 28),
          children: [
            _summary(record),
            const SizedBox(height: 14),
            AspectRatio(
              aspectRatio: 1,
              child: Semantics(
                label:
                    '第 ${frame.moveNumber}/${record.actions.length} 手棋盤'
                    '${frame.action == null ? '' : '，剛下 ${moveLabel(frame.action, record.boardSize)}'}',
                image: true,
                child: RepaintBoundary(
                  child: CustomPaint(
                    painter: BoardPainter(
                      size: record.boardSize,
                      board: frame.board,
                      lastMove: frame.action,
                    ),
                  ),
                ),
              ),
            ),
            const SizedBox(height: 14),
            _movePanel(record, frame),
            const SizedBox(height: 6),
            _timeline(),
            const SizedBox(height: 2),
            _controls(),
          ],
        ),
      ),
    );
  }

  Widget _summary(MatchRecord record) => Container(
    padding: const EdgeInsets.all(14),
    decoration: BoxDecoration(
      color: Sumi.panel,
      borderRadius: BorderRadius.circular(8),
    ),
    child: Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Expanded(
              child: Text(
                humanResultLabel(record),
                style: TextStyle(
                  color: record.winner == record.humanColor
                      ? Sumi.seal
                      : Sumi.paper,
                  fontSize: 26,
                  fontWeight: FontWeight.w800,
                ),
              ),
            ),
            Text(
              formatMatchDate(record.playedAt),
              style: const TextStyle(fontSize: 12, color: Sumi.paperDim),
            ),
          ],
        ),
        const SizedBox(height: 6),
        Text(
          '${resultLabel(record)} · 你執${colorLabel(record.humanColor)} · '
          '${matchLevelLabel(record.level)} · ${record.actions.length} 手',
          style: const TextStyle(fontSize: 13, color: Sumi.paperDim),
        ),
      ],
    ),
  );

  Widget _movePanel(MatchRecord record, ReviewFrame frame) {
    final moveText = frame.action == null
        ? '空盤'
        : '${frame.stoneColor == 1 ? '黑' : '白'} '
              '${moveLabel(frame.action, record.boardSize)}';
    final winrate = frame.blackWinrate;
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: Sumi.panel,
        borderRadius: BorderRadius.circular(8),
      ),
      child: Column(
        children: [
          Row(
            children: [
              Expanded(
                child: Text(
                  '第 ${frame.moveNumber}/${record.actions.length} 手',
                  style: const TextStyle(
                    fontSize: 17,
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ),
              Text(
                moveText,
                style: const TextStyle(
                  fontSize: 17,
                  fontWeight: FontWeight.w700,
                ),
              ),
            ],
          ),
          const SizedBox(height: 10),
          if (winrate == null)
            const Text(
              '黑勝率 --',
              style: TextStyle(fontSize: 12, color: Sumi.paperDim),
            )
          else
            Column(
              children: [
                ClipRRect(
                  borderRadius: BorderRadius.circular(4),
                  child: SizedBox(
                    height: 8,
                    child: Row(
                      children: [
                        Expanded(
                          flex: (winrate.clamp(0.02, 0.98) * 1000).round(),
                          child: Container(color: const Color(0xFF2B2823)),
                        ),
                        Expanded(
                          flex: ((1 - winrate.clamp(0.02, 0.98)) * 1000)
                              .round(),
                          child: Container(color: const Color(0xFFD8CFC0)),
                        ),
                      ],
                    ),
                  ),
                ),
                const SizedBox(height: 5),
                Text(
                  '模型評估 黑勝率 ${(winrate * 100).toStringAsFixed(0)}%',
                  style: const TextStyle(fontSize: 12, color: Sumi.paperDim),
                ),
              ],
            ),
        ],
      ),
    );
  }

  Widget _timeline() {
    final max = (_frames.length - 1).toDouble();
    return SliderTheme(
      data: SliderTheme.of(context).copyWith(
        activeTrackColor: Sumi.seal,
        inactiveTrackColor: Sumi.paperDim.withValues(alpha: 0.22),
        thumbColor: Sumi.seal,
        overlayColor: Sumi.seal.withValues(alpha: 0.14),
      ),
      child: Slider(
        value: _index.toDouble(),
        min: 0,
        max: max == 0 ? 1 : max,
        divisions: max == 0 ? null : max.round(),
        onChanged: max == 0 ? null : (value) => _jump(value.round()),
      ),
    );
  }

  Widget _controls() => Row(
    mainAxisAlignment: MainAxisAlignment.center,
    children: [
      _stepButton(
        icon: Icons.skip_previous,
        tooltip: '回到開局',
        enabled: _index > 0,
        onPressed: () => _jump(0),
      ),
      const SizedBox(width: 8),
      _stepButton(
        icon: Icons.chevron_left,
        tooltip: '上一手',
        enabled: _index > 0,
        onPressed: () => _jump(_index - 1),
      ),
      const SizedBox(width: 8),
      _stepButton(
        icon: Icons.chevron_right,
        tooltip: '下一手',
        enabled: _index < _frames.length - 1,
        onPressed: () => _jump(_index + 1),
      ),
      const SizedBox(width: 8),
      _stepButton(
        icon: Icons.skip_next,
        tooltip: '回到終局',
        enabled: _index < _frames.length - 1,
        onPressed: () => _jump(_frames.length - 1),
      ),
    ],
  );

  Widget _stepButton({
    required IconData icon,
    required String tooltip,
    required bool enabled,
    required VoidCallback onPressed,
  }) => IconButton.filledTonal(
    tooltip: tooltip,
    color: Sumi.paper,
    style: IconButton.styleFrom(
      backgroundColor: Sumi.panel,
      disabledBackgroundColor: Sumi.panel.withValues(alpha: 0.55),
      disabledForegroundColor: Sumi.paperDim.withValues(alpha: 0.45),
    ),
    onPressed: enabled ? onPressed : null,
    icon: Icon(icon),
  );
}

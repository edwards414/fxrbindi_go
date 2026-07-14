import 'package:flutter/material.dart';

import 'main.dart';
import 'match_history.dart';
import 'review_page.dart';

class HistoryPage extends StatefulWidget {
  const HistoryPage({super.key});

  @override
  State<HistoryPage> createState() => _HistoryPageState();
}

class _HistoryPageState extends State<HistoryPage> {
  var _loading = true;
  List<MatchRecord> _records = const [];

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    setState(() => _loading = true);
    final records = await MatchHistoryStore.load();
    if (!mounted) return;
    setState(() {
      _records = records;
      _loading = false;
    });
  }

  Future<void> _delete(MatchRecord record) async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        backgroundColor: Sumi.panel,
        title: const Text('刪除紀錄'),
        content: Text(
          '${formatMatchDate(record.playedAt)} · ${resultLabel(record)}',
          style: const TextStyle(color: Sumi.paperDim),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context, false),
            child: const Text('取消', style: TextStyle(color: Sumi.paperDim)),
          ),
          FilledButton(
            style: FilledButton.styleFrom(backgroundColor: Sumi.danger),
            onPressed: () => Navigator.pop(context, true),
            child: const Text('刪除'),
          ),
        ],
      ),
    );
    if (confirmed != true) return;

    await MatchHistoryStore.delete(record.id);
    await _load();
  }

  void _open(MatchRecord record) {
    Navigator.push(
      context,
      MaterialPageRoute(builder: (_) => ReviewPage(record: record)),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Sumi.bg,
        title: const Text(
          '對戰紀錄',
          style: TextStyle(fontSize: 20, fontWeight: FontWeight.w600),
        ),
        actions: [
          IconButton(
            tooltip: '重新整理',
            icon: const Icon(Icons.refresh, color: Sumi.paperDim),
            onPressed: _load,
          ),
        ],
      ),
      body: _loading
          ? const Center(child: CircularProgressIndicator(color: Sumi.seal))
          : _records.isEmpty
          ? const _EmptyHistory()
          : ListView.separated(
              padding: const EdgeInsets.fromLTRB(16, 8, 16, 28),
              itemCount: _records.length,
              separatorBuilder: (context, index) => const SizedBox(height: 10),
              itemBuilder: (context, index) => _recordTile(_records[index]),
            ),
    );
  }

  Widget _recordTile(MatchRecord record) {
    final humanColor = colorLabel(record.humanColor);
    final level = matchLevelLabel(record.level);
    return Material(
      color: Sumi.panel,
      borderRadius: BorderRadius.circular(8),
      child: InkWell(
        borderRadius: BorderRadius.circular(8),
        onTap: () => _open(record),
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
          child: Row(
            children: [
              _resultBadge(record),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      resultLabel(record),
                      style: const TextStyle(
                        fontSize: 17,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                    const SizedBox(height: 3),
                    Text(
                      '你執$humanColor · $level · ${record.actions.length} 手',
                      style: const TextStyle(
                        fontSize: 13,
                        color: Sumi.paperDim,
                      ),
                    ),
                    const SizedBox(height: 2),
                    Text(
                      formatMatchDate(record.playedAt),
                      style: const TextStyle(
                        fontSize: 12,
                        color: Sumi.paperDim,
                      ),
                    ),
                  ],
                ),
              ),
              IconButton(
                tooltip: '刪除',
                icon: const Icon(Icons.delete_outline, color: Sumi.paperDim),
                onPressed: () => _delete(record),
              ),
              const Icon(Icons.chevron_right, color: Sumi.paperDim),
            ],
          ),
        ),
      ),
    );
  }

  /// 用圖示（雲子色點 / 橫線 / 沙漏）取代硬塞文字的固定尺寸方塊：
  /// 系統字級調大也不會裁切，辨識也比讀「未完局」三個字快。
  Widget _resultBadge(MatchRecord record) {
    final youWon = record.winner == record.humanColor;
    final Widget icon;
    final String semanticLabel;
    if (record.winner == null) {
      icon = const Icon(Icons.hourglass_empty, size: 20, color: Sumi.paperDim);
      semanticLabel = '未完局';
    } else if (record.winner == 'draw') {
      icon = const Icon(Icons.remove, size: 20, color: Sumi.paperDim);
      semanticLabel = '和棋';
    } else {
      icon = _miniStone(black: record.winner == 'black');
      semanticLabel = youWon ? '你獲勝' : '玄石獲勝';
    }
    return Semantics(
      label: semanticLabel,
      child: Container(
        width: 44,
        height: 44,
        alignment: Alignment.center,
        decoration: BoxDecoration(
          color: youWon ? Sumi.seal.withValues(alpha: 0.18) : Sumi.bg,
          borderRadius: BorderRadius.circular(8),
          border: Border.all(color: youWon ? Sumi.seal : Sumi.line),
        ),
        child: icon,
      ),
    );
  }

  Widget _miniStone({required bool black}) => Container(
    width: 22,
    height: 22,
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
}

class _EmptyHistory extends StatelessWidget {
  const _EmptyHistory();

  @override
  Widget build(BuildContext context) => const Center(
    child: Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        SealGlyph('錄', size: 44, fontSize: 22),
        SizedBox(height: 12),
        Text(
          '尚無完局紀錄',
          style: TextStyle(fontSize: 18, fontWeight: FontWeight.w700),
        ),
        SizedBox(height: 4),
        Text(
          '完成一局後會自動保存',
          style: TextStyle(fontSize: 13, color: Sumi.paperDim),
        ),
      ],
    ),
  );
}

import 'dart:async';

import 'package:flutter/material.dart';

import 'api.dart';
import 'main.dart';
import 'player_id.dart';

/// 全 app 共用的體力狀態：首頁、設定頁、棋盤頁都看同一份。
/// 伺服器是真相；這裡只保存最後一次得知的數值並在本機倒數。
class StaminaModel extends ChangeNotifier {
  StaminaModel._();
  static final instance = StaminaModel._();

  StaminaInfo? info;
  bool supported = true; // 舊版伺服器沒有 /stamina 時關閉整個顯示
  Timer? _ticker;

  bool get isEmpty => info != null && info!.isEmptyAt(DateTime.now());

  /// 向伺服器重新讀取。連不上時保留舊值，不拋例外（體力顯示不該擋住整個 app）。
  Future<void> refresh(EngineApi api) async {
    try {
      final id = await PlayerId.get();
      final fresh = await api.stamina(id);
      if (fresh == null) {
        supported = false;
        info = null;
      } else {
        apply(fresh);
      }
    } catch (_) {
      /* 保留舊值 */
    }
    notifyListeners();
  }

  /// /new 的回應或 429 錯誤附帶的體力，直接採用。
  void apply(StaminaInfo fresh) {
    supported = true;
    info = fresh;
    notifyListeners();
    _ensureTicker();
  }

  void _ensureTicker() {
    if (_ticker != null) return;
    // 每 30 秒重新通知一次讓倒數文字更新；滿格後停掉
    _ticker = Timer.periodic(const Duration(seconds: 30), (_) {
      final i = info;
      if (i == null || i.nextInAt(DateTime.now()) == null) {
        _ticker?.cancel();
        _ticker = null;
      }
      notifyListeners();
    });
  }

  /// 「約 N 分鐘後回復 1 點」；滿格回 null。
  String? regenHint() {
    final i = info;
    if (i == null) return null;
    final d = i.nextInAt(DateTime.now());
    if (d == null) return null;
    final minutes = (d.inSeconds / 60).ceil();
    return minutes <= 1 ? '不到 1 分鐘後回復 1 點' : '約 $minutes 分鐘後回復 1 點';
  }

  String? exhaustedMessage() {
    final hint = regenHint();
    return hint == null ? null : '體力用完了，$hint';
  }
}

/// 「體力 n/24 · 約 x 分鐘後回復 1 點」一行小字。
class StaminaBar extends StatelessWidget {
  final TextAlign align;
  const StaminaBar({super.key, this.align = TextAlign.center});

  @override
  Widget build(BuildContext context) => ListenableBuilder(
    listenable: StaminaModel.instance,
    builder: (context, _) {
      final m = StaminaModel.instance;
      final i = m.info;
      if (!m.supported || i == null) return const SizedBox.shrink();
      final now = DateTime.now();
      final points = i.pointsAt(now);
      final hint = m.regenHint();
      final empty = points <= 0;
      return Semantics(
        label: '體力 $points／${i.max}${hint == null ? '' : '，$hint'}',
        child: Text(
          '體力 $points／${i.max}${hint == null ? '' : ' · $hint'}',
          textAlign: align,
          style: TextStyle(
            color: empty ? Sumi.danger : Sumi.paperDim,
            fontSize: 12,
          ),
        ),
      );
    },
  );
}

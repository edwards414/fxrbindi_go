"""玄石 app icon generator — renders 4 design variants at 1024px (+contact sheet).

Usage:
    .venv/bin/python scripts/gen_icon.py --out-dir /tmp/icons
    .venv/bin/python scripts/gen_icon.py --variant 1 --install   # write all iOS sizes
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib

import numpy as np
from PIL import Image, ImageDraw, ImageFont

S = 2048  # supersampled master; downscale to 1024 for AA

INK = (23, 19, 15)
PAPER = (237, 227, 210)
SEAL = (176, 58, 46)
WOOD_HI = (220, 169, 104)
WOOD_MID = (206, 152, 87)
WOOD_LO = (176, 126, 66)
LINE = (58, 47, 36)


def wood_bg() -> Image.Image:
    """對角漸層 + 波紋木紋。"""
    y, x = np.mgrid[0:S, 0:S].astype(np.float32)
    t = (x + y) / (2 * S)
    hi, mid, lo = (np.array(c, np.float32) for c in (WOOD_HI, WOOD_MID, WOOD_LO))
    col = np.where((t < 0.55)[..., None],
                   hi + (mid - hi) * (t / 0.55)[..., None],
                   mid + (lo - mid) * ((t - 0.55) / 0.45)[..., None])
    rng = np.random.default_rng(9)
    for _ in range(26):
        yy = rng.uniform(0, S)
        amp = rng.uniform(3, 9)
        phase = rng.uniform(0, 2 * math.pi)
        alpha = rng.uniform(0.05, 0.13)
        wave = yy + np.sin(x[0] / 68 + phase) * amp
        dist = np.abs(y - wave[None, :])
        m = np.clip(1.0 - dist / 3.0, 0, 1) * alpha
        col = col * (1 - m[..., None]) + np.array((107, 74, 34), np.float32) * m[..., None]
    return Image.fromarray(col.astype(np.uint8))


def stone(img: Image.Image, cx: float, cy: float, r: float, black: bool,
          shadow: float = 0.42):
    """雲子質感：柔影 + 徑向高光。"""
    pad = int(r * 1.6)
    x0, y0 = int(cx) - pad, int(cy) - pad
    y, x = np.mgrid[0:2 * pad, 0:2 * pad].astype(np.float32)
    dx, dy = x - pad, y - pad
    d = np.sqrt(dx ** 2 + dy ** 2)

    base = np.array(img.crop((x0, y0, x0 + 2 * pad, y0 + 2 * pad)), np.float32)
    # 陰影
    sd = np.sqrt((dx - r * 0.10) ** 2 + (dy - r * 0.14) ** 2)
    sh = np.clip(1 - sd / (r * 1.18), 0, 1) ** 1.5 * shadow
    base *= (1 - sh[..., None])
    # 石面
    hd = np.sqrt((dx + r * 0.36) ** 2 + (dy + r * 0.42) ** 2) / (2.1 * r)
    hd = np.clip(hd, 0, 1)
    if black:
        c0, c1, c2 = np.array((96, 96, 92), np.float32), np.array((38, 36, 32), np.float32), np.array((13, 12, 10), np.float32)
    else:
        c0, c1, c2 = np.array((255, 254, 249), np.float32), np.array((237, 232, 218), np.float32), np.array((195, 188, 172), np.float32)
    mid = 0.45 if black else 0.5
    face = np.where((hd < mid)[..., None], c0 + (c1 - c0) * (hd / mid)[..., None],
                    c1 + (c2 - c1) * ((hd - mid) / (1 - mid))[..., None])
    a = np.clip((r - d) / 2.5, 0, 1)  # 邊緣 AA
    out = base * (1 - a[..., None]) + face * a[..., None]
    img.paste(Image.fromarray(out.astype(np.uint8)), (x0, y0))


def grid(draw: ImageDraw.ImageDraw, cell: float, off_x: float, off_y: float, w: int):
    """整張畫布鋪滿棋盤線（局部特寫）。"""
    x = off_x
    while x < S + cell:
        draw.line([(x, -10), (x, S + 10)], fill=LINE, width=w)
        x += cell
    y = off_y
    while y < S + cell:
        draw.line([(-10, y), (S + 10, y)], fill=LINE, width=w)
        y += cell


def seal(img: Image.Image, cx: float, cy: float, size: float, char: str = "弈"):
    d = ImageDraw.Draw(img)
    r = size * 0.14
    d.rounded_rectangle([cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2],
                        radius=r, fill=SEAL)
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Songti.ttc",
                              int(size * 0.72), index=0)
    bb = d.textbbox((0, 0), char, font=font)
    d.text((cx - (bb[0] + bb[2]) / 2, cy - (bb[1] + bb[3]) / 2), char,
           font=font, fill=PAPER)


def v1():
    """木盤特寫 + 黑白對子 + 朱印。"""
    img = wood_bg()
    d = ImageDraw.Draw(img)
    cell = S / 3.2
    grid(d, cell, cell * 0.55, cell * 0.55, 10)
    gx = lambda i: cell * 0.55 + i * cell
    d.ellipse([gx(1) - 22, gx(1) - 22, gx(1) + 22, gx(1) + 22], fill=LINE)  # 星位
    stone(img, gx(2), gx(1), cell * 0.52, black=True)
    stone(img, gx(1), gx(2), cell * 0.40, black=False)
    seal(img, S * 0.82, S * 0.83, S * 0.20)
    return img


def v2():
    """深墨底 + 大玄石 + 朱印。"""
    img = Image.new("RGB", (S, S), INK)
    y, x = np.mgrid[0:S, 0:S].astype(np.float32)
    vg = np.clip(1 - np.sqrt((x - S/2)**2 + (y - S/2)**2) / (S * 0.75), 0, 1) * 0.12
    arr = np.array(img, np.float32)
    arr += (np.array((90, 75, 55), np.float32) - arr) * vg[..., None]
    img = Image.fromarray(arr.astype(np.uint8))
    stone(img, S * 0.5, S * 0.47, S * 0.34, black=True)
    seal(img, S * 0.78, S * 0.80, S * 0.19)
    return img


def v3():
    """木盤 + 完整 9 路小盤 + 三子。"""
    img = wood_bg()
    d = ImageDraw.Draw(img)
    m = S * 0.12
    cell = (S - 2 * m) / 8
    for i in range(9):
        w = 12 if i in (0, 8) else 7
        d.line([(m + i * cell, m), (m + i * cell, S - m)], fill=LINE, width=w)
        d.line([(m, m + i * cell), (S - m, m + i * cell)], fill=LINE, width=w)
    for p in ((2, 2), (2, 6), (4, 4), (6, 2), (6, 6)):
        cx, cy = m + p[0] * cell, m + p[1] * cell
        d.ellipse([cx - 16, cy - 16, cx + 16, cy + 16], fill=LINE)
    pt = lambda i, j: (m + i * cell, m + j * cell)
    stone(img, *pt(6, 2), cell * 0.62, black=True)
    stone(img, *pt(2, 6), cell * 0.62, black=False)
    stone(img, *pt(4, 4), cell * 0.62, black=True)
    return img


def v4():
    """宣紙極簡：墨圓 + 朱印。

    評審團定稿版：石下加淡墨天元交叉線（圍棋辨識）、棋子光學置中、
    陰影減淡、朱印內收避開 iOS 圓角遮罩。
    """
    img = Image.new("RGB", (S, S), PAPER)
    y, x = np.mgrid[0:S, 0:S].astype(np.float32)
    vg = np.clip(np.sqrt((x - S/2)**2 + (y - S/2)**2) / (S * 0.85), 0, 1) ** 2 * 0.10
    arr = np.array(img, np.float32) * (1 - vg[..., None])
    img = Image.fromarray(arr.astype(np.uint8))
    # 淡墨棋盤線：一縱一橫交會於石心（~15% 對比）
    d = ImageDraw.Draw(img)
    cx, cy = S * 0.49, S * 0.46
    faint = tuple(int(p * 0.85 + i * 0.15) for p, i in zip(PAPER, INK))
    d.line([(cx, S * 0.06), (cx, S * 0.94)], fill=faint, width=9)
    d.line([(S * 0.06, cy), (S * 0.94, cy)], fill=faint, width=9)
    stone(img, cx, cy, S * 0.30, black=True, shadow=0.29)
    seal(img, S * 0.78, S * 0.80, S * 0.17)
    return img


IOS_APPICONSET = "app/ios/Runner/Assets.xcassets/AppIcon.appiconset"


def install(img: Image.Image, root: pathlib.Path):
    iconset = root / IOS_APPICONSET
    meta = json.loads((iconset / "Contents.json").read_text())
    for entry in meta["images"]:
        pts = float(entry["size"].split("x")[0])
        scale = int(entry["scale"].rstrip("x"))
        px = int(round(pts * scale))
        img.resize((px, px), Image.LANCZOS).save(iconset / entry["filename"])
    print(f"installed {len(meta['images'])} sizes into {iconset}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="/tmp/icons")
    p.add_argument("--variant", type=int, default=None)
    p.add_argument("--install", action="store_true")
    args = p.parse_args()
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    root = pathlib.Path(__file__).resolve().parent.parent

    variants = {1: v1, 2: v2, 3: v3, 4: v4}
    todo = {args.variant: variants[args.variant]} if args.variant else variants
    sheet = Image.new("RGB", (4 * 300 + 100, 480), (40, 40, 40))
    for i, (k, fn) in enumerate(todo.items()):
        img = fn().resize((1024, 1024), Image.LANCZOS)
        img.save(out / f"icon_v{k}.png")
        if args.install:
            install(img, root)
        # contact sheet: 240px + 120px + 60px（模擬桌面/設定/通知尺寸）
        x0 = 20 + i * 300
        sheet.paste(img.resize((240, 240), Image.LANCZOS), (x0, 20))
        sheet.paste(img.resize((120, 120), Image.LANCZOS), (x0, 280))
        sheet.paste(img.resize((60, 60), Image.LANCZOS), (x0 + 140, 280))
    if not args.variant:
        sheet.save(out / "contact_sheet.png")
        print(f"wrote variants + contact sheet to {out}")


if __name__ == "__main__":
    main()

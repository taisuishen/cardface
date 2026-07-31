"""离线自测：合成"手机拍证件"的图，验证角点解码、姿态判定和透视矫正。

    python server/selftest.py

会在 selftest_out/ 写出可视化结果（绿框=合格，橙框=不合格）。
"""
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from card_detector import CardPoseDetector, PoseRule   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "selftest_out")


# ---------------------------------------------------------------- 合成证件
def make_card(w=856, h=540):
    """带圆角、渐变底纹、人像、印刷文字的仿真证件。返回 (图, 圆角遮罩)。"""
    c = np.zeros((h, w, 3), np.uint8)
    for y in range(h):
        t = y / h
        c[y, :] = (int(238 - 18 * t), int(240 - 12 * t), int(236 - 6 * t))
    xs = np.arange(w)
    for k in range(0, h, 7):
        yy = (k + 9 * np.sin(xs / 47.0 + k / 23.0)).astype(int)
        m = (yy >= 0) & (yy < h)
        c[yy[m], xs[m]] = np.clip(c[yy[m], xs[m]].astype(int) - 9, 0, 255)

    px, py, pw, ph = int(w * .66), int(h * .16), int(w * .26), int(h * .68)
    for y in range(ph):
        c[py + y, px:px + pw] = (200 - y // 6, 205 - y // 7, 212 - y // 8)
    cv2.circle(c, (px + pw // 2, py + int(ph * .32)), int(ph * .19), (150, 158, 170), -1)
    cv2.ellipse(c, (px + pw // 2, py + int(ph * .95)), (int(pw * .40), int(ph * .34)),
                0, 180, 360, (150, 158, 170), -1)
    cv2.rectangle(c, (px, py), (px + pw, py + ph), (170, 175, 185), 2)

    cv2.putText(c, "RESIDENT IDENTITY CARD", (int(w * .07), int(h * .13)),
                cv2.FONT_HERSHEY_SIMPLEX, .72, (52, 62, 92), 2)
    for i, lb in enumerate(["NAME", "SEX / ETHNIC", "BIRTH", "ADDRESS", "ID NO."]):
        y = int(h * .27) + i * int(h * .13)
        cv2.putText(c, lb, (int(w * .07), y), cv2.FONT_HERSHEY_SIMPLEX, .40, (110, 118, 132), 1)
        cv2.rectangle(c, (int(w * .07), y + 8),
                      (int(w * (.30 + .22 * ((i * 7) % 3) / 2)), y + 26), (58, 64, 78), -1)

    mask = np.zeros((h, w), np.uint8)
    r = int(h * .06)
    cv2.rectangle(mask, (r, 0), (w - r, h), 255, -1)
    cv2.rectangle(mask, (0, r), (w, h - r), 255, -1)
    for cx, cy in [(r, r), (w - r, r), (r, h - r), (w - r, h - r)]:
        cv2.circle(mask, (cx, cy), r, 255, -1)
    return c, mask


def shoot(card, cmask, W=1280, H=960, angle=0., persp=0., scale=1.0,
          blur=1.6, noise=5, light=.22, seed=0, shift=(0, 0)):
    """模拟手机拍摄：贴背景 + 旋转/透视 + 平移 + 光照梯度 + 模糊 + 噪点 + JPEG。"""
    rng = np.random.RandomState(seed)
    bg = np.full((H, W, 3), (96, 104, 118), np.uint8)
    bg = cv2.GaussianBlur(
        np.clip(bg.astype(np.int16) + rng.randn(H, W, 3) * 7, 0, 255).astype(np.uint8), (0, 0), 3)

    h, w = card.shape[:2]
    cw, ch = w * scale, h * scale
    cx, cy = W / 2, H / 2
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    base = np.float32([[cx - cw / 2, cy - ch / 2], [cx + cw / 2, cy - ch / 2],
                       [cx + cw / 2, cy + ch / 2], [cx - cw / 2, cy + ch / 2]])
    if persp:                       # 右边变短：模拟从左侧斜着拍
        d = ch * persp / 2
        base[1][1] += d
        base[2][1] -= d
    if angle:
        R = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        base = (R @ np.hstack([base, np.ones((4, 1), np.float32)]).T).T.astype(np.float32)
    if any(shift):
        base = base + np.float32(shift)

    M = cv2.getPerspectiveTransform(src, base)
    out = bg.copy()
    warped = cv2.warpPerspective(card, M, (W, H))
    mask = cv2.warpPerspective(cmask, M, (W, H))
    out[mask > 127] = warped[mask > 127]

    gx = np.linspace(1 - light, 1 + light, W)[None, :]
    gy = np.linspace(1 + light * .5, 1 - light * .5, H)[:, None]
    out = np.clip(out.astype(np.float32) * (gx * gy)[..., None], 0, 255).astype(np.uint8)
    if blur:
        out = cv2.GaussianBlur(out, (0, 0), blur)
    if noise:
        out = np.clip(out.astype(np.int16) + rng.randn(H, W, 3) * noise, 0, 255).astype(np.uint8)
    _, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR), base


# slug(用于输出文件名，保持 ASCII 以免 Windows 代码页问题), 说明, kwargs, 期望 ok, 期望 reason
# 注意：歪的用例要用较小的 scale，否则旋转后证件本来就伸出画面，
# 会先命中 out_of_frame，测不到 tilted。
CASES = [
    ("01_upright",     "正放",         dict(angle=0,   scale=1.35),              True,  "ok"),
    ("02_tilt3",       "轻微歪 3度",    dict(angle=3,   scale=1.35),              True,  "ok"),
    ("03_tilt15",      "歪 15度",      dict(angle=15,  scale=0.9),               False, "tilted"),
    ("04_tilt30",      "歪 30度",      dict(angle=30,  scale=0.9),               False, "tilted"),
    ("05_tilt90",      "侧放 90度",     dict(angle=90,  scale=0.9),               False, "tilted"),
    # 倒置证件的角点精度随尺度波动（0.85~1.0 能量出 ~180°，其它尺度四边形会变形），
    # 但任何尺度下都不会被判为合格，这里只断言"必须拒绝"。
    ("06_upsidedown",  "倒置 180度",    dict(angle=180, scale=0.9),               False,
     ("upside_down", "skewed", "tilted", "not_rect")),
    ("07_skew035",     "斜着拍 0.35",   dict(angle=0,   scale=1.35, persp=.35),   False, "skewed"),
    ("08_skew06",      "斜着拍 0.6",    dict(angle=0,   scale=1.35, persp=.6),    False, "skewed"),
    ("09_toofar",      "太远 scale0.5", dict(angle=0,   scale=0.5),               False, "too_small"),
    ("10_tilt_skew",   "歪+斜",        dict(angle=20,  scale=0.9,  persp=.4),     False, "tilted"),
    ("11_outofframe",  "出框(偏移)",     dict(angle=0,   scale=1.35, shift=(430, 0)), False, "out_of_frame"),
    ("12_nocard",      "空背景无证件",    None,                                     False, "no_card"),
]


def main():
    os.makedirs(OUT, exist_ok=True)
    print("加载模型…")
    t = time.time()
    det = CardPoseDetector(os.path.join(ROOT, "cardpose.onnx"), PoseRule())
    print(f"providers={det.providers}  imgsz={det.imgsz}  加载+预热 {time.time()-t:.1f}s\n")

    card, cmask = make_card()
    cv2.imwrite(os.path.join(OUT, "00_card_source.jpg"), card)

    ok_all, lat = True, []
    print(f"{'用例':<16}{'ok':<7}{'reason':<13}{'conf':<7}{'rot':>8}{'skew':>8}"
          f"{'area':>7}{'asp':>7}{'ms':>7}")
    print("-" * 82)

    for slug, name, kw, want_ok, want_reason in CASES:
        if kw is None:
            img = shoot(card, cmask, scale=1.35)[0]
            img[:] = cv2.GaussianBlur(img, (0, 0), 25)      # 糊成一片，当作没有证件
        else:
            img = shoot(card, cmask, **kw)[0]

        t0 = time.perf_counter()
        r = det.judge(img)
        ms = (time.perf_counter() - t0) * 1000
        lat.append(ms)

        want = (want_reason,) if isinstance(want_reason, str) else want_reason
        hit = (r.ok == want_ok) and (want_ok or r.reason in want)
        ok_all &= hit
        flag = "PASS" if hit else "FAIL"
        print(f"{name:<16}{str(r.ok):<7}{r.reason:<13}{r.conf:<7.2f}{r.rotate_deg:>8.2f}"
              f"{r.skew:>8.3f}{r.area_ratio:>7.3f}{r.aspect:>7.2f}{ms:>7.0f}  {flag}")
        if not hit:
            print(f"    ↳ 期望 ok={want_ok} reason={want_reason}")

        vis = img.copy()
        if r.quad:
            q = np.array(r.quad, np.int32)
            col = (0, 255, 0) if r.ok else (0, 165, 255)
            cv2.polylines(vis, [q], True, col, 3)
            for i, p in enumerate(q):
                cv2.circle(vis, tuple(p), 8, (0, 0, 255), -1)
                cv2.putText(vis, ["TL", "TR", "BR", "BL"][i], tuple(p + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 255), 2)
        cv2.putText(vis, f"{r.reason}  rot={r.rotate_deg:.1f} skew={r.skew:.2f}",
                    (20, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 255, 0) if r.ok else (0, 165, 255), 2)
        cv2.imwrite(os.path.join(OUT, f"{slug}.jpg"), vis)
        if r.ok:
            crop = CardPoseDetector.warp(img, r.quad)
            cv2.imwrite(os.path.join(OUT, f"{slug}_crop.jpg"), crop)

    # 角度测量精度：rotate_deg 必须在整个 ±180° 范围内都跟真实角度一致，
    # 而不是只在小角度对 —— 否则歪得厉害的证件会被量成 0° 而误判合格。
    print("\n旋转角测量精度（真实角度 -> 测得 rotate_deg，拍摄逆时针为正，图像顺时针为正故取反）")
    worst = 0.0
    for a in (0, 2, 5, 8, 12, 20, 30, 45, 60, 90, 135, 180, -15, -45, -90):
        r = det.judge(shoot(card, cmask, angle=a, scale=0.85)[0])
        pred = -r.rotate_deg
        err = abs((pred - a + 180) % 360 - 180)
        worst = max(worst, err)
        print(f"  真实 {a:5d}°  测得 {pred:8.2f}°  误差 {err:5.2f}°"
              f"  {'OK' if err < 6 else '偏差偏大'}")
    print(f"  最大误差 {worst:.2f}°")

    print(f"\n单帧推理: 平均 {sum(lat)/len(lat):.0f}ms  最快 {min(lat):.0f}ms  "
          f"→ 单路约 {1000/(sum(lat)/len(lat)):.1f} fps")
    print(f"可视化结果: {OUT}")
    print("\n总体:", "全部通过 ✓" if ok_all else "有用例不符合预期 ✗")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())

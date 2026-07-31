"""分析真机录制的帧，定位"对准了框还在跳"到底出在哪一环。

    python tools/analyze_dump.py dumps/20260731_213000

输出：
  * 每帧的检测置信度 / 角点 / 判定，以及相邻帧的角点跳动量
  * 平滑前 vs 平滑后的抖动对比（看时域平滑到底有没有用）
  * 判定翻转次数（"对准了还反复跳"的量化指标）
  * 清晰度(拉普拉斯方差)与抖动的相关性 —— 用来区分是模型不准还是手机糊了
  * 可视化叠加图写到 <dump>/_vis/
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import Counter

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "server"))
from card_detector import CardPoseDetector, CardTracker, PoseRule, SmoothRule   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(dump: str):
    files = sorted(glob.glob(os.path.join(dump, "*.jpg")))
    if not files:
        print(f"{dump} 里没有 .jpg —— 先在网页上点「录 50 帧供排查」")
        return 1
    print(f"{len(files)} 帧  <- {dump}\n")

    det = CardPoseDetector(os.path.join(ROOT, "cardpose.onnx"), PoseRule())
    vis_dir = os.path.join(dump, "_vis")
    os.makedirs(vis_dir, exist_ok=True)

    raws, sharp, reasons_raw = [], [], []
    print(f"{'帧':>4} {'conf':>6} {'清晰度':>8} {'跳动px':>8} {'rot':>7} {'skew':>7} {'判定':<14}")
    print("-" * 62)

    prev = None
    for i, f in enumerate(files):
        img = cv2.imread(f)
        if img is None:
            continue
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lap = float(cv2.Laplacian(g, cv2.CV_64F).var())     # 越小越糊
        sharp.append(lap)

        q, conf = det._infer(img)
        if q is None:
            raws.append(None)
            reasons_raw.append("no_card")
            print(f"{i:>4} {conf:>6.2f} {lap:>8.0f} {'-':>8} {'-':>7} {'-':>7} {'no_card':<14}")
            prev = None
            continue

        r = det.judge_quad(q, conf, img.shape[1], img.shape[0])
        jump = float(np.linalg.norm(q - prev, axis=1).mean()) if prev is not None else 0.0
        prev = q
        raws.append(q)
        reasons_raw.append(r.reason)
        print(f"{i:>4} {conf:>6.2f} {lap:>8.0f} {jump:>8.1f} {r.rotate_deg:>7.2f} "
              f"{r.skew:>7.3f} {r.reason:<14}")

        vq = np.array(r.quad, np.int32)
        v = img.copy()
        cv2.polylines(v, [vq], True, (0, 255, 0) if r.ok else (0, 165, 255), 2)
        cv2.putText(v, f"{i} {r.reason} rot={r.rotate_deg:.1f} sharp={lap:.0f}",
                    (12, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2)
        cv2.imwrite(os.path.join(vis_dir, f"{i:05d}.jpg"), v)

    got = [q for q in raws if q is not None]
    print(f"\n检出率 {len(got)}/{len(files)} = {len(got)/len(files)*100:.0f}%")
    print(f"清晰度(拉普拉斯方差): 中位 {np.median(sharp):.0f}  最低 {min(sharp):.0f}  "
          f"最高 {max(sharp):.0f}   （<100 基本就是糊了/失焦）")

    # 平滑前 vs 平滑后：同一批帧再跑一遍 tracker
    tr = CardTracker(det, SmoothRule())
    sm_reasons, sm_quads = [], []
    for f in files:
        img = cv2.imread(f)
        if img is None:
            continue
        r = tr.step(img)
        sm_reasons.append(r.reason)
        sm_quads.append(np.array(r.quad) if r.quad else None)

    def jumps(seq):
        out = []
        for a, b in zip(seq, seq[1:]):
            if a is not None and b is not None:
                out.append(float(np.linalg.norm(b - a, axis=1).mean()))
        return out

    jr, js = jumps(raws), jumps(sm_quads)

    def flips(rs):
        ok = [x == "ok" for x in rs]
        return sum(1 for a, b in zip(ok, ok[1:]) if a != b)

    print("\n              相邻帧角点跳动(px)          判定翻转次数   判定分布")
    if jr:
        print(f"  平滑前   平均 {np.mean(jr):5.1f}  最大 {max(jr):6.1f}"
              f"        {flips(reasons_raw):>4}        {dict(Counter(reasons_raw))}")
    if js:
        print(f"  平滑后   平均 {np.mean(js):5.1f}  最大 {max(js):6.1f}"
              f"        {flips(sm_reasons):>4}        {dict(Counter(sm_reasons))}")
    if jr and js:
        print(f"\n  抖动降低 {(1-np.mean(js)/max(np.mean(jr),1e-6))*100:.0f}%"
              f"，判定翻转 {flips(reasons_raw)} -> {flips(sm_reasons)} 次")

    # 抖动和清晰度的关系：如果强相关，说明问题在"拍得糊"而不是模型
    if len(jr) > 5:
        s = np.array(sharp[1:len(jr) + 1])
        c = np.corrcoef(s, np.array(jr))[0, 1]
        print(f"\n  清晰度 vs 抖动 相关系数 {c:+.2f}"
              f"  ({'负相关明显：越糊越抖，优先解决对焦/曝光' if c < -0.3 else '无明显相关：抖动主要来自模型本身'})")

    print(f"\n可视化: {vis_dir}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        cands = sorted(glob.glob(os.path.join(ROOT, "dumps", "*")))
        if not cands:
            print(__doc__)
            sys.exit(1)
        target = cands[-1]
        print(f"(未指定目录，用最新一次录制: {target})\n")
    else:
        target = sys.argv[1]
    sys.exit(main(target))

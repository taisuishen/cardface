"""并发压测：模拟 N 个手机同时以 3fps 推流，看服务端能扛住几个。

    python tools/load_test.py wss://<POD_ID>-8000.proxy.runpod.net/ws
    python tools/load_test.py ws://127.0.0.1:8000/ws --users 1,2,4,8,12

在 Pod 里对 127.0.0.1 跑最准 —— 从外网跑的话，你自己的上行带宽
（每用户约 300KB/s）很容易先成为瓶颈，那测出来的就不是服务端能力了。

判读：
  * 服务端耗时 p50/p95 —— 排队会让它涨上去
  * 丢帧率        —— 服务端来不及处理就会丢（latest-wins 背压）
  * 实际发送 fps  —— 明显低于目标说明是【客户端/网络】瓶颈，不是服务端
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "server"))

try:
    import websockets
except ImportError:
    sys.exit("需要 websockets: pip install websockets")


def build_frame(long_side: int = 960, target_kb: int = 100) -> bytes:
    """做一张 ~100KB 的合成证件帧，尺寸和网页真实上传的一致。"""
    from selftest import make_card, shoot
    card, cmask = make_card()
    img = shoot(card, cmask, angle=1, scale=1.3)[0]
    h, w = img.shape[:2]
    s = long_side / max(w, h)
    img = cv2.resize(img, (round(w * s), round(h * s)))
    q = 95
    while q > 30:
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
        if len(enc) <= target_kb * 1024:
            break
        q -= 5
    return enc.tobytes()


class Client:
    def __init__(self, url: str, frame: bytes, fps: float, secs: float, uid: int):
        self.url, self.frame, self.fps, self.secs, self.uid = url, frame, fps, secs, uid
        self.ms: list[float] = []
        self.rtt: list[float] = []
        self.sent = 0
        self.got = 0
        self.dropped = 0
        self.err: str | None = None

    async def run(self):
        try:
            async with websockets.connect(self.url, max_size=8 << 20,
                                          open_timeout=30, ping_interval=None) as ws:
                await ws.recv()                      # hello

                pending: dict[int, float] = {}
                stop = asyncio.Event()

                async def rx():
                    try:
                        while True:
                            m = json.loads(await ws.recv())
                            if m.get("type") != "result":
                                continue
                            self.got += 1
                            self.dropped += m.get("dropped", 0)
                            # 压测要测"持续吞吐"，但服务端抓到合格结果会闭锁。
                            # 收到 final 就立刻 reset，让它继续处理后面的帧。
                            if m.get("final"):
                                await ws.send(json.dumps({"type": "reset"}))
                            if "ms" in m:
                                self.ms.append(m["ms"])
                            t0 = pending.pop(m.get("seq", -1), None)
                            if t0:
                                self.rtt.append((time.perf_counter() - t0) * 1000)
                    except Exception:                # noqa: BLE001
                        pass

                rt = asyncio.create_task(rx())
                t_end = time.perf_counter() + self.secs
                period = 1.0 / self.fps
                nxt = time.perf_counter()
                while time.perf_counter() < t_end:
                    await ws.send(self.frame)
                    self.sent += 1
                    pending[self.sent] = time.perf_counter()
                    nxt += period
                    d = nxt - time.perf_counter()
                    if d > 0:
                        await asyncio.sleep(d)
                await asyncio.sleep(2.0)              # 等尾部结果
                stop.set()
                rt.cancel()
        except Exception as e:                        # noqa: BLE001
            self.err = f"{type(e).__name__}: {e}"


async def round_n(url: str, frame: bytes, n: int, fps: float, secs: float):
    cs = [Client(url, frame, fps, secs, i) for i in range(n)]
    t0 = time.perf_counter()
    await asyncio.gather(*(c.run() for c in cs))
    el = time.perf_counter() - t0

    errs = [c.err for c in cs if c.err]
    ms = [x for c in cs for x in c.ms]
    rtt = [x for c in cs for x in c.rtt]
    sent = sum(c.sent for c in cs)
    got = sum(c.got for c in cs)
    drop = sum(c.dropped for c in cs)

    def pct(v, p):
        if not v:
            return 0.0
        v = sorted(v)
        return v[min(len(v) - 1, int(len(v) * p))]

    target = n * fps
    actual_send = sent / max(el - 2.0, 1e-6)         # 减去尾部等待
    print(f"{n:>4} {target:>8.0f} {actual_send:>9.1f} {got/max(el-2.0,1e-6):>9.1f} "
          f"{statistics.median(ms) if ms else 0:>8.0f} {pct(ms,0.95):>8.0f} "
          f"{(statistics.median(rtt) if rtt else 0):>9.0f} "
          f"{drop:>6} {drop/max(sent,1)*100:>7.1f}% {len(errs):>5}")
    if errs:
        print(f"       错误示例: {errs[0]}")
    return {"n": n, "send_fps": actual_send, "res_fps": got / max(el - 2.0, 1e-6),
            "ms_p50": statistics.median(ms) if ms else 0, "ms_p95": pct(ms, 0.95),
            "drop_pct": drop / max(sent, 1) * 100, "errs": len(errs)}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url", nargs="?", default="ws://127.0.0.1:8000/ws")
    ap.add_argument("--users", default="1,2,4,8,12")
    ap.add_argument("--fps", type=float, default=3.0)
    ap.add_argument("--secs", type=float, default=8.0)
    a = ap.parse_args()

    frame = build_frame()
    print(f"目标: {a.url}   每用户 {a.fps}fps   单帧 {len(frame)/1024:.0f}KB   每档 {a.secs:.0f}s\n")
    print(f"{'并发':>4} {'目标fps':>8} {'实发fps':>9} {'结果fps':>9} "
          f"{'服务ms':>8} {'p95ms':>8} {'往返ms':>9} {'丢帧':>6} {'丢帧率':>8} {'错误':>5}")
    print("-" * 92)

    rows = []
    for n in [int(x) for x in a.users.split(",")]:
        rows.append(await round_n(a.url, frame, n, a.fps, a.secs))
        await asyncio.sleep(2)

    print("\n判读:")
    base = rows[0]["ms_p50"] or 1
    knee = None
    for r in rows:
        if r["drop_pct"] > 5 or r["ms_p50"] > base * 2.5 or r["errs"]:
            knee = r["n"]
            break
    if knee:
        print(f"  拐点出现在 {knee} 并发（丢帧>5% 或 服务端耗时翻倍以上）")
        print(f"  建议按 {max(1, knee - 1)} 个并发用户规划单卡容量")
    else:
        top = rows[-1]
        print(f"  测到 {top['n']} 并发仍健康（服务端 p50 {top['ms_p50']:.0f}ms，"
              f"丢帧 {top['drop_pct']:.1f}%），上限还没探到")
    if any(r["send_fps"] < r["n"] * a.fps * 0.85 for r in rows):
        print("  ⚠ 有档位实发 fps 明显低于目标 —— 那几档是客户端/网络上行瓶颈，")
        print("    服务端能力被低估了，请在 Pod 里对 ws://127.0.0.1:8000/ws 重测")


if __name__ == "__main__":
    asyncio.run(main())

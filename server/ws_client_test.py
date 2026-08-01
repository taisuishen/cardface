"""端到端测试：模拟手机按 3fps 推 ~100KB 的 JPEG，检查服务端回传是否正确。

    python server/ws_client_test.py                      # 默认 wss://localhost:8443/ws
    python server/ws_client_test.py ws://localhost:8000/ws
"""
import asyncio
import base64
import json
import os
import ssl
import sys
import time

import cv2
import numpy as np
import websockets

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from selftest import make_card, shoot          # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "selftest_out")
TESTDATA = os.path.join(ROOT, "testdata")
URL = sys.argv[1] if len(sys.argv) > 1 else "wss://localhost:8443/ws"
FPS = 3

# 人脸检测必须用真实照片才测得准（手画的脸 YuNet 检不出来）。
# 用 OpenCV 官方 samples 里的公开测试图，首次运行下载并缓存到 testdata/。
FACE_SAMPLE = ("messi5.jpg",
               "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/messi5.jpg")


def fetch_sample(name, url):
    """返回本地图片路径；下载不到就返回 None（离线时跳过真人脸用例）。"""
    os.makedirs(TESTDATA, exist_ok=True)
    p = os.path.join(TESTDATA, name)
    if os.path.exists(p) and os.path.getsize(p) > 10000:
        return p
    try:
        import urllib.request
        with urllib.request.urlopen(url, timeout=30) as r, open(p, "wb") as f:
            f.write(r.read())
        return p if os.path.getsize(p) > 10000 else None
    except Exception as e:                                       # noqa: BLE001
        print(f"  （下载 {name} 失败，跳过真人脸用例: {e}）")
        return None


def to_jpeg(bgr, target_kb=100):
    """像前端那样把图压到 ~100KB。"""
    q = 90
    for _ in range(8):
        ok, enc = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, q])
        kb = len(enc) / 1024
        if kb > target_kb * 1.1:
            q -= 8
        elif kb < target_kb * 0.7:
            q += 5
        else:
            break
        q = max(30, min(95, q))
    return enc.tobytes()


def draw_face(W=960, H=720):
    """画一张有明暗、五官比例大致正确的人脸，用来试 YuNet 能不能检到。"""
    img = np.full((H, W, 3), 70, np.uint8)
    cx, cy = W // 2, int(H * .52)
    fw, fh = int(W * .21), int(H * .30)
    # 脖子 + 肩
    cv2.ellipse(img, (cx, cy + int(fh * 1.5)), (int(fw * 1.9), int(fh * .9)), 0, 180, 360, (92, 104, 126), -1)
    cv2.rectangle(img, (cx - fw // 3, cy + fh // 2), (cx + fw // 3, cy + fh * 2), (176, 156, 142), -1)
    # 脸
    cv2.ellipse(img, (cx, cy), (fw, fh), 0, 0, 360, (188, 168, 152), -1)
    # 简单光照：左亮右暗
    m = np.zeros((H, W), np.uint8)
    cv2.ellipse(m, (cx, cy), (fw, fh), 0, 0, 360, 255, -1)
    grad = np.linspace(1.14, .86, W)[None, :]
    reg = img.astype(np.float32) * grad[..., None]
    img[m > 0] = np.clip(reg, 0, 255).astype(np.uint8)[m > 0]
    # 头发
    cv2.ellipse(img, (cx, cy - int(fh * .30)), (int(fw * 1.04), int(fh * .70)), 0, 180, 360, (46, 40, 38), -1)
    # 眼睛（眼白 + 瞳孔 + 上眼睑阴影）
    ey = cy - int(fh * .16)
    for sx in (-1, 1):
        ex = cx + sx * int(fw * .40)
        cv2.ellipse(img, (ex, ey), (int(fw * .21), int(fh * .10)), 0, 0, 360, (248, 246, 242), -1)
        cv2.circle(img, (ex, ey), int(fh * .075), (58, 44, 34), -1)
        cv2.circle(img, (ex, ey), int(fh * .030), (12, 10, 10), -1)
        cv2.ellipse(img, (ex, ey - int(fh * .06)), (int(fw * .23), int(fh * .07)), 0, 180, 360, (150, 128, 112), -1)
        cv2.ellipse(img, (ex, ey - int(fh * .17)), (int(fw * .24), int(fh * .06)), 0, 180, 360, (60, 50, 46), 3)
    # 鼻子
    cv2.ellipse(img, (cx, cy + int(fh * .17)), (int(fw * .15), int(fh * .10)), 0, 0, 180, (162, 142, 128), -1)
    cv2.line(img, (cx - 2, cy - int(fh * .05)), (cx - int(fw * .07), cy + int(fh * .14)), (168, 148, 134), 3)
    # 嘴
    cv2.ellipse(img, (cx, cy + int(fh * .48)), (int(fw * .34), int(fh * .11)), 0, 0, 180, (146, 92, 88), -1)
    cv2.line(img, (cx - int(fw * .34), cy + int(fh * .48)), (cx + int(fw * .34), cy + int(fh * .48)), (120, 74, 72), 2)
    # 眉毛
    for sx in (-1, 1):
        ex = cx + sx * int(fw * .40)
        cv2.ellipse(img, (ex, ey - int(fh * .30)), (int(fw * .25), int(fh * .10)), 0, 190, 350, (52, 44, 40), 6)
    img = cv2.GaussianBlur(img, (0, 0), 1.2)
    return np.clip(img.astype(np.int16) + np.random.RandomState(3).randn(H, W, 3) * 4,
                   0, 255).astype(np.uint8)


async def do_capture(ws, frame, mode, label="高清抓拍"):
    """两级分辨率：流式帧只出判定，判定连续合格后再传一张高清帧出裁图。
    这里模拟客户端那一步：先发 {"type":"capture"}，紧跟其后的二进制帧按高清处理。"""
    import cv2 as _cv2
    h, w = frame.shape[:2]
    s = 960 / max(w, h)
    big = _cv2.resize(frame, (round(w * s), round(h * s)))
    ok, e = _cv2.imencode(".jpg", big, [_cv2.IMWRITE_JPEG_QUALITY, 92])
    data = e.tobytes()
    await ws.send(json.dumps({"type": "capture"}))
    await ws.send(data)
    while True:
        m = json.loads(await ws.recv())
        if m.get("type") == "result" and m.get("capture"):
            break
    kb = len(m.get("image", "")) * 3 / 4 / 1024 if m.get("image") else 0
    print(f"  {label}: 上传 {len(data)/1024:.0f}KB -> ok={m['ok']} reason={m['reason']}"
          + (f"  回传图 {m['image_size'][0]}x{m['image_size'][1]} {kb:.0f}KB" if m.get("image") else ""))
    return m


async def run_mode(ws, mode, frames, label):
    # 服务端是"一次性抓拍"：出过合格结果就闭锁、不再回消息。
    # 每个场景开始前先 reset 解锁，否则第二个场景就一条回复也收不到。
    await ws.send(json.dumps({"type": "reset"}))
    await ws.send(json.dumps({"type": "config", "mode": mode}))
    print(f"\n--- {label} (mode={mode}, {FPS}fps) ---")
    got, saved = [], 0

    async def sender():
        # 流式帧走小图（长边 416 / q60，约 7KB），和前端一致
        for f in frames:
            h, w = f.shape[:2]
            s = 416 / max(w, h)
            small = cv2.resize(f, (round(w * s), round(h * s)))
            _, e = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 60])
            await ws.send(e.tobytes())
            await asyncio.sleep(1 / FPS)

    send_task = asyncio.create_task(sender())
    deadline = time.time() + len(frames) / FPS + 12
    while len(got) < len(frames) and time.time() < deadline:
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=12)
        except asyncio.TimeoutError:
            break
        m = json.loads(raw)
        if m.get("type") in ("config_ok", "pong", "hello", "reset_ok"):
            continue
        got.append(m)
        img_kb = len(m.get("image", "")) * 3 / 4 / 1024 if m.get("image") else 0
        print(f"  seq={m.get('seq'):<3} ok={str(m.get('ok')):<5} {m.get('reason','-'):<13} "
              f"msg={m.get('msg','')!s:<26} {m.get('ms')}ms  上传{m.get('bytes_in',0)/1024:.0f}KB"
              + (f"  回传图 {m['image_size'][0]}x{m['image_size'][1]} {img_kb:.0f}KB" if m.get("image") else ""))
        if m.get("image"):
            p = os.path.join(OUT, f"ws_{mode}_{m['seq']}.jpg")
            with open(p, "wb") as fh:
                fh.write(base64.b64decode(m["image"].split(",", 1)[1]))
            saved += 1
    await send_task
    print(f"  收到 {len(got)}/{len(frames)} 个结果，回传图 {saved} 张")
    return got


async def main():
    ctx = None
    if URL.startswith("wss"):
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

    os.makedirs(OUT, exist_ok=True)
    card, cmask = make_card()

    async with websockets.connect(URL, ssl=ctx, max_size=8 * 1024 * 1024) as ws:
        hello = json.loads(await ws.recv())
        print("hello:", json.dumps(hello, ensure_ascii=False))

        ok_all = True

        # 1) 证件：先歪着 3 帧，再摆正 4 帧 —— 应该先提示对准，摆正后回传证件图
        frames = [shoot(card, cmask, angle=25, scale=0.9, seed=i)[0] for i in range(3)] + \
                 [shoot(card, cmask, angle=1, scale=1.35, seed=i)[0] for i in range(4)]
        got = await run_mode(ws, "card", frames, "证件：歪 3 帧 -> 正 4 帧")
        tilted = [m for m in got[:3] if not m["ok"] and m["reason"] == "tilted"]
        streak_ok = [m for m in got[3:] if m["ok"]]
        cap = await do_capture(ws, frames[-1], "card")
        accepted = [cap] if (cap["ok"] and cap.get("image")) else []
        print(f"  检查: 歪的被拒 {len(tilted)}/3, 正的流式判定合格 {len(streak_ok)}/4, "
              f"高清帧回传图 {len(accepted)}/1")
        if not tilted or not streak_ok or not accepted:
            ok_all = False
            print("  !! 不符合预期")

        # 2) 证件：斜着拍，应全部拒绝且不回传图
        frames = [shoot(card, cmask, angle=0, scale=1.35, persp=.5, seed=i)[0] for i in range(3)]
        got = await run_mode(ws, "card", frames, "证件：斜着拍 3 帧（应全拒）")
        if any(m["ok"] for m in got):
            ok_all = False
            print("  !! 斜着拍竟然被接受")
        else:
            print("  检查: 全部拒绝 ✓")

        # 3) 人脸：空画面 -> 真实人脸，应该从 no_face 变成回传人脸裁图
        blank = np.full((720, 960, 3), 80, np.uint8)
        sample = fetch_sample(*FACE_SAMPLE)
        if sample:
            src = cv2.imread(sample)
            # 原图里脸只占 0.7%，先围着脸裁一块再放大，模拟"人凑近自拍"
            fd_box = (225, 93, 31, 40)
            fx, fy, fw_, fh_ = fd_box
            cx, cy = fx + fw_ // 2, fy + fh_ // 2
            half = int(fw_ * 1.9)
            x0, y0 = max(0, cx - half), max(0, cy - int(half * 1.25))
            x1, y1 = min(src.shape[1], cx + half), min(src.shape[0], cy + int(half * 1.25))
            near = cv2.resize(src[y0:y1, x0:x1], (720, 960), interpolation=cv2.INTER_CUBIC)

            got = await run_mode(ws, "face", [blank, blank] + [near] * 4,
                                 "人脸：空 2 帧 -> 真实人脸(凑近) 4 帧")
            no_face = [m for m in got[:2] if not m["ok"] and m["reason"] == "no_face"]
            cap = await do_capture(ws, near, "face")   # 必须用人脸帧，别误用上个场景的 frames
            face_ok = [cap] if (cap["ok"] and cap.get("image")) else []
            print(f"  检查: 空画面判无人脸 {len(no_face)}/2, 高清帧回传裁图 {len(face_ok)}/1"
                  f" (前1帧防抖不回传属正常)")
            if not no_face or not face_ok:
                ok_all = False
                print("  !! 不符合预期")

            # 4) 人脸太远：整张原图里脸占比很小，应判 too_small
            far = cv2.resize(src, (960, 600), interpolation=cv2.INTER_CUBIC)
            got = await run_mode(ws, "face", [far] * 3, "人脸：太远（脸占比小，应判 too_small）")
            if any(m["ok"] for m in got):
                ok_all = False
                print("  !! 太远的脸竟然被接受")
            else:
                print(f"  检查: 全部拒绝，reason={got[0]['reason'] if got else '-'} ✓")
        else:
            got = await run_mode(ws, "face", [blank, blank], "人脸：仅空画面（无网络，跳过真人脸）")
            if not all(m["reason"] == "no_face" for m in got):
                ok_all = False

        # 手画的脸留着做对比：说明合成图不足以验证人脸检测
        cv2.imwrite(os.path.join(OUT, "ws_face_drawn.jpg"), draw_face())

        # 5) 背压：一次性猛推 12 帧，服务端应丢掉中间帧只处理最新的
        print("\n--- 背压测试：一次性推 12 帧（不等结果）---")
        f = shoot(card, cmask, angle=1, scale=1.35)[0]
        jpg = to_jpeg(f)
        await ws.send(json.dumps({"type": "config", "mode": "card"}))
        t0 = time.time()
        for _ in range(12):
            await ws.send(jpg)
        n, drops = 0, 0
        while time.time() - t0 < 8:
            try:
                m = json.loads(await asyncio.wait_for(ws.recv(), timeout=2.5))
            except asyncio.TimeoutError:
                break
            if m.get("type") != "result":
                continue
            n += 1
            drops += m.get("dropped", 0)
        print(f"  推 12 帧 -> 返回 {n} 个结果, 服务端丢弃 {drops} 帧 "
              f"({'背压生效 ✓' if n < 12 and drops > 0 else '未观察到丢帧'})")

        print("\n总体:", "通过 ✓" if ok_all else "有异常 ✗")
        return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

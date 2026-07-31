"""证件 / 人脸 WebSocket 识别服务。

协议
----
客户端 -> 服务端
  * 文本 JSON 控制帧：{"type":"config","mode":"card"|"face"}
  * 二进制帧：一张 JPEG 图（约 100KB），按 mode 处理

服务端 -> 客户端
  {"type":"hello", "backend":..., "providers":[...], "imgsz":960}
  {"type":"result","mode":"card","ok":true,"msg":"证件已对准","reason":"ok",
   "conf":0.93,"rotate_deg":1.2,"skew":0.03,"quad":[[x,y]x4],
   "image":"data:image/jpeg;base64,...","ms":38,"seq":17,"dropped":2}
  {"type":"result","mode":"card","ok":false,"msg":"证件歪了，请对准证件","reason":"tilted",...}
  {"type":"error","msg":...}

只有 ok=true 且连续稳定帧达标时才带 image 字段回传。
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from card_detector import CardPoseDetector, CardTracker, PoseRule   # noqa: E402
from face_detector import FaceDetector, FaceRule                # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB_DIR = os.path.join(ROOT, "web")

MODEL_PATH = os.environ.get("CARD_MODEL", os.path.join(ROOT, "cardpose.onnx"))
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(ROOT, "models"))
JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", "88"))
DUMP_ROOT = os.environ.get("DUMP_DIR", os.path.join(ROOT, "dumps"))

card_det: CardPoseDetector | None = None
face_det: FaceDetector | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global card_det, face_det
    t0 = time.time()
    card_det = CardPoseDetector(MODEL_PATH, PoseRule())
    face_det = FaceDetector(MODEL_DIR, FaceRule())
    print(f"[init] card providers={card_det.providers} imgsz={card_det.imgsz}")
    print(f"[init] face backend={face_det.backend}")
    print(f"[init] ready in {time.time() - t0:.1f}s")
    yield


app = FastAPI(title="cardpose-ws", lifespan=lifespan)


@app.get("/health")
def health():
    return JSONResponse({
        "ok": card_det is not None,
        "card_providers": card_det.providers if card_det else [],
        "card_imgsz": card_det.imgsz if card_det else None,
        "face_backend": face_det.backend if face_det else None,
    })


def _jpeg(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return buf.tobytes() if ok else b""


def _b64(data: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")


# ------------------------------------------------------------------ 同步处理
def process_card(bgr: np.ndarray, stable: int, tracker: CardTracker) -> tuple[dict, int]:
    r = tracker.step(bgr)          # 带 EMA 平滑 / 滞回 / 投票，见 CardTracker
    need = card_det.rule.stable_frames
    payload = {
        "mode": "card", "ok": False, "reason": r.reason, "msg": r.msg,
        "conf": round(r.conf, 3), "rotate_deg": r.rotate_deg, "skew": r.skew,
        "area_ratio": r.area_ratio, "aspect": r.aspect, "quad": r.quad,
        "raw_quad": r.raw_quad, "jitter": r.jitter, "held": r.held, "votes": r.votes,
    }
    if not r.ok:
        return payload, 0

    stable += 1
    payload["stable"] = f"{stable}/{need}"
    if stable < need:
        payload["reason"] = "unstable"
        payload["msg"] = "保持不动…"
        return payload, stable

    card = CardPoseDetector.warp(bgr, r.quad)
    payload["ok"] = True
    payload["image"] = _b64(_jpeg(card))
    payload["image_size"] = [card.shape[1], card.shape[0]]
    return payload, stable


def process_face(bgr: np.ndarray, stable: int) -> tuple[dict, int]:
    r = face_det.judge(bgr)
    need = face_det.rule.stable_frames
    payload = {
        "mode": "face", "ok": False, "reason": r.reason, "msg": r.msg,
        "conf": r.conf, "box": r.box, "landmarks": r.landmarks,
        "roll_deg": r.roll_deg, "yaw_ratio": r.yaw_ratio,
        "area_ratio": r.area_ratio, "count": r.count,
    }
    if not r.ok:
        return payload, 0

    stable += 1
    payload["stable"] = f"{stable}/{need}"
    if stable < need:
        payload["reason"] = "unstable"
        payload["msg"] = "保持不动…"
        return payload, stable

    crop = face_det.crop(bgr, r.box)
    payload["ok"] = True
    payload["image"] = _b64(_jpeg(crop))
    payload["image_size"] = [crop.shape[1], crop.shape[0]]
    return payload, stable


# ------------------------------------------------------------------ WebSocket
class Conn:
    """每个连接一个：最新帧覆盖旧帧（latest-wins），避免推理跟不上时排队堆积。"""

    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.mode = "card"
        self.pending: bytes | None = None
        self.event = asyncio.Event()
        self.closed = False
        self.seq = 0
        self.dropped = 0
        self.stable_card = 0
        self.stable_face = 0
        self.tracker = CardTracker(card_det)   # 时域平滑是有状态的，必须每连接一份
        self.rec_left = 0                      # 还要录几帧
        self.rec_dir: str | None = None

    def submit(self, data: bytes):
        if self.pending is not None:
            self.dropped += 1          # 上一帧还没处理完，丢掉它
        self.pending = data
        self.event.set()

    async def worker(self):
        loop = asyncio.get_running_loop()
        while not self.closed:
            await self.event.wait()
            self.event.clear()
            data, self.pending = self.pending, None
            if data is None:
                continue

            self.seq += 1
            seq, dropped = self.seq, self.dropped
            self.dropped = 0
            mode = self.mode
            t0 = time.perf_counter()
            try:
                bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if bgr is None:
                    raise ValueError("JPEG 解码失败")
                if mode == "face":
                    payload, self.stable_face = await loop.run_in_executor(
                        None, process_face, bgr, self.stable_face)
                else:
                    payload, self.stable_card = await loop.run_in_executor(
                        None, process_card, bgr, self.stable_card, self.tracker)
                payload.update(type="result", seq=seq, dropped=dropped,
                               ms=round((time.perf_counter() - t0) * 1000, 1),
                               frame_size=[bgr.shape[1], bgr.shape[0]],
                               bytes_in=len(data))
            except Exception as e:                                   # noqa: BLE001
                payload = {"type": "error", "mode": mode, "seq": seq, "msg": str(e)}

            if self.rec_left > 0 and self.rec_dir:
                try:
                    self._dump(data, payload, seq)
                except Exception as e:                               # noqa: BLE001
                    print("[rec] 写盘失败:", e)
                self.rec_left -= 1
                if self.rec_left == 0:
                    payload["rec_done"] = self.rec_dir

            try:
                await self.ws.send_text(json.dumps(payload, ensure_ascii=False))
            except Exception:                                        # noqa: BLE001
                self.closed = True
                return

    def _dump(self, data: bytes, payload: dict, seq: int):
        """把原始上传帧和判定结果落盘，用于拿真机数据离线复现问题。"""
        with open(os.path.join(self.rec_dir, f"{seq:05d}.jpg"), "wb") as f:
            f.write(data)
        meta = {k: v for k, v in payload.items() if k != "image"}   # 图别写进 jsonl
        with open(os.path.join(self.rec_dir, "results.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    conn = Conn(ws)
    task = asyncio.create_task(conn.worker())
    await ws.send_text(json.dumps({
        "type": "hello", "mode": conn.mode,
        "card_providers": card_det.providers, "imgsz": card_det.imgsz,
        "face_backend": face_det.backend,
        "card_stable_frames": card_det.rule.stable_frames,
        "face_stable_frames": face_det.rule.stable_frames,
    }, ensure_ascii=False))

    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("bytes") is not None:
                conn.submit(msg["bytes"])
            elif msg.get("text"):
                try:
                    cmd = json.loads(msg["text"])
                except json.JSONDecodeError:
                    continue
                if cmd.get("type") == "config":
                    m = cmd.get("mode")
                    if m in ("card", "face") and m != conn.mode:
                        conn.mode = m
                        conn.stable_card = conn.stable_face = 0
                        conn.tracker.reset()
                        await ws.send_text(json.dumps(
                            {"type": "config_ok", "mode": m}, ensure_ascii=False))
                elif cmd.get("type") == "record":
                    n = max(1, min(int(cmd.get("n", 60)), 600))
                    d = os.path.join(DUMP_ROOT, datetime.now().strftime("%Y%m%d_%H%M%S"))
                    os.makedirs(d, exist_ok=True)
                    conn.rec_dir, conn.rec_left = d, n
                    print(f"[rec] 开始录制 {n} 帧 -> {d}")
                    await ws.send_text(json.dumps(
                        {"type": "record_started", "n": n, "dir": d}, ensure_ascii=False))
                elif cmd.get("type") == "reset":
                    conn.stable_card = conn.stable_face = 0
                    conn.tracker.reset()
                elif cmd.get("type") == "ping":
                    await ws.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        pass
    finally:
        conn.closed = True
        conn.event.set()
        task.cancel()


# ------------------------------------------------------------------ 静态页面
@app.get("/")
def index():
    return FileResponse(os.path.join(WEB_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


if __name__ == "__main__":
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--certfile", default=None)
    ap.add_argument("--keyfile", default=None)
    a = ap.parse_args()

    kw = {}
    if a.certfile and a.keyfile:
        kw["ssl_certfile"] = a.certfile
        kw["ssl_keyfile"] = a.keyfile
    uvicorn.run(app, host=a.host, port=a.port, ws_max_size=8 * 1024 * 1024, **kw)

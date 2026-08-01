"""人脸检测：优先 YuNet(ONNX，准且带 5 个关键点)，缺模型时退回 OpenCV Haar。

判定：检测到人脸且满足"够大 / 够正 / 在画面内"就算合格，合格时回传人脸裁图。
"""
from __future__ import annotations

import math
import os
import threading
from dataclasses import dataclass, field

import cv2
import numpy as np

YUNET_FILES = (
    "face_detection_yunet_2023mar.onnx",
    "face_detection_yunet_2022mar.onnx",
)


@dataclass
class FaceRule:
    conf_thres: float = 0.75
    min_area_ratio: float = 0.02      # 人脸面积 / 画面面积 下限（太远）
    max_area_ratio: float = 0.85
    max_roll_deg: float = 20.0        # 双眼连线倾角上限（歪头），仅 YuNet 可用
    max_yaw_ratio: float = 0.30       # 鼻子相对双眼中点的水平偏移比例（侧脸）
    margin_ratio: float = 0.0         # 人脸框允许贴边的比例
    crop_expand: float = 0.35         # 裁人脸时向外扩张的比例
    stable_frames: int = 2
    # "清晰"闸门：把人脸区域缩到 128x128 后算拉普拉斯方差，越小越糊。
    #
    # 默认取 20，只挡明显不可用的帧。这个值是照着实测曲线定的
    # （样图是放大裁切的人脸，本身偏软，所以绝对值偏低）：
    #     原图 44.5 | 高斯σ=2 25.1 | σ=4 16.6 | σ=8 8.0 | 运动模糊k=15 25.9
    # 真机清晰自拍通常远高于这个范围（上百），所以 15 几乎不会误挡，
    # 但能拦住严重失焦/大幅手抖的帧。
    #
    # 每帧实测值都会在 sharp 字段回传，前端状态表也显示。真机跑一轮看看
    # 清晰帧和糊帧各是多少，再用 FACE_MIN_SHARP 调到两者之间即可。
    #
    # 逃生开关：真机上如果一直卡在"画面不清晰"，先 FACE_MIN_SHARP=0 关掉这道闸门，
    # 记下状态表里清晰帧的实际数值，再设成它的一半左右。
    min_sharpness: float = float(os.environ.get("FACE_MIN_SHARP", "20"))


REASON_TEXT = {
    "no_face": "未检测到人脸，请把脸放入框内",
    "multi_face": "画面中有多张人脸，请只保留一人",
    "too_small": "请靠近一点",
    "too_large": "请稍微拉远一点",
    "out_of_frame": "人脸超出画面，请居中",
    "rolled": "请把头摆正",
    "yawed": "请正对镜头，不要侧脸",
    "blurry": "画面不清晰，请拿稳手机等对焦完成",
}


@dataclass
class FaceResult:
    ok: bool
    reason: str
    msg: str
    conf: float = 0.0
    box: list[float] = field(default_factory=list)      # x, y, w, h
    landmarks: list[list[float]] = field(default_factory=list)
    roll_deg: float = 0.0
    yaw_ratio: float = 0.0
    area_ratio: float = 0.0
    count: int = 0
    sharp: float = 0.0        # 人脸区域拉普拉斯方差，越大越清晰
    face_jpeg: bytes | None = None
    stable: int = 0


class FaceDetector:
    """线程安全的人脸检测器。

    重要：cv2.FaceDetectorYN 和 cv2.CascadeClassifier 的实例都【不是线程安全的】。
    实测（8 线程并发调用同一个 FaceDetectorYN 实例，即使输入尺寸完全相同）会在
    OpenCV 原生层直接崩溃，连 Python traceback 都没有，整个服务进程一起挂掉。
    而单线程跑同样的负载完全正常。

    本服务里 FaceDetector 是全局单例、被 run_in_executor 的线程池并发调用，
    所以这里用 threading.local() 给每个工作线程各自持有一份实例。
    模型只有 232KB，线程池上限 min(32, cpu+4)，多占几 MB 换来无锁并行是值得的。
    """

    def __init__(self, model_dir: str = "models", rule: FaceRule | None = None):
        self.rule = rule or FaceRule()
        self._local = threading.local()

        # 先在主线程试建一次，确定用哪个后端、并让加载失败在启动时就暴露
        self._yunet_path: str | None = None
        for name in YUNET_FILES:
            p = os.path.join(model_dir, name)
            if os.path.exists(p):
                try:
                    cv2.FaceDetectorYN.create(
                        p, "", (320, 320),
                        score_threshold=self.rule.conf_thres, nms_threshold=0.3, top_k=50,
                    )
                    self._yunet_path = p
                    break
                except cv2.error:
                    self._yunet_path = None

        self.backend = "yunet" if self._yunet_path else "haar"
        self._haar_xml = os.path.join(cv2.data.haarcascades,
                                      "haarcascade_frontalface_default.xml")
        if self.backend == "haar" and cv2.CascadeClassifier(self._haar_xml).empty():
            raise RuntimeError("Haar cascade 加载失败")

    # ---------- 每线程各持一份检测器 ----------
    def _yunet(self):
        d = getattr(self._local, "yunet", None)
        if d is None:
            d = cv2.FaceDetectorYN.create(
                self._yunet_path, "", (320, 320),
                score_threshold=self.rule.conf_thres, nms_threshold=0.3, top_k=50,
            )
            self._local.yunet = d
            self._local.size = (0, 0)
        return d

    def _haar(self):
        d = getattr(self._local, "haar", None)
        if d is None:
            d = cv2.CascadeClassifier(self._haar_xml)
            self._local.haar = d
        return d

    # ---------- 原始检测 ----------
    def _detect(self, bgr: np.ndarray):
        """返回 [(x, y, w, h, score, landmarks_or_None), ...]"""
        H, W = bgr.shape[:2]
        if self.backend == "yunet":
            det = self._yunet()
            if getattr(self._local, "size", None) != (W, H):
                det.setInputSize((W, H))
                self._local.size = (W, H)
            _, faces = det.detect(bgr)
            if faces is None:
                return []
            out = []
            for f in faces:
                # 显式转 Python float：numpy 标量虽然能被 json 序列化
                # （np.float64 是 float 的子类），但整数类不是，统一转掉更保险
                x, y, w, h = (float(v) for v in f[:4])
                lm = f[4:14].reshape(5, 2).astype(float).tolist()
                out.append((x, y, w, h, float(f[14]), lm))
            return out

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        rects = self._haar().detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5,
            minSize=(int(min(W, H) * 0.12), int(min(W, H) * 0.12)),
        )
        # Haar 无分数，用 0.9 占位
        return [(float(x), float(y), float(w), float(h), 0.9, None) for x, y, w, h in rects]

    # ---------- 判定 ----------
    def judge(self, bgr: np.ndarray) -> FaceResult:
        H, W = bgr.shape[:2]
        R = self.rule
        faces = self._detect(bgr)
        if not faces:
            return FaceResult(False, "no_face", REASON_TEXT["no_face"])

        faces.sort(key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h, score, lm = faces[0]
        box = [round(float(x), 1), round(float(y), 1), round(float(w), 1), round(float(h), 1)]
        area_ratio = float(w * h) / float(W * H)

        roll = 0.0
        yaw = 0.0
        if lm is not None:
            (rx, ry), (lx, ly), (nx, ny) = lm[0], lm[1], lm[2]
            roll = math.degrees(math.atan2(ly - ry, lx - rx))
            eye_cx, eye_dist = (rx + lx) / 2, max(abs(lx - rx), 1e-6)
            yaw = abs(nx - eye_cx) / eye_dist

        # 清晰度：只在人脸框内算，避免背景的高频纹理把分数抬上去。
        # 归一化到脸的尺寸之后再算，否则脸越大方差越高，阈值就没法固定。
        fx0, fy0 = max(0, int(x)), max(0, int(y))
        fx1, fy1 = min(W, int(x + w)), min(H, int(y + h))
        sharp = 0.0
        if fx1 - fx0 > 8 and fy1 - fy0 > 8:
            face = cv2.cvtColor(bgr[fy0:fy1, fx0:fx1], cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face, (128, 128), interpolation=cv2.INTER_AREA)
            sharp = float(cv2.Laplacian(face, cv2.CV_64F).var())

        def res(ok, reason):
            return FaceResult(ok, reason, REASON_TEXT.get(reason, "请对准人脸"),
                              conf=round(score, 3), box=box,
                              landmarks=[[round(a, 1), round(b, 1)] for a, b in (lm or [])],
                              roll_deg=round(roll, 2), yaw_ratio=round(yaw, 3),
                              area_ratio=round(area_ratio, 4), count=len(faces),
                              sharp=round(sharp, 1))

        mx, my = W * R.margin_ratio, H * R.margin_ratio
        if x < -mx or y < -my or x + w > W + mx or y + h > H + my:
            return res(False, "out_of_frame")
        if area_ratio < R.min_area_ratio:
            return res(False, "too_small")
        if area_ratio > R.max_area_ratio:
            return res(False, "too_large")
        if lm is not None:
            if abs(roll) > R.max_roll_deg:
                return res(False, "rolled")
            if yaw > R.max_yaw_ratio:
                return res(False, "yawed")
        if sharp < R.min_sharpness:
            return res(False, "blurry")

        r = res(True, "ok")
        r.msg = "检测到人脸"
        return r

    # ---------- 裁剪 ----------
    def crop(self, bgr: np.ndarray, box: list[float]) -> np.ndarray:
        H, W = bgr.shape[:2]
        x, y, w, h = box
        e = self.rule.crop_expand
        x0 = int(max(0, x - w * e))
        y0 = int(max(0, y - h * e))
        x1 = int(min(W, x + w * (1 + e)))
        y1 = int(min(H, y + h * (1 + e)))
        return bgr[y0:y1, x0:x1].copy()

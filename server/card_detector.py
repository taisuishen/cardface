"""证件姿态检测：YOLO11s-pose (1 class 'card', 4 keypoints = 四个角点)。

模型输出 [1, 13, 18900]，转置后每行：
  0..3  cx, cy, w, h      (letterbox 后的 960x960 像素坐标)
  4     class conf
  5..12 kx0,ky0 … kx3,ky3 (同上坐标系)

关键点顺序是"语义固定"的 —— 经实测（0/15/30/60/90/180/-30 度全部一致）：
  kpt0 = 证件自身左下  kpt1 = 左上  kpt2 = 右上  kpt3 = 右下
所以用固定置换 KPT_PERM=[1,2,3,0] 就能得到 [左上, 右上, 右下, 左下]，
不需要靠几何位置猜角点 —— 这一点很重要：几何猜法在证件旋转超过 45°
时会把"左边"错认成"上边"，导致 30°/90° 的歪证件被量成 0° 而误判合格。

判定：只有"正对镜头、放平、够大、没出框"的证件才算合格，
合格时返回透视矫正后的证件图；否则返回不合格原因。
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import cv2
import numpy as np
import onnxruntime as ort

# 模型关键点 -> [TL, TR, BR, BL]。若换了模型且角点顺序不同，可用环境变量覆盖。
KPT_PERM = [int(v) for v in os.environ.get("KPT_PERM", "1,2,3,0").split(",")]


def _envf(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


# ---------------------------------------------------------------- 判定阈值
@dataclass
class PoseRule:
    conf_thres: float = _envf("CARD_CONF", 0.45)        # 检测置信度下限
    nms_iou: float = 0.5
    max_rotate_deg: float = _envf("MAX_ROTATE", 8.0)    # 画面内旋转（歪）上限，度
    upside_down_deg: float = 135.0                      # 超过此角度认为证件倒置
    max_skew: float = _envf("MAX_SKEW", 0.14)           # 透视（斜）上限：对边长度差/均值
    max_corner_err_deg: float = _envf("MAX_CORNER_ERR", 12.0)   # 四角与 90° 的最大偏差
    min_area_ratio: float = _envf("MIN_AREA", 0.12)     # 证件/画面 面积下限（太远）
    max_area_ratio: float = _envf("MAX_AREA", 0.92)     # 上限（太近）
    margin_ratio: float = _envf("MARGIN", 0.01)         # 角点允许超出画面的比例
    border_ratio: float = _envf("BORDER", 0.04)         # 角点"贴边"判定比例，见 judge()
    min_aspect: float = 1.30                            # 矫正后宽高比区间（身份证≈1.585）
    max_aspect: float = 1.90


CARD_W, CARD_H = 856, 540             # 输出证件图尺寸（ID-1 标准 85.6x54mm）
CARD_PAD = _envf("CARD_PAD", 0.03)    # 裁剪时四周多留一圈，只比证件本身大一点

REASON_TEXT = {
    "no_card": "No card detected. Place your card inside the frame",
    "tilted": "Card is tilted. Align it with the frame",
    "upside_down": "Card is the wrong way up. Turn it upright",
    "skewed": "Face the camera straight on, do not shoot at an angle",
    "not_rect": "Lay the card flat so all four corners are clear",
    "too_small": "Move closer so the card fills the frame",
    "too_large": "Move back a little",
    "out_of_frame": "Card is out of frame. Align it with the frame",
    "bad_aspect": "Card not recognized. Please align it again",
    "unstable": "Hold still...",
}


# ---------------------------------------------------------------- 几何工具
def _letterbox(img: np.ndarray, size: int = 960):
    """等比缩放 + 灰边填充到 size x size，返回图、缩放比、左上偏移。"""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    dx, dy = (size - nw) // 2, (size - nh) // 2
    canvas[dy:dy + nh, dx:dx + nw] = resized
    return canvas, r, dx, dy


def _rotation_deg(tl: np.ndarray, tr: np.ndarray) -> float:
    """证件上边相对水平线的夹角，范围 (-180, 180]。
    因为角点是语义固定的，这里不做 ±45° 折叠 —— 折叠会让 90° 的歪证件量成 0°。
    正值 = 画面里顺时针歪（图像 y 轴向下）。"""
    return math.degrees(math.atan2(tr[1] - tl[1], tr[0] - tl[0]))


def _corner_angles(q: np.ndarray) -> list[float]:
    out = []
    for i in range(4):
        prev, cur, nxt = q[i - 1], q[i], q[(i + 1) % 4]
        v1, v2 = prev - cur, nxt - cur
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            out.append(0.0)
            continue
        cosv = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1))
        out.append(math.degrees(math.acos(cosv)))
    return out


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> list[int]:
    idxs = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 0.0, iou_thres)
    if len(idxs) == 0:
        return []
    return np.array(idxs).flatten().tolist()


# ---------------------------------------------------------------- 结果
@dataclass
class CardResult:
    ok: bool
    reason: str
    msg: str
    conf: float = 0.0
    rotate_deg: float = 0.0
    skew: float = 0.0
    corner_err: float = 0.0
    area_ratio: float = 0.0
    aspect: float = 0.0
    quad: list[list[float]] = field(default_factory=list)   # 原图坐标 [TL,TR,BR,BL]
    card_jpeg: bytes | None = None
    stable: int = 0
    raw_quad: list[list[float]] = field(default_factory=list)  # 未平滑的原始角点（调试用）
    held: bool = False        # 本帧漏检，用的是上一帧平滑结果
    jitter: float = 0.0       # 本帧原始角点相对平滑值的平均偏移(px)，反映抖动大小
    votes: str = ""           # 投票窗口状态，如 "3/4"


class CardPoseDetector:
    def __init__(self, model_path: str, rule: PoseRule | None = None,
                 providers: list[str] | None = None, threads: int = 0):
        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads:
            so.intra_op_num_threads = threads
        if providers is None:
            avail = ort.get_available_providers()
            providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                         if p in avail] or ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, so, providers=providers)
        self.providers = self.session.get_providers()
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.imgsz = int(inp.shape[2]) if isinstance(inp.shape[2], int) else 960
        self.rule = rule or PoseRule()
        # 预热，避免第一帧特别慢
        self.session.run(None, {self.input_name:
                                np.zeros((1, 3, self.imgsz, self.imgsz), np.float32)})

    # ---------- 推理 ----------
    def _infer(self, bgr: np.ndarray):
        """返回 ([TL,TR,BR,BL] 原图坐标, conf)，没检到则 (None, 0)。"""
        lb, r, dx, dy = _letterbox(bgr, self.imgsz)
        blob = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        blob = np.ascontiguousarray(blob, dtype=np.float32) / 255.0
        out = self.session.run(None, {self.input_name: blob[None]})[0]  # (1,13,N)
        pred = out[0].T                                                 # (N,13)

        keep = pred[:, 4] >= self.rule.conf_thres
        pred = pred[keep]
        if len(pred) == 0:
            return None, 0.0

        cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        boxes = np.stack([cx - w / 2, cy - h / 2, w, h], axis=1)
        idxs = _nms(boxes, pred[:, 4], self.rule.nms_iou)
        if not idxs:
            return None, 0.0
        best = pred[idxs[int(np.argmax(pred[idxs, 4]))]]

        kpts = best[5:13].reshape(4, 2).astype(np.float64)[KPT_PERM]
        kpts[:, 0] = (kpts[:, 0] - dx) / r          # 还原到原图坐标
        kpts[:, 1] = (kpts[:, 1] - dy) / r
        return kpts, float(best[4])

    # ---------- 判定 ----------
    def judge(self, bgr: np.ndarray) -> CardResult:
        """单帧无状态判定。需要抗抖动请用 CardTracker。"""
        H, W = bgr.shape[:2]
        q, conf = self._infer(bgr)
        if q is None:
            return CardResult(False, "no_card", REASON_TEXT["no_card"])
        return self.judge_quad(q, conf, W, H)

    def judge_quad(self, q: np.ndarray, conf: float, W: int, H: int,
                   relax: float = 1.0) -> CardResult:
        """对给定的四角做几何判定。

        relax > 1 表示放宽姿态阈值 —— 用于滞回：已经判合格之后要"更难"被推翻，
        否则真实角度刚好落在阈值附近时，噪声会让判定在合格/不合格之间反复翻转。
        """
        R = self.rule
        qlist = [[round(float(x), 1), round(float(y), 1)] for x, y in q]

        # 先把所有指标算全，再决定不合格原因 —— 这样前端始终能拿到完整诊断数据
        top = float(np.linalg.norm(q[1] - q[0]))
        right = float(np.linalg.norm(q[2] - q[1]))
        bottom = float(np.linalg.norm(q[3] - q[2]))
        left = float(np.linalg.norm(q[0] - q[3]))

        rot = _rotation_deg(q[0], q[1])
        skew = max(abs(top - bottom) / max((top + bottom) / 2, 1e-6),
                   abs(left - right) / max((left + right) / 2, 1e-6))
        corner_err = max(abs(a - 90) for a in _corner_angles(q))
        area_ratio = float(cv2.contourArea(q.astype(np.float32))) / float(W * H)
        aspect = ((top + bottom) / 2) / max((left + right) / 2, 1e-6)

        mx, my = W * R.margin_ratio, H * R.margin_ratio
        out_of_frame = bool((q[:, 0] < -mx).any() or (q[:, 0] > W + mx).any() or
                            (q[:, 1] < -my).any() or (q[:, 1] > H + my).any())

        # 证件被画面边缘裁掉时，模型不会把角点预测到画面外，而是沿着边缘
        # 给出一个"压扁"的四边形（实测 aspect 1.03 vs 正常 1.51）。
        # 所以真正的判据是宽高比异常；再结合"角点贴边"来区分是被边缘裁掉
        # 还是在画面中间被遮挡。
        bx, by = W * R.border_ratio, H * R.border_ratio
        near_border = bool((q[:, 0] < bx).any() or (q[:, 0] > W - bx).any() or
                           (q[:, 1] < by).any() or (q[:, 1] > H - by).any())
        bad_aspect = not (R.min_aspect <= aspect <= R.max_aspect)

        def res(ok: bool, reason: str) -> CardResult:
            return CardResult(ok, reason, REASON_TEXT.get(reason, "Align the card with the frame"),
                              conf=conf, rotate_deg=round(rot, 2), skew=round(skew, 3),
                              corner_err=round(corner_err, 2),
                              area_ratio=round(area_ratio, 3), aspect=round(aspect, 3),
                              quad=qlist)

        if min(top, right, bottom, left) < 8:
            return res(False, "bad_aspect")

        # 优先级：姿态问题(歪/倒/斜) 先报，因为歪着的证件往往同时超出画面，
        # 这时提示"证件歪了"比提示"超出画面"更有用。
        if abs(rot) > R.upside_down_deg:
            return res(False, "upside_down")
        if abs(rot) > R.max_rotate_deg * relax:
            return res(False, "tilted")
        if skew > R.max_skew * relax:
            return res(False, "skewed")
        if corner_err > R.max_corner_err_deg * relax:
            return res(False, "not_rect")
        if area_ratio < R.min_area_ratio / relax:
            return res(False, "too_small")
        if area_ratio > min(R.max_area_ratio * relax, 0.99):
            return res(False, "too_large")
        if out_of_frame:
            return res(False, "out_of_frame")
        if bad_aspect:
            return res(False, "out_of_frame" if near_border else "bad_aspect")

        r = res(True, "ok")
        r.msg = "Card aligned"
        return r

    # ---------- 矫正裁剪 ----------
    @staticmethod
    def warp(bgr: np.ndarray, quad: list[list[float]],
             out_w: int = CARD_W, out_h: int = CARD_H,
             pad: float | None = None) -> np.ndarray:
        """透视矫正并裁出证件。

        pad 是四周多留的边（相对证件边长的比例，默认 3%）：证件本体被摆放在
        画布正中，只比证件本身大一圈，其余背景全部截掉。
        """
        if pad is None:
            pad = CARD_PAD
        px, py = int(round(out_w * pad)), int(round(out_h * pad))
        cw, ch = out_w + 2 * px, out_h + 2 * py

        src = np.array(quad, dtype=np.float32)
        dst = np.array([[px, py], [px + out_w - 1, py],
                        [px + out_w - 1, py + out_h - 1], [px, py + out_h - 1]],
                       dtype=np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(bgr, M, (cw, ch), flags=cv2.INTER_CUBIC,
                                   borderMode=cv2.BORDER_REPLICATE)


# ---------------------------------------------------------------- 时域稳定
@dataclass
class SmoothRule:
    """抗抖动参数。抖动有两种来源，这里分别对付：

    1) 角点本身逐帧微动  -> EMA 平滑（画面上的框不再乱跳）
    2) 真实姿态刚好卡在阈值上 -> 滞回 + 投票（判定结论不再反复翻转）
    """
    ema: float = _envf("SMOOTH_EMA", 0.45)          # 新帧权重，越小越平滑（0=不更新）
    reset_px: float = _envf("SMOOTH_RESET_PX", 90)  # 单帧跳动超过此值视为换了目标，直接重置
    hyst: float = _envf("SMOOTH_HYST", 1.35)        # 已合格后阈值放宽倍数（滞回）
    vote_win: int = int(_envf("VOTE_WIN", 4))       # 投票窗口长度
    vote_need: int = int(_envf("VOTE_NEED", 3))     # 窗口内至少几帧合格才算合格
    miss_keep: int = int(_envf("MISS_KEEP", 2))     # 允许连续漏检几帧仍沿用上次平滑结果


class CardTracker:
    """带时域稳定的证件检测器。每个 WebSocket 连接独立持有一个实例。

    注意：平滑只用于"画框"和"判定"，最终回传的矫正图仍然用平滑后的角点去
    warp —— 平滑值比单帧值更接近真实角点位置，裁出来的图也更稳。
    """

    def __init__(self, det: CardPoseDetector, smooth: SmoothRule | None = None):
        self.det = det
        self.s = smooth or SmoothRule()
        self.reset()

    def reset(self):
        self.sq: np.ndarray | None = None    # 平滑后的四角
        self.miss = 0
        self.ok_state = False
        self.votes: list[bool] = []

    def step(self, bgr: np.ndarray) -> CardResult:
        """判定用【当前帧的原始角点】，画框用【平滑角点】。

        为什么必须分开 —— 平滑角点混了前几帧的位置，证件一移动就系统性落后：
        实测证件持续平移时，平滑角点只覆盖真实证件的 94.5%（偏移 26px，
        且随移动持续变大到 37px），而原始角点覆盖 97%（偏移 17px）。
        缺的那 5% 就是"回传的证件不全"。证件静止时两者完全一致（98.6%），
        所以平滑对画框有益、对裁图有害。

        判定也必须跟着用原始角点：否则会出现"拿平滑值判合格、拿原始值裁图"
        的错配，而且平滑还能把不合格的帧救成合格（EMA 会把斜证件的 skew
        从 0.25 拉到 0.11）。防抖交给投票窗口做，不靠放宽阈值。
        """
        H, W = bgr.shape[:2]
        q, conf = self.det._infer(bgr)
        jitter = 0.0
        raw = [] if q is None else [[round(float(x), 1), round(float(y), 1)] for x, y in q]

        if q is None:
            # 偶尔漏检：沿用上次平滑结果只为让框不闪，绝不判合格
            self.miss += 1
            if self.sq is None or self.miss > self.s.miss_keep:
                self.reset()
                return CardResult(False, "no_card", REASON_TEXT["no_card"])
            r = self.det.judge_quad(self.sq, 0.0, W, H)
            r.ok = False
            if r.reason == "ok":
                r.reason, r.msg = "unstable", "Hold still..."
            self.votes.append(False)
            if len(self.votes) > self.s.vote_win:
                self.votes.pop(0)
            self.ok_state = False
            r.quad = [[round(float(x), 1), round(float(y), 1)] for x, y in self.sq]
            r.raw_quad = []
            r.held = True
            r.votes = f"{sum(self.votes)}/{len(self.votes)}"
            return r

        self.miss = 0

        # ---- 严格闸门：用未平滑的原始角点判定，不放宽阈值 ----
        r = self.det.judge_quad(q, conf, W, H, relax=1.0)
        frame_ok = r.ok

        # ---- 平滑只用于画框 ----
        if self.sq is None:
            self.sq = q.copy()
        else:
            jitter = float(np.linalg.norm(q - self.sq, axis=1).mean())
            if jitter > self.s.reset_px:
                self.sq = q.copy()          # 换了位置/换了一张，别把两个位置平均
                self.votes.clear()
            else:
                a = self.s.ema
                self.sq = a * q + (1 - a) * self.sq

        # ---- 投票：窗口内够票 且 本帧自己也过，才对外宣布合格 ----
        self.votes.append(frame_ok)
        if len(self.votes) > self.s.vote_win:
            self.votes.pop(0)
        passed = sum(self.votes)
        voted_ok = frame_ok and passed >= min(self.s.vote_need, len(self.votes))
        self.ok_state = voted_ok

        r.ok = voted_ok
        if not voted_ok and r.reason == "ok":
            r.reason, r.msg = "unstable", "Hold still..."
        r.quad = [[round(float(x), 1), round(float(y), 1)] for x, y in self.sq]  # 画框用
        r.raw_quad = raw                                                        # 裁图用
        r.held = False
        r.jitter = round(jitter, 1)
        r.votes = f"{passed}/{len(self.votes)}"
        return r

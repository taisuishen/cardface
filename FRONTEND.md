# 前端对接文档

证件 / 人脸识别 WebSocket 服务。**两级分辨率 + 一次性抓拍**：

1. 持续推**小图**（约 7KB）做姿态判定和实时提示，服务端只回判定，**不回图**
2. 客户端数到**连续 N 帧合格**（证件 3 帧 / 人脸 1 帧）后，自动补传**一张高清帧**（约 100KB）
3. 服务端对高清帧做严格判定，通过就回裁好的图并**闭锁**，之后不再处理任何帧

这样设计的原因：判定只需要粗特征（实测长边 416 与 960 的角点误差是 17.8px vs 18.2px，
几乎无差别），而最终裁图是从上传帧透视矫正来的、必须清晰。全程用高清帧是浪费，
全程用小图则最终裁图不够清晰。分开之后上行流量降了 90%+。

---

## 1. 连接

### 当前环境地址

```
WS      wss://tr05uksdg3cpmu-8000.proxy.runpod.net/ws
健康检查 https://tr05uksdg3cpmu-8000.proxy.runpod.net/health
负载状态 https://tr05uksdg3cpmu-8000.proxy.runpod.net/stats
调试页面 https://tr05uksdg3cpmu-8000.proxy.runpod.net/
```

```js
const WS_URL = 'wss://tr05uksdg3cpmu-8000.proxy.runpod.net/ws';
const ws = new WebSocket(WS_URL);
ws.binaryType = 'arraybuffer';
```

RunPod 的这个域名是**受信任证书**，手机可以直接开，不用处理证书告警。

> **地址会变。** `tr05uksdg3cpmu` 是 Pod ID —— 停机再开不变，但**重建 Pod 会换成新的**。
> 所以别把它写死在代码里，用环境变量 / 配置项注入。
> 拿法：Pod 详情页 → Connect → HTTP Services 里 8000 那条。

### 本地联调

服务自带调试页面，同源时可以直接推导地址：

```js
const proto = location.protocol === 'https:' ? 'wss' : 'ws';
const WS_URL = proto + '://' + location.host + '/ws';
```

本机跑服务：`python server/app.py --host 0.0.0.0 --port 8000`
→ `ws://localhost:8000/ws`

### 其它

| | |
|---|---|
| 心跳 | 建议每 20s 发 `{"type":"ping"}`，反向代理会掐掉空闲连接 |
| 跨域 | WebSocket 不受 CORS 限制，前端在任何域名下都能连 |
| 首次启动 | 冷启动约 100 秒（CUDA 初始化 + 模型预热），期间 RunPod 代理返回 **502**，属正常 |

### 关闭码

| code | 含义 | 前端该做什么 |
|---|---|---|
| `4429` | 服务并发已满（默认上限 20） | 提示稍后重试，不要立刻重连风暴 |
| 其他 | 正常断开 / 网络问题 | 按需重连 |

---

## 2. 上传的帧

两种帧都是**二进制 JPEG**，直接 `ws.send(arrayBuffer)`，不要 base64（会多 33% 流量）。

| | 流式帧 | 高清抓拍帧 |
|---|---|---|
| 长边 | **416 px** | **960 px** |
| JPEG 质量 | 0.60 | 0.82 |
| 大小 | 约 **7 KB** | 约 **70~110 KB** |
| 频率 | 3 fps 持续 | 整个流程**只传一次** |
| 发送方式 | 直接发二进制 | 先发 `{"type":"capture"}`，紧跟二进制 |
| 服务端返回 | 判定 + 画框坐标，**无图** | 判定 + **裁好的图** + `final:true` |

```js
const STREAM_LONG = 416, STREAM_Q = 0.60;
const CAP_LONG    = 960, CAP_Q    = 0.82;
const CAP_MAX_KB  = 110;          // 高清帧体积上限
const cap = document.createElement('canvas'), cctx = cap.getContext('2d');

function grabAt(video, longSide, q){
  const vw = video.videoWidth, vh = video.videoHeight;
  if (!vw) return null;
  const s = longSide / Math.max(vw, vh);
  cap.width  = Math.round(vw * s);
  cap.height = Math.round(vh * s);
  cctx.drawImage(video, 0, 0, cap.width, cap.height);
  return new Promise(r => cap.toBlob(r, 'image/jpeg', q));
}

// 高清帧带体积上限：证件的细纹/防伪底纹很多，同质量下比普通照片大得多
// （真机实测 q92 到过 177KB）。只降质量不降分辨率 —— 分辨率直接决定最终裁图清晰度。
async function grabCapture(video){
  let blob = null;
  for (let q = CAP_Q; q >= 0.5; q -= 0.12){
    blob = await grabAt(video, CAP_LONG, q);
    if (!blob || blob.size / 1024 <= CAP_MAX_KB) break;
  }
  return blob;
}
```

### 两条必须遵守的规则

**① 必须持续读取回传消息。** 只发不读会把服务端的发送缓冲堵死，连接会被断开
（实测：8 条只发不读的连接，本该 8 秒的任务卡了 37.9 秒并报 `ConnectionClosedError`）。
合格帧回传的图有 ~90KB，积几帧就堵住了。

**② 连续发送，不要等上一帧的结果。** 服务端对流式帧是 **latest-wins**：新帧会覆盖还没
来得及处理的旧帧（不排队、不堆积），所以多发是安全的。

千万**别做成"一帧一答"**（收到结果才发下一帧）—— 那样发送速率会被锁死成 `1 / RTT`，
和服务端多快完全无关。实测（往返 376ms）：

| 策略 | 目标 fps | 实际发出 |
|---|---|---|
| 一帧一答 | 3 | 2.16 |
| 一帧一答 | 5 | **2.19**（提高目标完全无效，上限就是 1/0.376） |
| 连续发送 | 3 | 2.98 |
| 连续发送 | 5 | 4.91 |

**背压要看 `bufferedAmount`，不要看有没有收到结果。** 它是"还没发出去的字节数"，
反映**上行是否堵住**，与 RTT 无关 —— 这才是真正需要限制的量。

阈值按"能容忍多少排队延迟"算：积压 B 字节、上行 U 字节/秒，新帧就要多等 `B/U` 秒。
手机上行常只有 ~300KB/s，流式帧 7KB，取 24KB（约 3 帧 ≈ 70ms）合适。
**别按 100KB 的老帧尺寸去算** —— 那样 3 帧就是 1 秒排队，比一帧一答还慢。

```js
const MAX_BUFFERED = 24 * 1024;

async function tick(video){
  if (!ws || ws.readyState !== 1) return;
  if (ws.bufferedAmount > MAX_BUFFERED) return;   // 上行堵了就跳过这帧
  if (capturing) return;                          // 高清帧在途，别抢上行
  const blob = await grabAt(video, STREAM_LONG, STREAM_Q);
  if (blob) ws.send(await blob.arrayBuffer());
}
setInterval(() => tick(videoEl), 1000 / 3);
```

---

## 3. 客户端 → 服务端（文本 JSON）

| 消息 | 作用 |
|---|---|
| `{"type":"config","mode":"card"}` | 切模式：`card` 证件 / `face` 人脸。切换会自动重置状态并解锁 |
| **`{"type":"capture"}`** | **声明紧跟其后的那个二进制帧是高清抓拍帧** |
| `{"type":"reset"}` | 解开闭锁、清空防抖状态，重新开始一次抓拍 |
| `{"type":"ping"}` | 心跳，回 `{"type":"pong"}` |
| `{"type":"record","n":50}` | 排查用：让服务端把接下来 n 帧原图+判定存到 `dumps/` |

> 默认模式是 `card`，连上后不发 `config` 也能直接推帧。

**`capture` 的用法** —— WebSocket 保证消息顺序，所以「先发标记、紧跟发帧」是可靠的：

```js
ws.send(JSON.stringify({type:'capture'}));
ws.send(await blob.arrayBuffer());          // 这一帧按高清抓拍处理
```

高清帧**不参与 latest-wins 丢弃**（服务端单独存放、优先处理）—— 它只传一次，
被丢掉整个流程就卡住了。

---

## 4. 服务端 → 客户端（全部是文本 JSON）

### 4.1 连接建立

```json
{ "type":"hello", "mode":"card", "imgsz":960,
  "card_providers":["CUDAExecutionProvider","CPUExecutionProvider"],
  "face_backend":"yunet",
  "active":3, "capacity":20 }
```

`card_providers` 里有 `CUDAExecutionProvider` 才说明在用 GPU。

### 4.2 其它控制回执

```json
{"type":"config_ok","mode":"face"}
{"type":"reset_ok","ignored":3}                  // ignored = 闭锁期间忽略掉的帧数
{"type":"record_started","n":50,"dir":"/workspace/cardface/dumps/20260801_..."}
{"type":"pong"}
{"type":"error","mode":"card","seq":12,"msg":"JPEG 解码失败"}
```

### 4.3 每帧结果 `type:"result"`

**公共字段**

| 字段 | 说明 |
|---|---|
| `mode` | `card` / `face` |
| `capture` | **`true` 表示这是高清抓拍帧的回执**；流式帧没有这个字段 |
| `ok` | 本帧是否合格 |
| `reason` | 机器可读的判定码，见第 5 节 |
| `msg` | 已经写好的中文提示，可直接显示给用户 |
| `conf` | 检测置信度 |
| `seq` | 帧序号（服务端自增） |
| `ms` | 服务端处理耗时（毫秒），不含网络 |
| `dropped` | 因 latest-wins 被丢弃的流式帧数 |
| `frame_size` | `[w,h]` **本帧的坐标基准，画框必须用它** |
| `bytes_in` | 收到的字节数 |
| `final` | `true` = 已出结果、服务端已闭锁。**收到就必须停止发送** |
| `image` | `data:image/jpeg;base64,...` 裁好的图。**只有高清帧回执里才有** |
| `image_size` | `[w,h]` 上面那张图的尺寸 |

> **流式帧即使 `ok:true` 也不带 `image`。** 它的 `ok` 只表示"这一帧姿态合格，
> 可以计入连续帧计数"。图只在高清帧回执里出现。

**证件模式额外字段**

| 字段 | 说明 |
|---|---|
| `quad` | 四个角点 `[[x,y]×4]`，顺序固定 **左上→右上→右下→左下**。流式帧里这是**时域平滑后**的值，画框用它（稳）；高清帧回执里就是该帧的原始角点 |
| `raw_quad` | 该帧未平滑的原始角点。**判定用的是这一组**，不是 `quad` |
| `rotate_deg` | 画面内旋转角，正值=顺时针。合格要求 \|角度\| ≤ 8° |
| `skew` | 透视倾斜度（对边长度差/均值），合格要求 ≤ 0.14 |
| `aspect` | 矫正后宽高比（身份证约 1.585） |
| `area_ratio` | 证件面积 / 画面面积 |
| `jitter` | 本帧原始角点相对平滑值的偏移(px)，越大越抖 |
| `held` | `true` = 本帧漏检，沿用了上一帧的框 |
| `votes` | 形如 `"3/4"`，投票窗口状态 |

合格时 `image` 是**透视矫正后的证件**，默认 908×572（856×540 加 3% 边）。

**人脸模式额外字段**

| 字段 | 说明 |
|---|---|
| `box` | `[x, y, w, h]` 人脸框 |
| `landmarks` | 5 个关键点 `[[x,y]×5]`，顺序：右眼、左眼、鼻尖、右嘴角、左嘴角（YuNet 后端才有） |
| `roll_deg` | 双眼连线倾角（歪头），合格要求 \|角度\| ≤ 20° |
| `yaw_ratio` | 侧脸程度，合格要求 ≤ 0.30 |
| `sharp` | 清晰度（拉普拉斯方差），合格要求 ≥ 20 |
| `area_ratio` | 人脸面积 / 画面面积，合格要求 0.02 ~ 0.85 |
| `count` | 检出的人脸总数（服务端只取最大的那张） |

合格时 `image` 是**向外扩 35% 的人脸裁图**。

---

## 5. 判定码与提示

`msg` 字段已经是可直接展示的中文，`reason` 供你做自定义 UI / 埋点。

### 证件

| reason | msg |
|---|---|
| `ok` | 证件已对准 |
| `no_card` | 未检测到证件，请将证件放入框内 |
| `tilted` | 证件歪了，请对准证件 |
| `upside_down` | 证件方向不对，请正向放置 |
| `skewed` | 请正对镜头拍摄，不要斜着拍 |
| `not_rect` | 请把证件放平，四角要清晰 |
| `too_small` | 请靠近一点，让证件充满取景框 |
| `too_large` | 请稍微拉远一点 |
| `out_of_frame` | 证件超出画面，请对准证件 |
| `bad_aspect` | 证件识别异常，请重新对准 |
| `unstable` | 保持不动…（姿态已合格，正在等防抖确认） |

### 人脸

| reason | msg |
|---|---|
| `ok` | 检测到人脸 |
| `no_face` | 未检测到人脸，请把脸放入框内 |
| `too_small` | 请靠近一点 |
| `too_large` | 请稍微拉远一点 |
| `out_of_frame` | 人脸超出画面，请居中 |
| `rolled` | 请把头摆正 |
| `yawed` | 请正对镜头，不要侧脸 |
| `blurry` | 画面不清晰，请拿稳手机等对焦完成 |

> `unstable` **只出现在证件模式**（服务端投票窗口还没够票），人脸模式不会出现。
> 它不是错误，是"这一帧合格了但还要再确认几帧"，UI 上建议按"即将成功"处理，
> 别显示成红色报错。

---

## 6. ⚠ 画框的坐标系（最容易踩的坑）

`quad` / `box` / `landmarks` 的坐标属于**你上传那张图的像素空间**，也就是 `frame_size`
返回的尺寸（例如 960×720），**不是** `video.videoWidth`（例如 1280×960）。

用错会导致框整体缩小并偏向左上角 —— 识别其实是对的，但看起来完全对不上位置。
（实测偏差 127px / 951px 宽的取景区。）

预览用 `object-fit: cover` 时的正确映射：

```js
// viewW / viewH = 取景容器的 CSS 像素尺寸
function mapper(m, viewW, viewH){
  const [fw, fh] = m.frame_size;                  // 关键：用 frame_size，不是 videoWidth
  const s  = Math.max(viewW / fw, viewH / fh);    // cover：等比放大到铺满
  const ox = (viewW - fw * s) / 2;                // 居中裁切的偏移
  const oy = (viewH - fh * s) / 2;
  return (x, y) => [x * s + ox, y * s + oy];
}

// 用法
const P = mapper(m, viewW, viewH);
ctx.beginPath();
m.quad.forEach(([x, y], i) => {
  const [px, py] = P(x, y);
  i ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
});
ctx.closePath();
ctx.stroke();
```

如果预览用的是 `object-fit: contain`，把 `Math.max` 换成 `Math.min` 即可。

---

## 7. 一次性抓拍的生命周期

连续合格帧的阈值按模式区分：**证件 3 帧、人脸 1 帧**。
证件的四角回归对手抖敏感，多确认几帧能避免抓到正在移动的一帧；人脸的判定项
（大小 / 角度 / 清晰度）本身就已经把不合格的挡掉了，再等 3 帧只是拖慢体验。

```
连接 ─▶ hello
  │
  ├─ 小图7KB ─▶ result(ok=false, reason=tilted)     按 msg 提示用户调整，连续计数清零
  ├─ 小图7KB ─▶ result(ok=true)                     连续 1/3   （注意：没有 image）
  ├─ 小图7KB ─▶ result(ok=true)                     连续 2/3
  ├─ 小图7KB ─▶ result(ok=true)                     连续 3/3 → 达标！
  │
  ├─ {"type":"capture"} + 高清帧100KB
  │      │
  │      ├─▶ result(capture=true, ok=true, final=true, image)   ✓ 拿到裁图，已闭锁
  │      └─▶ result(capture=true, ok=false, reason=tilted)      ✗ 未闭锁，计数清零继续推流
  │
  │   闭锁后：服务端对任何帧【不解码、不处理、不回消息】
  │           前端必须自己停止发送，否则纯烧流量
  │
  └─ {"type":"reset"} ─▶ reset_ok ──▶ 可以重新抓一次
```

```js
const CAPTURE_AFTER = { card: 3, face: 1 };
let okStreak = 0, capturing = false;

ws.onmessage = e => {
  const m = JSON.parse(e.data);
  if (m.type !== 'result') return;         // hello / config_ok / reset_ok / pong / error 等

  if (m.quad) drawQuad(m.quad, m.frame_size);   // 必须用 m.frame_size 换算，见第 6 节
  if (m.box)  drawBox(m.box, m.frame_size);

  if (m.capture){
    // 高清帧回执
    capturing = false;
    if (m.ok && m.image){
      saveResult(m.image);                 // data URL，可直接塞 <img src> 或转 Blob 上传
    } else {
      okStreak = 0;                        // 没通过：服务端未闭锁，继续推流重试
      showTip('高清帧未通过（' + m.msg + '），继续对准…');
    }
  } else {
    // 流式帧：数连续合格帧
    showTip(m.msg);
    okStreak = m.ok ? okStreak + 1 : 0;
    if (okStreak >= CAPTURE_AFTER[mode] && !capturing) sendCapture();
  }

  if (m.final) stopSending();              // ★ 必须停，服务端已经不回了
};

async function sendCapture(){
  if (capturing || !ws || ws.readyState !== 1) return;
  capturing = true;                        // 期间暂停推流，别抢上行
  const blob = await grabCapture(videoEl);
  if (!blob){ capturing = false; return; }
  ws.send(JSON.stringify({type:'capture'}));
  ws.send(await blob.arrayBuffer());
}

function again(){
  okStreak = 0; capturing = false;
  ws.send(JSON.stringify({type:'reset'}));
  startSending();
}
```

**摄像头预览不用关**，只停发送即可，这样用户点"重新识别"能立刻继续。

### 高清帧为什么可能不通过

流式帧判定的是**小图**、高清帧判定的是**另一张刚拍的图**，两者不是同一帧 ——
用户手抖一下、或者刚好在移动，高清帧就可能变成 `tilted` / `out_of_frame`。
这是**正常且必要的**：宁可让用户多对一次，也不要回传一张歪的或缺角的证件图。
前端只要清零计数继续推流即可，服务端在这种情况下**不会闭锁**。

---

## 8. 手机端的两个硬性前提

1. **必须 HTTPS。** `getUserMedia` 只在安全上下文可用，手机访问 `http://192.168.x.x`
   会被浏览器直接禁掉摄像头（WS 本身在 http 下是通的，但拿不到画面）。
   RunPod 的 `https://<POD_ID>-8000.proxy.runpod.net` 是受信任证书，可以直接用。
2. **后置摄像头建议让用户可选。** 手机后置常有主摄/广角/长焦多颗，
   `facingMode:'environment'` 不保证挑到最清晰的那颗。用
   `navigator.mediaDevices.enumerateDevices()` 列出来给用户选（注意：**拿到权限后**
   `label` 才有内容）。

推荐约束：

```js
{ video: { facingMode:{ideal:'environment'},   // 人脸模式用 'user'
           width:{ideal:1280}, height:{ideal:960} }, audio:false }
```

---

## 9. 服务端可调参数（环境变量）

不用改代码，起服务时带上即可。

| 变量 | 默认 | 说明 |
|---|---|---|
| `MAX_USERS` | 20 | 并发上限，超了新连接收 4429 |
| `MAX_ROTATE` | 8 | 证件允许的旋转角（度） |
| `MAX_SKEW` | 0.14 | 证件允许的透视倾斜 |
| `MIN_AREA` / `MAX_AREA` | 0.12 / 0.92 | 证件占画面比例区间 |
| `FACE_MIN_SHARP` | 20 | 人脸清晰度下限，**设 0 可关掉这道闸门** |
| `JPEG_QUALITY` | 88 | 回传图的 JPEG 质量 |
| `VOTE_WIN` / `VOTE_NEED` | 4 / 3 | 投票窗口，调小可缩短等待 |
| `SMOOTH_EMA` | 0.45 | 角点平滑系数，越小越平滑 |

> 注意：投票窗口按**帧数**算而非时间。3 fps 下 4 帧窗口约 1.3 秒，
> 觉得等太久就把 `VOTE_WIN=3 VOTE_NEED=2`。

---

## 10. 完整最小示例

```js
// 生产环境请用配置注入，别写死 —— 重建 Pod 会换 ID
const WS_URL = 'wss://tr05uksdg3cpmu-8000.proxy.runpod.net/ws';

const FPS = 3;
const STREAM_LONG = 416, STREAM_Q = 0.60;       // 流式帧 约 7KB
const CAP_LONG    = 960, CAP_Q    = 0.82;       // 高清帧 约 70~110KB
const CAP_MAX_KB  = 110;
const MAX_BUFFERED = 24 * 1024;                 // 背压：只看未发出字节数
const CAPTURE_AFTER = { card: 3, face: 1 };

let ws, timer, mode = 'card';
let okStreak = 0, capturing = false;
const cap = document.createElement('canvas'), cctx = cap.getContext('2d');

/* ---------- 采集 ---------- */
function grabAt(videoEl, longSide, q){
  const vw = videoEl.videoWidth, vh = videoEl.videoHeight;
  if (!vw) return null;
  const s = longSide / Math.max(vw, vh);
  cap.width  = Math.round(vw * s);
  cap.height = Math.round(vh * s);
  cctx.drawImage(videoEl, 0, 0, cap.width, cap.height);
  return new Promise(r => cap.toBlob(r, 'image/jpeg', q));
}

// 高清帧：只降质量不降分辨率，直到进入体积上限
async function grabCapture(videoEl){
  let blob = null;
  for (let q = CAP_Q; q >= 0.5; q -= 0.12){
    blob = await grabAt(videoEl, CAP_LONG, q);
    if (!blob || blob.size / 1024 <= CAP_MAX_KB) break;
  }
  return blob;
}

/* ---------- 连接 ---------- */
function connect(videoEl, m = 'card'){
  mode = m;
  ws = new WebSocket(WS_URL);
  ws.binaryType = 'arraybuffer';

  ws.onopen = () => {
    ws.send(JSON.stringify({type:'config', mode}));
    setInterval(() => ws.readyState === 1 && ws.send(JSON.stringify({type:'ping'})), 20000);
    startSending(videoEl);
  };
  ws.onclose = e => {
    stopSending();
    if (e.code === 4429) onError('服务并发已满，请稍后重试');
  };
  ws.onmessage = e => {
    const msg = JSON.parse(e.data);
    if (msg.type !== 'result') return;      // hello / config_ok / reset_ok / pong / error

    // 画框：坐标属于上传帧空间，必须用 msg.frame_size 换算（见第 6 节）
    if (msg.quad) onQuad(msg.quad, msg.frame_size, msg.ok);
    if (msg.box)  onBox(msg.box,  msg.frame_size, msg.ok);

    if (msg.capture){
      capturing = false;
      if (msg.ok && msg.image){
        onCaptured(msg.image, msg.mode);     // ★ 最终结果图
      } else {
        okStreak = 0;                        // 高清帧没过，服务端未闭锁，继续对准
        onTip('高清帧未通过（' + msg.msg + '），继续对准…', false, msg.reason);
      }
    } else {
      onTip(msg.msg, msg.ok, msg.reason);
      okStreak = msg.ok ? okStreak + 1 : 0;  // 流式帧即使 ok 也没有 image
      if (okStreak >= (CAPTURE_AFTER[mode] || 1) && !capturing) sendCapture(videoEl);
    }

    if (msg.final) stopSending();            // ★ 已闭锁，必须停
  };
}

/* ---------- 推流 ---------- */
function startSending(videoEl){
  clearInterval(timer);
  okStreak = 0; capturing = false;
  timer = setInterval(async () => {
    if (!ws || ws.readyState !== 1) return;
    if (ws.bufferedAmount > MAX_BUFFERED) return;   // 上行堵了就跳过
    if (capturing) return;                          // 高清帧在途，别抢上行
    const blob = await grabAt(videoEl, STREAM_LONG, STREAM_Q);
    if (blob) ws.send(await blob.arrayBuffer());
  }, 1000 / FPS);
}

function stopSending(){ clearInterval(timer); timer = null; }

async function sendCapture(videoEl){
  if (capturing || !ws || ws.readyState !== 1) return;
  capturing = true;
  const blob = await grabCapture(videoEl);
  if (!blob){ capturing = false; return; }
  ws.send(JSON.stringify({type:'capture'}));        // 声明下一帧是高清帧
  ws.send(await blob.arrayBuffer());
}

/* ---------- 重新识别 / 切模式 ---------- */
function again(videoEl){
  okStreak = 0; capturing = false;
  ws.send(JSON.stringify({type:'reset'}));
  startSending(videoEl);
}
function switchMode(videoEl, m){
  mode = m; okStreak = 0; capturing = false;
  ws.send(JSON.stringify({type:'config', mode: m}));   // 切模式会自动解锁
  startSending(videoEl);
}
```

需要你自己实现的四个回调：`onTip(msg, ok, reason)`、`onQuad(quad, frameSize, ok)`、
`onBox(box, frameSize, ok)`、`onCaptured(dataUrl, mode)`、`onError(msg)`。

---

## 11. 排查连接问题

| 现象 | 原因 | 处理 |
|---|---|---|
| HTTP **502** / RunPod "Waiting for service to respond" | 服务没起，或还在冷启动（约 100s） | SSH 进 Pod 起服务，等 100 秒 |
| HTTP **404**（响应头只有 Cloudflare、`Content-Length: 0`） | 端口没暴露 | Edit Pod → Expose HTTP Ports 加 `8000` |
| 关闭码 **4429** | 并发已满（默认 20） | 稍后重试，别重连风暴 |
| 手机打不开摄像头 | 页面不是 HTTPS | 必须用 `https://` 地址 |
| 框位置对不上 | 用了 `videoWidth` 换算 | 见第 6 节，必须用 `frame_size` |
| 连接莫名断开 | 只发不读，发送缓冲堵死 | 必须持续读 `onmessage` |
| 实发 fps 上不去、卡在 2~3 | 做成了「一帧一答」，被 RTT 锁死 | 见第 2 节规则 ②，改连续发送 |
| 首次成功慢（1 秒以上） | 同上，或 fps 设得太低 | 连续发送 + 提到 5fps |

先用 curl 确认服务本身是否正常，排除前端因素：

```bash
curl https://tr05uksdg3cpmu-8000.proxy.runpod.net/health
```

正常应返回，且 `card_providers` 里要有 `CUDAExecutionProvider`（否则是在用 CPU，慢 3 倍）：

```json
{"ok":true,"card_providers":["CUDAExecutionProvider","CPUExecutionProvider"],
 "card_imgsz":960,"face_backend":"yunet","gpu":true,"active":0,"capacity":20}
```

在 Pod 里起服务（`nohup` 保证 SSH 断开不会带走进程）：

```bash
cd /workspace/cardface && nohup python server/app.py --host 0.0.0.0 --port 8000 > /workspace/app.log 2>&1 &
```

---

## 12. 联调建议

* `web/index.html` 是**可运行的参考实现**，含实时状态表（帧率/丢帧/服务端耗时/往返延迟/
  判定/抖动/清晰度），联调时可以对照它看服务端到底返回了什么。
* 判定不对时点页面上的「录 50 帧供排查」，服务端会把原始帧+判定存到 `dumps/`，
  再跑 `python tools/analyze_dump.py` 能离线复现（含检出率、逐帧跳动、清晰度相关性）。
* 压测：`python tools/load_test.py ws://127.0.0.1:8000/ws --users 8,16,32`
* 单机实测容量：24 并发持续推流丢帧 0.3%；32 并发 5%；建议按 20~24 规划。

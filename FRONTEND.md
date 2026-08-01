# 前端对接文档

证件 / 人脸识别 WebSocket 服务。前端按固定帧率推 JPEG，服务端逐帧回判定结果；
**只有合格的那一帧才会带回裁好的图**，之后服务端闭锁不再处理（一次性抓拍）。

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

* **二进制** JPEG，直接 `ws.send(arrayBuffer)`，不要 base64（base64 会多 33% 流量）
* **长边 960 px** —— 和模型输入一致，再大只是浪费带宽
* **约 100 KB/帧** —— 建议按实际大小反馈调节 JPEG quality
* **3 fps**（可到 5 fps；GPU 上单帧只要 20~30ms，瓶颈在网络往返而不是算力）

```js
const LONG_SIDE = 960, TARGET_KB = 100;
let quality = 0.7;
const cap = document.createElement('canvas'), cctx = cap.getContext('2d');

async function grabFrame(video){
  const vw = video.videoWidth, vh = video.videoHeight;
  if (!vw) return null;
  const s = LONG_SIDE / Math.max(vw, vh);
  cap.width = Math.round(vw * s);
  cap.height = Math.round(vh * s);
  cctx.drawImage(video, 0, 0, cap.width, cap.height);
  const blob = await new Promise(r => cap.toBlob(r, 'image/jpeg', quality));
  const kb = blob.size / 1024;                       // 反馈调节，稳定在 100KB 附近
  if (kb > TARGET_KB * 1.15)      quality = Math.max(0.35, quality - 0.05);
  else if (kb < TARGET_KB * 0.75) quality = Math.min(0.92, quality + 0.04);
  return blob.arrayBuffer();
}
```

### 两条必须遵守的规则

**① 必须持续读取回传消息。** 只发不读会把服务端的发送缓冲堵死，连接会被断开
（实测：8 条只发不读的连接，本该 8 秒的任务卡了 37.9 秒并报 `ConnectionClosedError`）。
合格帧回传的图有 ~90KB，积几帧就堵住了。

**② 连续发送，不要等上一帧的结果。** 服务端是 **latest-wins**：新帧会覆盖还没来得及
处理的旧帧（不排队、不堆积），所以多发是安全的。

千万**别做成"一帧一答"**（收到结果才发下一帧）—— 那样发送速率会被锁死成 `1 / RTT`，
和服务端多快完全无关。实测数据（往返 376ms）：

| 策略 | 目标 fps | 实际发出 | 首次拿到结果图 |
|---|---|---|---|
| 一帧一答 | 3 | 2.16 | 911ms |
| 一帧一答 | 5 | **2.19**（提高目标无效） | 897ms |
| 连续发送 | 3 | 2.98 | 783ms |
| 连续发送 | 5 | 4.91 | 656ms |
| 连续发送 | 8 | 7.84 | 559ms |

服务端单帧只要 32ms，瓶颈全在往返。改成连续发送后首次成功快了约 **38%**。

**背压要看 `bufferedAmount`，不要看有没有收到结果。** `bufferedAmount` 是"还没发出去的
字节数"，反映的是**上行是否堵住**，与 RTT 无关 —— 这才是真正需要限制的东西。

```js
const TARGET_KB = 100;
const MAX_BUFFERED = 3 * TARGET_KB * 1024;   // 约 3 帧的量

async function tick(video){
  if (ws.readyState !== 1) return;
  // 上行堵了就跳过这一帧（服务端反正只处理最新的），不必等结果
  if (ws.bufferedAmount > MAX_BUFFERED) return;
  const buf = await grabFrame(video);
  if (buf) ws.send(buf);
}
setInterval(() => tick(videoEl), 1000 / 3);
```

> **fps 怎么选。** 发得越快首次成功越早，但被丢弃的帧是白花的上行流量。
> GPU 上服务端 32ms 就能处理一帧（约 30fps 的处理能力），所以 **5fps 基本不会被丢**，
> 是速度和流量的较好平衡点（约 500KB/s）。3fps 更省流量（300KB/s）。
> 上面表格里"回传 fps"低于"实发 fps"是因为测试服务跑在 CPU 上（50~60ms/帧）。

---

## 3. 客户端 → 服务端（文本 JSON）

| 消息 | 作用 |
|---|---|
| `{"type":"config","mode":"card"}` | 切模式：`card` 证件 / `face` 人脸。切换会自动重置状态并解锁 |
| `{"type":"reset"}` | 解开闭锁、清空防抖状态，重新开始一次抓拍 |
| `{"type":"ping"}` | 心跳，回 `{"type":"pong"}` |
| `{"type":"record","n":50}` | 排查用：让服务端把接下来 n 帧原图+判定存到 `dumps/` |

> 默认模式是 `card`，连上后不发 `config` 也能直接推帧。

---

## 4. 服务端 → 客户端（全部是文本 JSON）

### 4.1 连接建立

```json
{ "type":"hello", "mode":"card", "imgsz":960,
  "card_providers":["CUDAExecutionProvider","CPUExecutionProvider"],
  "face_backend":"yunet",
  "card_stable_frames":2, "face_stable_frames":2,
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
| `ok` | 是否合格。**只有 `true` 时才有 `image`** |
| `reason` | 机器可读的判定码，见第 5 节 |
| `msg` | 已经写好的中文提示，可直接显示给用户 |
| `conf` | 检测置信度 |
| `seq` | 帧序号（服务端自增） |
| `ms` | 服务端处理耗时（毫秒），不含网络 |
| `dropped` | 因背压被丢弃的帧数 |
| `frame_size` | `[w,h]` **本帧的坐标基准，画框必须用它** |
| `bytes_in` | 收到的字节数 |
| `stable` | 形如 `"1/2"`，防抖进度（未达标时才有） |
| `final` | `true` 表示已抓到结果、服务端已闭锁 |
| `image` | `data:image/jpeg;base64,...` 裁好的图，仅 `ok=true` 时有 |
| `image_size` | `[w,h]` 上面那张图的尺寸 |

**证件模式额外字段**

| 字段 | 说明 |
|---|---|
| `quad` | 四个角点 `[[x,y]×4]`，顺序固定 **左上→右上→右下→左下**。已做时域平滑，用它画框 |
| `raw_quad` | 未平滑的原始角点，调试用 |
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
| `unstable` | 保持不动… |

> `unstable` 不是错误，是"这一帧合格了但还要再确认几帧"。UI 上建议按"即将成功"处理，
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

```
连接 ─▶ hello
  │
  ├─ 推帧 ─▶ result(ok=false, reason=tilted)      按 msg 提示用户调整
  ├─ 推帧 ─▶ result(ok=false, reason=unstable)    快成了
  ├─ 推帧 ─▶ result(ok=true, final=true, image)   ✓ 拿到图
  │
  │   ◀── 此时服务端已闭锁：后续帧【不解码、不处理、不回任何消息】
  │        前端必须自己停止发送，否则就是白烧流量
  │
  └─ {"type":"reset"} ─▶ reset_ok ──▶ 可以重新抓一次
```

```js
ws.onmessage = e => {
  const m = JSON.parse(e.data);
  if (m.type !== 'result') return;          // hello / config_ok / pong / error 等

  showTip(m.msg);                           // 直接用服务端的中文提示
  if (m.quad) drawQuad(m);                  // 注意用 m.frame_size 换算

  if (m.ok && m.image){
    saveResult(m.image);                     // data URL，可直接塞 <img src> 或转 Blob 上传
  }
  if (m.final){
    stopSending();                           // ★ 必须停，服务端已经不回了
  }
};

function again(){
  ws.send(JSON.stringify({type:'reset'}));
  startSending();
}
```

**摄像头预览不用关**，只停发送即可，这样用户点"重新识别"能立刻继续。

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
| `STABLE_FRAMES` | 2 | 连续几帧合格才回传，防抖 |
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

const FPS = 3, LONG_SIDE = 960, TARGET_KB = 100;
const MAX_BUFFERED = 3 * TARGET_KB * 1024;      // 背压：只看未发出字节数
let ws, timer, quality = 0.7;
const cap = document.createElement('canvas'), cctx = cap.getContext('2d');

async function connect(videoEl, mode = 'card'){
  ws = new WebSocket(WS_URL);
  ws.binaryType = 'arraybuffer';

  ws.onopen = () => {
    ws.send(JSON.stringify({type:'config', mode}));
    setInterval(() => ws.readyState === 1 && ws.send(JSON.stringify({type:'ping'})), 20000);
    startSending(videoEl);
  };
  ws.onclose = e => {
    stopSending();
    if (e.code === 4429) alert('服务并发已满，请稍后重试');
  };
  ws.onmessage = e => {
    const m = JSON.parse(e.data);
    if (m.type !== 'result') return;

    onTip(m.msg, m.ok, m.reason);
    if (m.mode === 'card' && m.quad) onQuad(m.quad, m.frame_size, m.ok);
    if (m.mode === 'face' && m.box)  onBox(m.box, m.frame_size, m.ok);

    if (m.ok && m.image) onCaptured(m.image, m.mode);
    if (m.final) stopSending();          // ★ 服务端已闭锁
  };
}

function startSending(videoEl){
  clearInterval(timer);
  timer = setInterval(async () => {
    if (!ws || ws.readyState !== 1) return;
    // 连续发送：不等结果，只在上行堵住时跳过（服务端 latest-wins）
    if (ws.bufferedAmount > MAX_BUFFERED) return;

    const vw = videoEl.videoWidth, vh = videoEl.videoHeight;
    if (!vw) return;
    const s = LONG_SIDE / Math.max(vw, vh);
    cap.width = Math.round(vw * s); cap.height = Math.round(vh * s);
    cctx.drawImage(videoEl, 0, 0, cap.width, cap.height);

    const blob = await new Promise(r => cap.toBlob(r, 'image/jpeg', quality));
    const kb = blob.size / 1024;
    if (kb > TARGET_KB * 1.15)      quality = Math.max(0.35, quality - 0.05);
    else if (kb < TARGET_KB * 0.75) quality = Math.min(0.92, quality + 0.04);

    ws.send(await blob.arrayBuffer());
  }, 1000 / FPS);
}

function stopSending(){ clearInterval(timer); timer = null; }
function retry(videoEl){ ws.send(JSON.stringify({type:'reset'})); startSending(videoEl); }
```

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

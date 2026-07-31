# cardpose — 证件姿态 / 人脸 WebSocket 识别服务

手机端每秒推 3 帧、每帧约 100KB 的 JPEG 上来；服务端判断：

- **证件模式**：只有"正对镜头、放平、够大、没出框"的证件才算合格。
  合格 → 回传**透视矫正后的证件图**（只比证件本身大一圈，其余截掉）；
  歪的 / 斜的 → 不回图，回传 `请对准证件` 之类的提示。
- **人脸模式**：检测到人脸且够大够正 → 回传人脸裁图；否则回传提示。

```
web/index.html   手机页面：开摄像头、3fps、自适应压到 ~100KB、WS 收发、画角点框
server/app.py    FastAPI + WebSocket，最新帧覆盖旧帧的背压
server/card_detector.py  YOLO11s-pose 解码 + 姿态判定 + 透视矫正
server/face_detector.py  YuNet(优先) / Haar(兜底) 人脸检测
cardpose.onnx    YOLO11s-pose，1 类 card，4 个角点，输入 960x960
models/          YuNet 人脸模型
```

---

## 一、本地跑起来（手机能访问）

```bash
powershell -ExecutionPolicy Bypass -File .\run_local.ps1
```

脚本会自动：找本机内网 IP → 生成带该 IP 的自签证书 → 放通防火墙 → 用 HTTPS 启动。
然后手机连**同一个 WiFi**，打开它打印出来的 `https://<内网IP>:8443/`。

> **为什么必须 HTTPS**：浏览器只在"安全上下文"（HTTPS 或 localhost）下才允许
> `getUserMedia` 开摄像头。手机用 `http://192.168.x.x` 访问一定拿不到摄像头，
> 页面会直接提示。自签证书会报"不安全"，手动继续即可：
> Android Chrome → 高级 → 继续前往；iOS Safari → 显示详细信息 → 访问此网站。

> **防火墙**：加入站规则需要管理员权限。普通权限下脚本会打印
> `添加防火墙规则失败（需要管理员权限）` 然后继续启动——服务本身能跑，
> 但**手机可能连不上**。手机连不上就用管理员 PowerShell 执行一次：
>
> ```powershell
> New-NetFirewallRule -DisplayName "cardpose-8443" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8443 -Profile Private
> ```
>
> 另外确认手机和电脑在同一个 WiFi，且这个网络在 Windows 里是**"专用网络"**
> 而不是"公用网络"（公用网络下入站默认全拦）。

换端口 / 只用 HTTP 调试：

```bash
powershell -ExecutionPolicy Bypass -File .\run_local.ps1 -Port 8443
powershell -ExecutionPolicy Bypass -File .\run_local.ps1 -Http
```

手动启动（不用脚本）：

```bash
python server/app.py --host 0.0.0.0 --port 8443 --certfile certs/cert.pem --keyfile certs/key.pem
```

---

## 二、两个自测脚本

```bash
python server/selftest.py                 # 离线：合成各种姿态的证件，验证判定逻辑
python server/ws_client_test.py           # 端到端：模拟手机 3fps 推流，检查回传
python server/ws_client_test.py ws://localhost:8000/ws    # 测远端/HTTP 部署
```

`selftest.py` 会在 `selftest_out/` 写出可视化结果（绿框=合格，橙框=不合格）
和矫正裁剪图。本机 CPU 实测：**单帧 65ms，约 15fps**，够 3fps 用。

---

## 三、WebSocket 协议

**客户端 → 服务端**

| 内容 | 说明 |
|---|---|
| 文本 `{"type":"config","mode":"card"}` | 切模式，`card` 或 `face` |
| 文本 `{"type":"reset"}` | 清掉防抖计数 |
| 文本 `{"type":"ping"}` | 保活，服务端回 `pong` |
| **二进制** | 一帧 JPEG，按当前 mode 处理 |

用二进制发原始 JPEG，不要 base64——省 33% 流量。

**服务端 → 客户端**（都是 JSON 文本）

连上先发一条：

```json
{"type":"hello","mode":"card","card_providers":["CUDAExecutionProvider"],
 "imgsz":960,"face_backend":"yunet","card_stable_frames":2,"face_stable_frames":2}
```

每帧一条结果。不合格时**没有** `image` 字段：

```json
{"type":"result","mode":"card","ok":false,"reason":"tilted","msg":"证件歪了，请对准证件",
 "conf":0.95,"rotate_deg":-29.15,"skew":0.036,"corner_err":3.1,"area_ratio":0.295,
 "aspect":1.66,"quad":[[x,y],[x,y],[x,y],[x,y]],
 "seq":4,"dropped":0,"ms":69.0,"frame_size":[960,720],"bytes_in":106496}
```

合格时带矫正好的证件图：

```json
{"type":"result","mode":"card","ok":true,"reason":"ok","msg":"证件已对准",
 "image":"data:image/jpeg;base64,...","image_size":[908,572], "...":"..."}
```

人脸模式的字段是 `box`(x,y,w,h)、`landmarks`(5点)、`roll_deg`、`yaw_ratio`、`count`。

`quad` 顺序固定是 **[左上, 右上, 右下, 左下]**，原图像素坐标，可直接画框。

### 不合格原因一览

| reason | 提示语 | 触发条件 |
|---|---|---|
| `no_card` | 未检测到证件，请将证件放入框内 | 置信度 < 0.45 |
| `tilted` | 证件歪了，请对准证件 | 画面内旋转 > 8° |
| `upside_down` | 证件方向不对，请正向放置 | 旋转 > 135° |
| `skewed` | 请正对镜头拍摄，不要斜着拍 | 对边长度差 / 均值 > 0.14 |
| `not_rect` | 请把证件放平，四角要清晰 | 四角偏离 90° > 12° |
| `too_small` / `too_large` | 请靠近 / 拉远一点 | 面积占比 < 0.12 或 > 0.92 |
| `out_of_frame` | 证件超出画面，请对准证件 | 角点出框，或宽高比异常且贴边 |
| `unstable` | 保持不动… | 合格但还没连续够 2 帧 |

人脸侧：`no_face` / `too_small` / `too_large` / `out_of_frame` / `rolled`(歪头>20°) / `yawed`(侧脸)。

### 防抖

合格帧要**连续 2 帧**才回传图（`STABLE_FRAMES`），中间那帧 `reason=unstable`。
避免手抖时刚好一帧合格就误抓。

### 页面切后台

浏览器会把后台标签页的 `setInterval` 限流到 **1 秒一次**，摄像头也会停帧。
所以页面切后台时会自动暂停推流，回到前台自动恢复（`visibilitychange`）。
调试时如果发现帧率只有 1fps 左右，先确认页面在前台。

### 背压

服务端每连接只留**最新一帧**，处理不过来就丢掉中间的，返回的 `dropped`
告诉你丢了几帧。前端同样限制"上一帧结果没回来就不发新帧"（`MAX_INFLIGHT=1`）。
实测：一次性猛推 12 帧，服务端只处理 2 帧、丢 10 帧，不会排队堆积到延迟爆炸。

---

## 四、调阈值

全部走环境变量，不用改代码：

```bash
MAX_ROTATE=6       # 旋转容忍度（度），越小越严
MAX_SKEW=0.10      # 透视容忍度
MIN_AREA=0.15      # 证件至少占画面多大
MAX_AREA=0.92
MARGIN=0.01        # 角点允许超出画面的比例
BORDER=0.04        # "贴边"判定比例
CARD_CONF=0.45     # 检测置信度门槛
CARD_PAD=0.03      # 裁剪时四周多留多少（相对证件边长）
STABLE_FRAMES=2    # 连续几帧合格才回传
JPEG_QUALITY=88    # 回传图质量
KPT_PERM=1,2,3,0   # 模型角点 -> [TL,TR,BR,BL] 的置换，换模型才需要改
```

前端在 `web/index.html` 顶部：`FPS`、`TARGET_KB`、`LONG_SIDE`、`MAX_INFLIGHT`。

---

## 五、关于这个模型的两个坑（实测结论）

**1. 角点是"语义固定"的，不能靠几何位置猜。**
实测 0/15/30/60/90/180/-30 度，模型输出的关键点顺序恒定：
`kpt0=证件自身左下, kpt1=左上, kpt2=右上, kpt3=右下`。
所以用固定置换 `[1,2,3,0]` 就能拿到 `[TL,TR,BR,BL]`。

一开始我按"最靠左上的点当左上角"这种几何方法排序，结果证件转过 45° 后
"左边"会被认成"上边"，**30° 和 90° 的歪证件被量成 0° 旋转、误判为合格**——
正好是最不能出错的地方。现在用完整 `atan2`（不折叠到 ±45°）量角度，
±180° 全范围实测最大误差 5°。

**2. 证件被画面边缘裁掉时，模型不会把角点预测到画面外**，而是沿着边缘给出一个
"压扁"的四边形。实测：卡一半出框时 `aspect=1.03`，而正常是 1.51。
所以判"出框"靠的是**宽高比异常 + 角点贴边**，光看角点坐标是抓不到的
（一个正常的、充满取景框的证件，角点比出框的还更靠边）。

另外：合成图容易骗自己。我第一版用粗糙的矩形当假证件，模型在 0° 时置信度只有
0.0002、旋转 20° 才有 0.91，看着像"模型只认歪的卡"；换成带圆角、渐变底纹、
人像、印刷字、光照梯度和噪点的仿真图后，0° 置信度 0.98。**是合成图太假，不是模型的问题。**

---

## 六、部署到 RunPod

见 [RUNPOD.md](RUNPOD.md)。

一句话版本：RunPod 的 HTTP 代理自带合法 HTTPS 证书，
手机直接开 `https://<POD_ID>-8000.proxy.runpod.net/` 就能用摄像头，
**不用再折腾自签证书**。

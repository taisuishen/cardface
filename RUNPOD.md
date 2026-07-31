# 部署到 RunPod Pod

> 用 **Pod**（常驻 GPU 容器）而不是 Serverless：WebSocket 是长连接，
> Serverless 的请求-响应模型不适合。

**最大的好处**：RunPod 的 HTTP 代理给你一个带**合法 HTTPS 证书**的域名
（`https://<POD_ID>-8000.proxy.runpod.net`）。手机浏览器只在 HTTPS 下才让开摄像头，
所以部署到 RunPod 之后就不用再折腾自签证书、也不用点"继续前往不安全网站"了。

容器里跑普通 HTTP 就行，TLS 由代理终结。前端 `index.html` 会根据
`location.protocol` 自动选 `ws://` 还是 `wss://`，不用改代码。

---

## 方式 A：不 build 镜像，直接用模板 Pod（推荐先这样跑通）

### 1. 开 Pod

RunPod 控制台 → **Pods** → **Deploy**：

| 项 | 填什么 |
|---|---|
| GPU | **RTX A4000 / RTX 4000 Ada 就够**。YOLO11s 很小，别买 A100 浪费钱 |
| Template | `RunPod PyTorch 2.x`（自带 CUDA + cuDNN，省事） |
| Container Disk | 20 GB |
| **Expose HTTP Ports** | **加上 `8000`** ← 关键，不加就访问不到 |
| Expose TCP Ports | 不用（TCP 直连没有 TLS，手机开不了摄像头） |

点 Deploy，等状态变 Running。

### 2. 把代码传上去

进 Pod 的 **Web Terminal**（或 Connect → SSH）。三种传法选一个：

**(a) git（最省事，推荐）** — 先把这个目录推到你自己的 git 仓库，然后：

```bash
cd /workspace
git clone <你的仓库地址> cardpose
cd cardpose
```

> `cardpose.onnx` 有 39MB，走 git 记得用 Git LFS，或者模型单独传（见 c）。

**(b) runpodctl（本机 → Pod，不用 git）**

本机（Windows PowerShell，先装 [runpodctl](https://github.com/runpod/runpodctl)）：

```powershell
runpodctl send cardpose.onnx server web models
```

它会打印一个一次性 code，在 Pod 终端里执行：

```bash
cd /workspace && mkdir -p cardpose && cd cardpose
runpodctl receive <上面打印的 code>
```

**(c) 模型太大就用 wget** — 把 `cardpose.onnx` 放到任意可下载的地址（对象存储 / HF），
然后在 Pod 里 `wget -O cardpose.onnx <url>`。

### 3. 装依赖

```bash
cd /workspace/cardpose
pip uninstall -y onnxruntime          # 重要：CPU 版和 GPU 版不能共存
pip install -r server/requirements-gpu.txt
```

### 4. 确认 GPU 真的被用上了

> **别用 `ort.get_available_providers()` 判断。** 它列的是**编译时**支持的 provider，
> 即使运行时 CUDA 动态库加载失败、实际跑在 CPU 上，它照样会打印
> `['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']`。

真正能确认的是**能不能建出 session**：

```bash
python -c "
import onnxruntime as ort
s = ort.InferenceSession('cardpose.onnx', providers=['CUDAExecutionProvider'])
print('实际生效:', s.get_providers())"
```

或者直接看服务启动日志里的这一行 —— 这是最终判据：

```
[init] card providers=['CUDAExecutionProvider', 'CPUExecutionProvider']   # 对
[init] card providers=['CPUExecutionProvider']                            # 没用上 GPU
```

只看到 CPU 的话，按这个顺序查：

- **`libcublasLt.so.13: cannot open shared object file`**（最常见）
  → 装了 CUDA 13 版的 ORT，但镜像是 CUDA 12。ORT **1.27 起** PyPI 的 GPU wheel
  默认按 CUDA 13 构建，最后一个 CUDA 12 版本是 1.26.x：

  ```bash
  pip uninstall -y onnxruntime-gpu && pip install "onnxruntime-gpu<1.27"
  ```

  反之如果 `nvidia-smi` 显示 CUDA 13.x，那就要装 `onnxruntime-gpu>=1.27`。
  用 `nvidia-smi` 右上角的 CUDA Version 对一下再选。
- 忘了 `pip uninstall onnxruntime` → 卸干净重装 `onnxruntime-gpu`
- `nvidia-smi` 看不到卡 → Pod 没分到 GPU，重开
- cuDNN 版本不匹配 → 换 `nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04` 基础镜像（方式 B）

### 5. 起服务

```bash
python server/app.py --host 0.0.0.0 --port 8000
```

启动日志里应该看到：

```
[init] card providers=['CUDAExecutionProvider', 'CPUExecutionProvider'] imgsz=960
[init] face backend=yunet
```

想让它退出终端也不停，用 nohup 或 tmux：

```bash
nohup python server/app.py --host 0.0.0.0 --port 8000 > /workspace/app.log 2>&1 &
tail -f /workspace/app.log
```

### 6. 手机访问

Pod 详情页 → **Connect** → HTTP Services 里那个 8000 的链接，形如：

```
https://<POD_ID>-8000.proxy.runpod.net
```

手机直接打开就能用（合法证书，不会报警）。先验证一下：

```bash
curl https://<POD_ID>-8000.proxy.runpod.net/health
# {"ok":true,"card_providers":["CUDAExecutionProvider",...],"face_backend":"yunet"}
```

本机也可以拿自测脚本打远端：

```bash
python server/ws_client_test.py wss://<POD_ID>-8000.proxy.runpod.net/ws
```

---

## 方式 B：自己 build Docker 镜像（要长期跑 / 要可复现就用这个）

仓库里的 `Dockerfile` 用 `nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04`，
CUDA 12 + cuDNN 9，`onnxruntime-gpu` 能直接用。

```bash
docker build -t <你的dockerhub用户名>/cardpose:latest .
docker push <你的dockerhub用户名>/cardpose:latest
```

RunPod → Deploy → **Custom Image**：

| 项 | 填什么 |
|---|---|
| Container Image | `<你的用户名>/cardpose:latest` |
| Expose HTTP Ports | `8000` |
| Container Start Command | 留空（用 Dockerfile 的 CMD） |

私有镜像要在 Settings → Container Registry Auth 里配凭证。

调阈值不用重新 build，Pod 的 **Environment Variables** 里加就行：

```
MAX_ROTATE=6
MAX_SKEW=0.10
MIN_AREA=0.15
STABLE_FRAMES=2
CARD_PAD=0.03
```

---

## 性能与容量估算

本机 CPU 实测单帧 **65ms**。同样的模型在 GPU 上大约 **10~20ms/帧**
（960x960、batch 1、YOLO11s-pose；具体看卡）。

按每个用户 3fps 算：

```
单用户 GPU 占用 ≈ 3 帧/秒 × 15ms = 45ms/秒 ≈ 4.5%
```

理论上一张卡能扛 20 来个并发用户，但实际要打折——JPEG 解码和前后处理在 CPU 上、
Python GIL、以及 `run_in_executor` 默认线程池的限制。**保守按一张卡 8~12 个并发用户估。**

要压更多的话：

- 前后处理（letterbox / imdecode）也搬到 GPU，或用 `cv2.setNumThreads` 调线程
- 上 TensorRT EP（这个模型是 `dynamic=False`、固定 960x960、batch 1，很适合转 TRT）
- 多进程 + 一个反向代理分流，绕开 GIL

---

## 会踩的坑

**1. `Expose HTTP Ports` 忘了填 8000** → 代理域名 502。这是最常见的。

**2. `onnxruntime` 和 `onnxruntime-gpu` 装了两个** → 只会走 CPU，而且不报错，
只是慢 4 倍。一定用第 4 步那条命令确认 `CUDAExecutionProvider` 在列表里。

**3. 长连接被代理掐掉** → 前端已经每 20 秒发一次 `{"type":"ping"}` 保活了
（`web/index.html` 里的 `pingTimer`）。自己写客户端的话记得也加上。

**4. Pod 停了数据就没了** → `/workspace` 挂的是容器盘，Pod 一 Terminate 就清空。
模型和代码要留着就挂 **Network Volume**，并把 `CARD_MODEL` 指过去：

```bash
CARD_MODEL=/runpod-volume/cardpose.onnx MODEL_DIR=/runpod-volume/models \
  python server/app.py --host 0.0.0.0 --port 8000
```

**5. 冷启动** → 模型加载 + 预热大约几秒（代码里已经跑了一次 dummy 推理预热，
不然第一帧会特别慢）。`/health` 返回 200 之后再让手机连。

**6. 别用 TCP 端口直连** → 直连是 `ws://<ip>:<port>`，没有 TLS，
手机浏览器不给开摄像头，白折腾。老老实实用 HTTP 代理。

**7. 证件图是明文 base64 走 WS 回来的** → 生产环境注意这是敏感数据：
代理这一段是 TLS 加密的，但服务端日志别把 `image` 字段打出来，
也别在 Pod 上把回传的证件图落盘。

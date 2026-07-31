<#
  本地启动（手机可访问）。
  手机浏览器只有在 HTTPS 或 localhost 下才允许开摄像头，
  所以这里自动生成一张带本机 IP 的自签证书，用 HTTPS 起服务。

      powershell -ExecutionPolicy Bypass -File .\run_local.ps1
      powershell -ExecutionPolicy Bypass -File .\run_local.ps1 -Port 8443
#>
param(
  [int]$Port = 8443,
  [string]$BindHost = "0.0.0.0",
  [switch]$Http,            # 只用 HTTP（手机开不了摄像头，仅调试用）
  [switch]$NewCert          # 强制重新生成证书
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# ---- 找本机内网 IP ----
$ip = (Get-NetIPConfiguration |
       Where-Object { $_.IPv4DefaultGateway -ne $null -and $_.NetAdapter.Status -eq "Up" } |
       Select-Object -First 1).IPv4Address.IPAddress
if (-not $ip) {
  $ip = (Get-NetIPAddress -AddressFamily IPv4 |
         Where-Object { $_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "169.254.*" } |
         Select-Object -First 1).IPAddress
}
Write-Host "本机内网 IP: $ip" -ForegroundColor Cyan

# ---- 依赖检查 ----
python -c "import fastapi, uvicorn, cv2, onnxruntime" 2>$null
if ($LASTEXITCODE -ne 0) {
  Write-Host "正在安装依赖…" -ForegroundColor Yellow
  python -m pip install -r server/requirements-cpu.txt
}

if (-not (Test-Path "cardpose.onnx")) { throw "找不到 cardpose.onnx" }

$scheme = "http"
$sslArgs = @()

if (-not $Http) {
  New-Item -ItemType Directory -Force -Path certs | Out-Null
  $needCert = $NewCert -or -not (Test-Path "certs/cert.pem") -or -not (Test-Path "certs/key.pem")

  # openssl 会把进度点写到 stderr。Windows PowerShell 5.1 会把原生命令的 stderr
  # 包成 ErrorRecord，配合 $ErrorActionPreference='Stop' 会直接中断脚本，
  # 所以调 openssl 时临时把 EAP 降成 Continue，只看 $LASTEXITCODE 判成败。
  # 注意：参数不能叫 $Args —— 那是 PowerShell 的自动变量，会导致 splat 失效
  function Invoke-OpenSsl {
    param([string[]]$SslArgs, [switch]$Capture)
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
      if ($Capture) { $out = & openssl @SslArgs 2>&1 | Out-String }
      else          { & openssl @SslArgs 2>&1 | Out-Null; $out = "" }
      return [pscustomobject]@{ Code = $LASTEXITCODE; Out = $out }
    } finally { $ErrorActionPreference = $prev }
  }

  if (-not (Get-Command openssl -ErrorAction SilentlyContinue)) {
    throw "没找到 openssl。装了 Git for Windows 的话它在 C:\Program Files\Git\usr\bin\，把该目录加到 PATH，或改用 -Http 参数。"
  }

  # 证书里没有当前 IP 的话也要重新签，否则手机会报证书不匹配
  if (-not $needCert) {
    $r = Invoke-OpenSsl -Capture -SslArgs @('x509','-in','certs/cert.pem','-noout','-ext','subjectAltName')
    if ($r.Code -ne 0 -or $r.Out -notmatch [regex]::Escape($ip)) {
      Write-Host "证书里没有 $ip，重新生成…" -ForegroundColor Yellow
      $needCert = $true
    }
  }

  if ($needCert) {
    Write-Host "生成自签证书 (SAN: $ip)…" -ForegroundColor Yellow
    $r = Invoke-OpenSsl -SslArgs @(
      'req','-x509','-newkey','rsa:2048','-nodes',
      '-keyout','certs/key.pem','-out','certs/cert.pem','-days','825',
      '-subj','/CN=cardpose-local',
      '-addext',"subjectAltName=IP:$ip,IP:127.0.0.1,DNS:localhost")
    if ($r.Code -ne 0) { throw "证书生成失败 (openssl 退出码 $($r.Code))" }
    if (-not (Test-Path 'certs/cert.pem')) { throw "证书生成失败：没有产出 certs/cert.pem" }
    Write-Host "证书已生成" -ForegroundColor Green
  }

  $scheme = "https"
  $sslArgs = @("--certfile", "certs/cert.pem", "--keyfile", "certs/key.pem")
}

# ---- 放通防火墙（手机要能连进来）----
$ruleName = "cardpose-$Port"
if (-not (Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue)) {
  try {
    New-NetFirewallRule -DisplayName $ruleName -Direction Inbound -Action Allow `
      -Protocol TCP -LocalPort $Port -Profile Private | Out-Null
    Write-Host "已添加防火墙入站规则: $ruleName (专用网络)" -ForegroundColor Green
  } catch {
    Write-Host "添加防火墙规则失败（需要管理员权限）。手机连不上的话请手动放通 TCP $Port。" -ForegroundColor Yellow
  }
}

Write-Host ""
Write-Host "======================================================" -ForegroundColor Green
Write-Host " 手机用同一个 WiFi 打开:  $scheme`://$ip`:$Port/" -ForegroundColor Green
Write-Host " 电脑本机打开:            $scheme`://localhost:$Port/" -ForegroundColor Green
if ($scheme -eq "https") {
  Write-Host ""
  Write-Host ' 自签证书会提示"不安全"，这是正常的：' -ForegroundColor Yellow
  Write-Host '   Android Chrome: 高级 -> 继续前往' -ForegroundColor Yellow
  Write-Host '   iOS Safari:     显示详细信息 -> 访问此网站' -ForegroundColor Yellow
}
Write-Host "======================================================" -ForegroundColor Green
Write-Host ""

python server/app.py --host $BindHost --port $Port @sslArgs

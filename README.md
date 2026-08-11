# GLM-5.2 on RTX PRO 6000 Blackwell — 環境重建與部署

在 GPUhub / AutoDL 的 RTX PRO 6000 Blackwell（sm_120）機器上，從乾淨的 base image
重建可跑 GLM-5.2 NVFP4 的 vLLM 環境，並部署到 8 卡做 benchmark。

所有版本組合與踩雷紀錄都是實機驗證過的，不是照文件抄的。

---

## 這份 repo 有什麼

| 檔案 | 用途 |
|---|---|
| `rebuild-env.sh` | 一鍵重建環境（torch / vLLM / flashinfer / nvcc / 系統層設定），約 20–30 分鐘 |
| `verify-weights.sh` | 權重完整性 + vLLM 環境驗證，**單卡就能跑完**，用來在升配 8 卡前確認一切就緒 |
| `sanitize-for-image.sh` | 把 image 公開分享前，清除所有個人憑證與身分痕跡 |
| `requirements-frozen.txt` | 完整 pip freeze，供精確重現 |

---

## 目標環境

| 項目 | 值 |
|---|---|
| GPU | RTX PRO 6000 Blackwell SE 96GB × 8（**sm_120**） |
| Driver / CUDA | 595.58.03 / CUDA 13.2 |
| OS | Ubuntu 22.04 + 內建 miniconda py3.12 |
| 模型 | [`nvidia/GLM-5.2-NVFP4`](https://huggingface.co/nvidia/GLM-5.2-NVFP4) — 432.9 GB，47 分片，232,385 tensors |

模型架構 `GlmMoeDsaForCausalLM`，hidden 6144 / 78 層，NVFP4 量化 + FP8 KV cache（group_size 16）。

### 為什麼選 NVFP4

實測比對 vLLM 0.26.0 支援的 quant method 對照 HF 上的 GLM-5.2 量化版本：

| Repo | 大小 | quant_method | 結論 |
|---|---|---|---|
| **nvidia/GLM-5.2-NVFP4** | **432.9 GB** | modelopt (NVFP4) | **選這個** — Blackwell 原生 FP4 |
| cyankiwi/GLM-5.2-AWQ-INT4 | 474.3 GB | compressed-tensors | 備案 |
| QuantTrio/GLM-5.2-Int4-Int8Mix | 405.5 GB | compressed-tensors | 備案（最小） |
| PhalaCloud/GLM-5.2-W4AFP8 | 399.8 GB | w4afp8 | **不可用** — vLLM 0.26 無此 method |
| zai-org/GLM-5.2-FP8 | 755.7 GB | fp8 | 塞得下但沒空間給 KV cache |
| zai-org/GLM-5.2 (BF16) | 1506.7 GB | — | 8×96GB 放不下 |

---

## 磁碟規則（最重要的環境常識）

| 路徑 | 角色 | 特性 |
|---|---|---|
| `/` (30GB) | System Disk | **會存進 image**。裝環境用，空間極小 |
| `/root/autodl-tmp` (可擴) | Data Disk | **不會存進 image**，綁定實體 host。放權重、快取 |

要進 image 的（conda env、pip 套件、JIT 產物）放 system disk；
一次性大檔（模型權重、HF cache、pip cache）放 data disk。

**data disk 需求：600GB**（433GB 權重 + 下載暫存 + 結果，並留餘裕換備案模型重試）。

---

## 快速開始

```bash
git clone https://github.com/onlyfly34/test-llm-with-cloud-gpu.git
cd test-llm-with-cloud-gpu

# 1. 重建環境（約 20–30 分鐘，大部分是下載）
bash rebuild-env.sh 2>&1 | tee rebuild.log

# 2. 下載權重（433GB，@14MB/s 約 8 小時）
screen -S dl
export HF_HOME=/root/autodl-tmp/hf-home        # ← 不能省，見下方「.bashrc 陷阱」
hf download nvidia/GLM-5.2-NVFP4 --local-dir /root/autodl-tmp/glm-5.2-nvfp4
# detach: Ctrl-A D

# 3. 驗證（單卡即可，升配 8 卡前務必先跑）
bash verify-weights.sh

# 4. 8 卡起服務
vllm serve /root/autodl-tmp/glm-5.2-nvfp4 --served-model-name glm-5.2 \
  -tp 8 --max-model-len 131072 --gpu-memory-utilization 0.92
```

長時間指令一律放 `screen` 內（session 命名 `dl` / `serve` / `bench`），SSH 斷線才不會中斷。

---

## 最終環境版本

| 元件 | 版本 |
|---|---|
| torch | 2.11.0+cu130 |
| vLLM | 0.26.0 |
| flashinfer-python | 0.6.14 |
| nvcc / cuda-crt / nvvm | 13.0.88（必須與 cuda-toolkit 13.0.x 對齊） |
| NCCL | 2.28.9（pip 版，非系統的 2.28.3） |
| transformers | 5.15.0 |

`rebuild-env.sh` 跑完會自動驗證：`GlmMoeDsaForCausalLM` 已支援、`has_flashinfer=True`、
NCCL 可由名稱載入、device 為 sm_120。

---

## 網路：GPUhub Singapore 的兩個陷阱

### IPv6 完全不通，但 DNS 優先回 AAAA

`curl -6` 一律 `code=000`。不修的話每條連線都要先卡一次才 fallback。

```bash
echo "precedence ::ffff:0:0/96  100" >> /etc/gai.conf
```

容器內沒權限改 `sysctl net.ipv6.conf.*.disable_ipv6`，只能走 gai.conf。
`rebuild-env.sh` 第一步就會做這件事。

### 不要用 uv

uv 自帶 hickory-dns，**不讀 `/etc/gai.conf`**，會一直撞壞掉的 IPv6。
實測 **0.2 MB/s（uv）vs 11 MB/s（pip）**，差約 50 倍。同一時刻 curl 也有 9.8 MB/s，
所以瓶頸確定在 uv 的解析路徑，不是網路。

### 各來源實測速度

| 來源 | Singapore-A | Singapore-B |
|---|---|---|
| `huggingface.co` | 13.4 MB/s | 14.4 MB/s |
| `download.pytorch.org` (R2) | 8.7–11.5 MB/s | 14.4 MB/s |
| `pypi.nvidia.com` | 8.7 MB/s | 15.9 MB/s |
| `files.pythonhosted.org` (Fastly) | **0.22 MB/s** | 7.6 MB/s |
| 中國鏡像（aliyun / tuna / nju） | 0.02–6.4 MB/s | — |

**同區不同機差異極大**，PyPI 在 Singapore-A 慢到不可用。所以 `rebuild-env.sh`
安裝 torch 時**刻意不加** `--extra-index-url pypi.org`：

```bash
pip install --index-url https://download.pytorch.org/whl/cu130 \
    torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0
```

加了的話 pip 會在兩個 index 間挑「版本最高」，把 8GB 的 `nvidia-*` CUDA libs
從慢的 Fastly 拉而不是快的 `pypi.nvidia.com`。

---

## 8 卡才會炸的三個雷（單卡測不出來）

1. **conda 的 libstdc++ 只到 GLIBCXX_3.4.29**，系統 `libnccl.so.2` 需要 3.4.30。
   vLLM 的 pynccl 用 `ctypes.CDLL("libnccl.so.2")` 按名稱載入 → TP>1 時直接崩。
   修法：把 conda 的 `libstdc++.so.6` 指向系統版（backup 留在同目錄）。

2. **pynccl 撿到系統 NCCL 2.28.3**，與 torch 的 2.28.9 版本錯開。
   修法：明確設定 `VLLM_NCCL_SO_PATH` 指向 pip 版。

3. **`~/.bashrc` 第 6 行 `[ -z "$PS1" ] && return`** — 非互動 shell 直接返回。
   放在檔案結尾的 export 對 `vllm serve`（從腳本 / screen 啟動）**完全無效**，會導致：
   - `has_flashinfer` 靜默變 False → NVFP4 MoE 走慢路徑，誤判成模型本身慢
   - `HF_HOME` 失效 → 433GB 權重下到 30GB 的 system disk，中途爆盤

   修法：環境變數同時寫 `/etc/profile.d/` 與 `/etc/environment`；
   nvcc 改用 `/usr/local/bin/` wrapper、函式庫走 `/etc/ld.so.conf.d/` + ldconfig，
   讓關鍵路徑完全不依賴環境變數。

---

## 其他踩過的坑

### 套件版本

- **vLLM 0.26.0 硬性要求 `flashinfer-python==0.6.14`**。為了配合只到 0.6.13 的
  flashinfer-cubin 而降版，會把 torch 拖到 2.10.0 並拉進整套 cu12 套件 →
  `undefined symbol: torch_from_blob`。不要降版。
- **nvcc 版本必須對齊 cuda-toolkit 13.0.x**。裝最新的 13.3 會報
  "CUDA compiler and CUDA toolkit headers are incompatible"。
- **cu12/cu13 套件裝在同一個目錄**，`pip uninstall` 某個 cu12 套件會連帶刪掉
  cu13 的 `.so` 檔（pip 仍記錄為已安裝，但檔案沒了）。清理後務必用 `ldd` 驗證，
  不要信 pip metadata。
- 保留 `nvidia-cutlass-dsl-libs-cu12`（vLLM 的 nvidia-cutlass-dsl 依賴它）。

### flashinfer 首次啟動要 JIT 編譯

第一次 `vllm serve` 時 flashinfer 會為 sm_120 即時編譯 kernel，`sampling.cu`
光是 template 實例化就要數分鐘，**冷啟動實測約 10 分鐘**。不知道的話很容易誤判成當機。

產物快取在 `/root/.cache/flashinfer`（6.3MB，**在 system disk → 會進 image**），
之後啟動只要約 1 分鐘。存 image 前**不要**清掉這個目錄，8 卡機開機時可直接省下這段
（8 卡是 $7.30/hr）。

### 下載權重時不要同時裝套件

兩者搶頻寬會產生連鎖失敗：pip 被壓到 32 kB/s → `ReadTimeoutError` →
半截檔留在快取 → 下次重跑變成 **wheel 雜湊不符**（`THESE PACKAGES DO NOT MATCH THE HASHES`）。
遇到就 `pip cache purge` 再重跑。先裝環境、再下權重，或反過來，別並行。

### Xet 下載行為

huggingface_hub ≥ 1.x 用 Xet 傳輸，`hf_transfer` 已棄用（設了只會噴警告）。行為特點：

- **chunk 先聚在記憶體，攢滿整個 ~9GB 分片才落地**。所以短時間量 `du` 會看到 0 MB/s
  而網路仍是滿速——不是卡住。要判斷是否存活請看網路計數器或 `.incomplete` 檔。
- 中斷過的下載會在 `.cache/huggingface/download/` 留下**孤兒 `.incomplete` 檔**，
  即使後續重跑成功也不會清掉。單純數個數會把陳年殘留誤判成「下載未完成」。
  清理：`find <local-dir>/.cache/huggingface/download -name '*.incomplete' -mmin +60 -delete`
- `--local-dir` 在 ≥0.23 是直接下載到目標目錄，不會再經 cache 複製一份（不需雙倍空間）。

### 無卡模式（No-GPU mode）配額很緊

AutoDL/GPUhub 的無卡模式適合下載權重（費率極低），但實測配額是
**RAM 2GB / CPU 0.5 core**，而 Claude Code 之類的工具自己就吃掉約 900MB →
`hf download` 會被 SIGKILL（`EXIT=137`）。要用無卡模式下載就別在同一台跑別的東西，
或直接用單卡開機（$0.91/hr × 8hr ≈ $7）。

---

## 8 卡部署與 benchmark

```bash
nvidia-smi topo -m      # 先看 PCIe/NUMA topology

screen -S serve
vllm serve /root/autodl-tmp/glm-5.2-nvfp4 --served-model-name glm-5.2 \
  -tp 8 --max-model-len 131072 --gpu-memory-utilization 0.92
# 載入 433GB 權重需數分鐘，log 出現 "Application startup complete" 才算好
```

若 topology 顯示 8 卡跨兩個 NUMA node（4+4），額外測對照組 `-tp 4 -pp 2`。

```bash
screen -S bench
vllm bench serve --base-url http://localhost:8000 --model glm-5.2 \
  --dataset-name random --random-input-len 4096 --random-output-len 512 \
  --num-prompts 64 --max-concurrency 8
```

建議記錄：prefill / decode throughput、TTFT p50/p99、ITL p50/p99；
並發 1 / 8 / 32 三組；TP=8 vs TP=4+PP=2；`nvidia-smi dmon` 抽樣的功耗與利用率。

### 接到本機的 coding agent

```bash
# 本機執行
ssh -p <PORT> -L 8000:localhost:8000 root@<YOUR-INSTANCE-HOST>
```

OpenAI-compatible provider：`base_url = http://localhost:8000/v1`、`model = glm-5.2`、
api key 隨意填。

---

## 升配 8 卡：兩條路

| 方式 | data disk | 說明 |
|---|---|---|
| 同 host「更改配置」 | 保留 | 最省事，但要該 host 剛好有 8 卡空著 |
| **Clone instance** | **一起搬** | 沒資源時用這個，433GB 權重不用重下 |
| Save image → 新實例 | **不會帶** | image 只有 system disk，權重要重下 |

⚠️ **不要「釋放」實例** —— data disk 會跟著消失。

---

## 把 image 公開分享

image 只包含 system disk（環境），**不含 data disk 的權重**，所以拿到的人仍需自行下載 433GB。

公開前務必跑 `sanitize-for-image.sh`（**從純 SSH 執行，不要在 Claude Code 內跑**）：

```bash
bash sanitize-for-image.sh --dry-run   # 先看會刪什麼
bash sanitize-for-image.sh
```

它處理的是 `/logout` 清不掉的東西：`~/.claude.json`（accountUuid / email /
organizationName / billing）、`~/.claude/projects/`（完整對話逐字稿）、
`~/.claude/.credentials.json`（可直接使用你訂閱額度的 OAuth token）、SSH 私鑰、git 身分。

刪掉本機私鑰後，記得再到 GitHub → repo → Settings → Deploy keys 移除對應公鑰。

---

## 成本參考

| 項目 | 費率 | 實際 |
|---|---|---|
| 單卡（環境 + 下載權重） | $0.91/hr | ~9hr ≈ $8 |
| 八卡（部署 + benchmark） | $7.30/hr | 3–5hr ≈ $22–37 |
| Data disk 600GB | ~$0.1/GB/day 級 | 依天數 |
| **總計** | | **~$30–45** |

單卡能驗完的事就不要留到 8 卡做——`verify-weights.sh` 存在的理由就是這個。

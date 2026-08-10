# Phase 1 完成報告 — GLM-5.2 on 8x RTX PRO 6000 Blackwell

實測環境：GPUhub Singapore-A 單卡 RTX PRO 6000 Blackwell SE 96GB
driver 595.58.03 / CUDA 13.2 / Ubuntu 22.04 / sm_120

## 最終環境

| 元件 | 版本 |
|---|---|
| torch | 2.11.0+cu130 |
| vLLM | 0.26.0 |
| flashinfer-python | 0.6.14 |
| nvcc / cuda-crt / nvvm | 13.0.88（必須與 cuda-toolkit 13.0.x 對齊）|
| NCCL | 2.28.9（pip 版，非系統的 2.28.3）|
| transformers | 5.15.0 |

驗證結果：`GlmMoeDsaForCausalLM` 已支援、NVFP4 原生 FP4 GEMM 在 sm_120 實跑通過、
`has_flashinfer=True`、Qwen3-1.7B 端到端 serve+curl 通過（零環境變數依賴）。

## 量化版本選擇

已驗證 vLLM 0.26.0 的 31 種 quant method 對照 HF 上的 GLM-5.2 量化版本：

| Repo | 大小 | quant_method | 結論 |
|---|---|---|---|
| **nvidia/GLM-5.2-NVFP4** | **464.9 GB** | modelopt (NVFP4) | **選這個** — Blackwell 原生 FP4 |
| cyankiwi/GLM-5.2-AWQ-INT4 | 474.3 GB | compressed-tensors | 備案 |
| QuantTrio/GLM-5.2-Int4-Int8Mix | 405.5 GB | compressed-tensors | 備案（最小）|
| PhalaCloud/GLM-5.2-W4AFP8 | 399.8 GB | w4afp8 | **不可用** — vLLM 0.26 無此 method |
| zai-org/GLM-5.2-FP8 | 755.7 GB | fp8 | 塞得下但沒空間給 KV cache |
| zai-org/GLM-5.2 (BF16) | 1506.7 GB | — | 8x96GB 放不下 |

**data disk 建議 600GB**：465GB 權重 + 下載暫存 + 結果，且留餘裕可換成 405GB 備案重試。

## 網路實測（Singapore-A）

| 來源 | 速度 |
|---|---|
| huggingface.co (IPv4) | 13.4 MB/s 單流 / 14 MB/s 聚合（多流疊不上去）|
| download.pytorch.org (R2) | 8.7–11.5 MB/s |
| pypi.nvidia.com | 8.7 MB/s |
| files.pythonhosted.org (Fastly) | 0.22 MB/s 冷門檔 / 2.95 MB/s 熱門檔 |
| mirror.nju.edu.cn | 6.44 MB/s |
| mirrors.aliyun.com / tencent | 0.02–3 MB/s |

**IPv6 完全不通**（`curl -6` 一律 code=000），但 DNS 預設優先回 AAAA。
→ 必須設 gai.conf 優先 IPv4，否則每條連線都先卡一次。

**不要用 uv**：uv 自帶 hickory-dns，不讀 gai.conf，會一直撞壞掉的 IPv6，
實測 0.2 MB/s vs pip 的 11 MB/s（差約 50 倍）。

465GB @ 14MB/s ≈ **7.9 小時**。

## 8 卡才會炸的三個雷（已修，單卡測不出來）

1. **conda libstdc++ 只到 GLIBCXX_3.4.29**，系統 libnccl.so.2 需要 3.4.30。
   vLLM 的 pynccl 用 `ctypes.CDLL("libnccl.so.2")` 按名稱載入 → TP>1 時直接崩。
   修法：把 conda 的 libstdc++.so.6 指向系統版（backup 留在同目錄）。

2. **pynccl 撿到系統 NCCL 2.28.3**，與 torch 的 2.28.9 版本錯開。
   修法：明確設定 `VLLM_NCCL_SO_PATH` 指向 pip 版。

3. **`~/.bashrc` 第 6 行 `[ -z "$PS1" ] && return`** — 非互動 shell 直接返回。
   放在檔案結尾的 export 對 `vllm serve`（從腳本/screen 啟動）完全無效，會導致：
   - `has_flashinfer` 靜默變 False → NVFP4 MoE 走慢路徑，誤判成模型本身慢
   - `HF_HOME` 失效 → 465GB 權重下到 30GB 的 system disk，中途爆盤

   修法：環境變數放在 guard 之前，並同時寫 `/etc/profile.d/` 與 `/etc/environment`；
   nvcc 改用 `/usr/local/bin/` wrapper、函式庫走 `/etc/ld.so.conf.d/` + ldconfig，
   讓關鍵路徑完全不依賴環境變數。

## 其他踩過的坑

- **vLLM 0.26.0 硬性要求 flashinfer-python==0.6.14**。為了配合只到 0.6.13 的
  flashinfer-cubin 而降版，會把 torch 拖到 2.10.0 並拉進整套 cu12 套件 →
  `undefined symbol: torch_from_blob`。不要降版。
- **nvcc 版本必須對齊 cuda-toolkit 13.0.x**。裝最新的 13.3 會報
  "CUDA compiler and CUDA toolkit headers are incompatible"。
- **cu12/cu13 套件裝在同一個目錄**，`pip uninstall` 某個 cu12 套件會連帶刪掉
  cu13 的 .so 檔（pip 仍記錄為已安裝，但檔案沒了）。清理後務必用 `ldd` 驗證，
  不要信 pip metadata。
- 保留 `nvidia-cutlass-dsl-libs-cu12`（vLLM 的 nvidia-cutlass-dsl 依賴它）。

## 下一步

1. 關機 → 保存鏡像 → data disk 擴到 600GB → 開機（**image 存完才能開機**）
2. 下載權重（單卡 $0.91/hr，約 8 小時）：
   ```
   hf download nvidia/GLM-5.2-NVFP4 --local-dir /root/autodl-tmp/glm-5.2-nvfp4
   ```
3. 升到 8 卡：優先同 host「更改配置」；卡被佔走就用 **Clone instance**
   （clone 會連 data disk 一起搬，權重不用重下；save image 不會）
4. 起服務：
   ```
   vllm serve /root/autodl-tmp/glm-5.2-nvfp4 --served-model-name glm-5.2 \
     -tp 8 --max-model-len 131072 --gpu-memory-utilization 0.92
   ```
   先跑 `nvidia-smi topo -m`，若 8 卡跨兩個 NUMA node (4+4)，加測 `-tp 4 -pp 2` 對照組。

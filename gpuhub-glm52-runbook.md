# GPUhub 8x RTX PRO 6000 — GLM-5.2 部署測試 Runbook

> 交給 Claude Code 執行用。目前狀態：Singapore-A / G117-R 單卡實例已開機（RTX PRO 6000 Blackwell SE 96GB, driver 595.58.03, CUDA 13.2, Ubuntu 22.04, 25 vCPU / 120GB RAM）。
>
> **執行環境**：Claude Code 直接跑在 GPUhub 實例上（`root@gpuhub-container-*`）。

## 磁碟規則（最重要的環境常識）

| 路徑 | 角色 | 特性 |
|---|---|---|
| `/` (30GB) | System Disk | **會存進 image**。裝環境用，空間極小，勿放大檔 |
| `/root/autodl-tmp` (50GB, 可擴) | Data Disk | **不會存進 image**，綁定實體 host。放權重、快取 |

原則：要進 image 的東西（conda env、pip 套件）放 system disk；一次性大檔（模型權重、HF cache、pip cache）放 data disk。

## 給 Claude Code 的約束

1. 長時間指令（下載、benchmark）一律在 `screen` session 內執行，session 命名 `dl` / `serve` / `bench`。
2. 這台機器按秒計費。**不要**閒置等待——需要人工操作 console（儲值、關機、存 image、升配）的步驟，停下來明確告知使用者，不要自己輪詢空轉。
3. 任何憑證（HF token、API key）只放在 data disk 或環境變數，**不寫入** `~/.bashrc`、shell history 或 system disk 上的設定檔（會被烤進 image）。
4. `pip install` 若遇 CUDA wheel 不相容，先試 vLLM 官方 extra index 的 cu12x wheel（driver 595 向下相容），不要貿然從 source 編譯（耗時且 30GB system disk 可能不夠）。
5. 不掃描、不外連與任務無關的服務；此為租用機器。

---

## Phase 1 — 單卡：環境安裝與驗證

```bash
# 1.1 快取全部導向 data disk
cat >> ~/.bashrc << 'EOF'
export PIP_CACHE_DIR=/root/autodl-tmp/pip-cache
export HF_HOME=/root/autodl-tmp/hf-home
export HF_HUB_ENABLE_HF_TRANSFER=1
EOF
source ~/.bashrc

# 1.2 安裝 vLLM（進內建 miniconda base env → system disk → 會進 image）
pip install -U vllm hf_transfer

# 1.3 驗證 torch/CUDA
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 1.4 小模型端到端驗證
screen -S serve
vllm serve Qwen/Qwen3-1.7B --max-model-len 8192
# (detach: Ctrl-A D)

curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-1.7B","messages":[{"role":"user","content":"hi"}]}' | head -c 500
```

**通過標準**：curl 回傳正常 completion。之後 `screen -S serve -X quit` 停掉服務。

### 1.5 存 image 前的清理

```bash
pip cache purge
rm -f ~/.cache/huggingface/token        # 若存在（HF_HOME 已改，通常不在這）
history -c && rm -f ~/.bash_history
# 確認 system disk 用量 < 25GB
df -h /
```

**⏸ 停：請使用者到 console 操作**
1. 關機（Shutdown）
2. 實例列表 → 保存鏡像（Save Image）→ 存入 My Images
3. Operate → More → 尋找同 host「更改配置 / Change Configuration」選項，確認可否升配 8 卡
   - **有** → 走 Phase 2A（單卡下載，最省錢）
   - **沒有** → 走 Phase 2B（8 卡直接下載）

---

## Phase 2A — 同 host 可升配：單卡先下載權重（$0.91/hr）

**⏸ 使用者先在 console**：關機狀態下把 data disk 擴到 **600GB**，再開機（維持單卡）。

```bash
# 2A.1 下載 GLM-5.2 W4 權重（repo 名以實際 HF 上的量化版本為準，先查證）
screen -S dl
hf download <GLM-5.2-W4-REPO> --local-dir /root/autodl-tmp/glm-5.2-w4
# (detach: Ctrl-A D；預計 30–90 分鐘，可 screen -r dl 查進度)

# 2A.2 完整性確認
du -sh /root/autodl-tmp/glm-5.2-w4
ls /root/autodl-tmp/glm-5.2-w4/*.safetensors | wc -l   # 對照 repo 檔案數
```

**⏸ 停：使用者到 console** → 關機 → 同 host 升配至 **8x GPU** → 開機 → 進 Phase 3。

## Phase 2B — 不可升配：直接開 8 卡下載

**⏸ 使用者到 console**：釋放或保留單卡實例 → Create Instance：8x RTX PRO 6000、data disk 擴 600GB、Image 選 **My Images** 裡剛存的 → 開機後在 8 卡實例上執行 2A.1–2A.2。

---

## Phase 3 — 8 卡：GLM-5.2 部署與 benchmark

```bash
# 3.1 驗機
nvidia-smi                              # 應為 8x 96GB
nvidia-smi topo -m                      # 記錄 PCIe/NUMA topology → 貼給使用者判讀

# 3.2 起服務（TP=8 為 baseline）
screen -S serve
vllm serve /root/autodl-tmp/glm-5.2-w4 \
  --served-model-name glm-5.2 \
  -tp 8 --max-model-len 131072 \
  --gpu-memory-utilization 0.92
# 載入 400GB 權重需數分鐘，log 出現 "Application startup complete" 才算好
```

若 `nvidia-smi topo -m` 顯示 8 卡跨兩個 NUMA node（4+4），額外測對照組：`-tp 4 -pp 2`。

```bash
# 3.3 Throughput benchmark（vLLM 內建）
screen -S bench
vllm bench serve \
  --base-url http://localhost:8000 --model glm-5.2 \
  --dataset-name random \
  --random-input-len 4096 --random-output-len 512 \
  --num-prompts 64 --max-concurrency 8
```

**記錄項目**（輸出成 `/root/autodl-tmp/results/bench-$(date +%m%d-%H%M).txt`）：
- Prefill / decode throughput (tok/s)、TTFT p50/p99、ITL p50/p99
- 三組並發：`--max-concurrency 1 / 8 / 32`
- TP=8 vs TP=4+PP=2（若有測）
- `nvidia-smi dmon` 抽樣的功耗與利用率

## Phase 4 — openCode 接入

服務埠 8000 在容器內。使用者本機開 tunnel：

```bash
# 使用者本機執行（port 依實際 SSH 連線資訊）
ssh -p <PORT> -L 8000:localhost:8000 root@<YOUR-INSTANCE-HOST>
```

openCode 設定 OpenAI-compatible provider：`base_url = http://localhost:8000/v1`、`model = glm-5.2`、api key 隨意填。跑實際 agentic coding 任務（**僅用公開 repo**），主觀記錄 tool-calling 穩定性與體感速度。

## Phase 5 — 收尾

```bash
# 結果檔拉回本機（使用者本機執行）
scp -P <PORT> root@<YOUR-INSTANCE-HOST>:/root/autodl-tmp/results/* ./
```

**⏸ 使用者到 console**：關機。若近期不再測 → 直接釋放實例（data disk 一起消失，留著也是 15 天後釋放；權重可重下，不值得養）。

---

## 成本備忘

| 項目 | 費率 | 預估 |
|---|---|---|
| Phase 1 單卡 | $0.91/hr | ~1hr ≈ $1 |
| Phase 2A 下載 | $0.91/hr | 1–2hr ≈ $2 |
| Phase 3–4 八卡 | $7.30/hr | 3–5hr ≈ $22–37 |
| Data disk 600GB | ~$0.1/GB/day 級 | 依實際天數 |
| **總計** | | **~$30–45** |

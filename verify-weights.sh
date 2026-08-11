#!/bin/bash
# =============================================================================
# GLM-5.2 NVFP4 權重完整性 + vLLM 環境驗證（單卡可跑，不需載入權重）
#
# 用途：在升配 8 卡（$7.30/hr）之前，先在單卡（$0.91/hr）確定
#       「權重沒下壞」且「vLLM 認得這個模型」，避免昂貴的失敗。
#
# 用法: bash verify-weights.sh
# =============================================================================
set -uo pipefail

MODEL=/root/autodl-tmp/glm-5.2-nvfp4
fail=0
note() { printf "\n\033[1m=== %s ===\033[0m\n" "$1"; }
ok()   { echo "  ✅ $1"; }
bad()  { echo "  ❌ $1"; fail=1; }

note "[1/5] 下載完整性"
# .incomplete 是 Xet 的暫存檔。被中斷的下載會留下孤兒檔，即使後續重跑已經成功，
# 它們也不會被清掉 —— 單純數個數會把「陳年殘留」誤判成「下載未完成」。
# 用修改時間區分：60 分鐘內動過的才代表下載仍在進行。
STAGE="$MODEL/.cache/huggingface/download"
active=$(find "$STAGE" -name "*.incomplete" -mmin -60 2>/dev/null | wc -l)
stale=$(find "$STAGE" -name "*.incomplete" -mmin +60 2>/dev/null | wc -l)
[ "$active" -eq 0 ] && ok "無進行中的下載" || bad "有 $active 個 .incomplete 仍在寫入，下載尚未結束"
[ "$stale" -gt 0 ] && echo "  ℹ️  另有 $stale 個逾一小時未動的孤兒暫存檔（$(du -shc "$STAGE"/*.incomplete 2>/dev/null | tail -1 | cut -f1)），可安全刪除：
       find $STAGE -name '*.incomplete' -mmin +60 -delete"

shards=$(ls "$MODEL"/*.safetensors 2>/dev/null | wc -l)
[ "$shards" -eq 47 ] && ok "分片數 47/47" || bad "分片數 $shards/47"

note "[2/5] index.json 對照（每個 weight_map 提到的檔案都要存在且非空）"
python - <<'EOF'
import json, os, sys
M = "/root/autodl-tmp/glm-5.2-nvfp4"
idx = os.path.join(M, "model.safetensors.index.json")
if not os.path.exists(idx):
    print("  ❌ 找不到 model.safetensors.index.json"); sys.exit(1)
wm = json.load(open(idx))["weight_map"]
files = sorted(set(wm.values()))
missing = [f for f in files if not os.path.exists(os.path.join(M, f))]
empty   = [f for f in files if os.path.exists(os.path.join(M, f)) and os.path.getsize(os.path.join(M, f)) == 0]
print(f"  index 宣告 {len(files)} 個分片、{len(wm)} 個 tensor")
if missing: print(f"  ❌ 缺少 {len(missing)} 個: {missing[:3]}")
elif empty: print(f"  ❌ {len(empty)} 個檔案為空: {empty[:3]}")
else:       print("  ✅ 全部存在且非空")
sys.exit(1 if (missing or empty) else 0)
EOF
[ $? -ne 0 ] && fail=1

note "[3/5] safetensors 標頭可解析（抽驗 3 個分片，只讀標頭不載入權重）"
python - <<'EOF'
import glob, json, struct, sys, os
M = "/root/autodl-tmp/glm-5.2-nvfp4"
files = sorted(glob.glob(os.path.join(M, "*.safetensors")))
picks = [files[0], files[len(files)//2], files[-1]] if len(files) >= 3 else files
bad = False
for f in picks:
    try:
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        size = os.path.getsize(f)
        end = max((v["data_offsets"][1] for k, v in hdr.items() if k != "__metadata__"), default=0)
        if 8 + n + end != size:
            print(f"  ❌ {os.path.basename(f)}: 標頭宣告 {8+n+end} != 實際 {size}（檔案被截斷）"); bad = True
        else:
            print(f"  ✅ {os.path.basename(f)}: {len(hdr)-1} tensors, {size/1024**3:.1f} GB")
    except Exception as e:
        print(f"  ❌ {os.path.basename(f)}: 標頭解析失敗 {e}"); bad = True
sys.exit(1 if bad else 0)
EOF
[ $? -ne 0 ] && fail=1

note "[4/5] vLLM 認得這個模型嗎（只讀 config，不配置 GPU 記憶體）"
python - <<'EOF'
import json, sys
M = "/root/autodl-tmp/glm-5.2-nvfp4"
cfg = json.load(open(f"{M}/config.json"))
arch = cfg.get("architectures", ["?"])[0]
print(f"  architecture   {arch}")
print(f"  hidden/layers  {cfg.get('hidden_size')} / {cfg.get('num_hidden_layers')}")
try:
    q = json.load(open(f"{M}/hf_quant_config.json"))
    print(f"  quant config   {json.dumps(q.get('quantization', q))[:80]}")
except FileNotFoundError:
    print("  quant config   (無 hf_quant_config.json)")

from vllm.model_executor.models.registry import ModelRegistry
supported = arch in ModelRegistry.get_supported_archs()
print(f"  {'✅' if supported else '❌'} vLLM {'支援' if supported else '不支援'} {arch}")

from vllm.utils.flashinfer import has_flashinfer
hf_ok = has_flashinfer()
print(f"  {'✅' if hf_ok else '❌'} has_flashinfer = {hf_ok}  (NVFP4 MoE 快路徑必需)")

import torch, ctypes
cap = torch.cuda.get_device_capability(0)
print(f"  ✅ torch {torch.__version__}  device sm_{cap[0]}{cap[1]}")
ctypes.CDLL("libnccl.so.2")
print(f"  ✅ NCCL {torch.cuda.nccl.version()} 可由名稱載入（TP>1 必需）")
sys.exit(0 if (supported and hf_ok) else 1)
EOF
[ $? -ne 0 ] && fail=1

note "[5/5] 小模型端到端 serve（驗證 vLLM 本身，非 GLM-5.2）"
echo "  啟動 Qwen3-1.7B..."
# 首次啟動時 flashinfer 要為 sm_120 即時編譯 kernel（sampling.cu 的 template
# 實例化量很大，單檔就要數分鐘）。實測冷啟動 ~10 分鐘，7.5 分鐘上限會誤判成失敗。
# 編譯產物快取在 /root/.cache/flashinfer（system disk → 會進 image），之後啟動只要 ~1 分鐘。
SERVE_LOG=/root/autodl-tmp/serve-test.log
: > "$SERVE_LOG"
nohup vllm serve Qwen/Qwen3-1.7B --max-model-len 8192 --port 8000 \
      > "$SERVE_LOG" 2>&1 &
serve_pid=$!
# 等成功、失敗、或程序死亡 —— 只等成功訊號的話，崩潰跟「還在編譯」看起來一模一樣
for i in $(seq 1 240); do
    grep -qE "Application startup complete|EngineCore failed|Engine core initialization failed" "$SERVE_LOG" 2>/dev/null && break
    kill -0 $serve_pid 2>/dev/null || break
    sleep 5
done
if grep -q "Application startup complete" "$SERVE_LOG" 2>/dev/null; then
    resp=$(curl -s -m 60 http://localhost:8000/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{"model":"Qwen/Qwen3-1.7B","messages":[{"role":"user","content":"say hi"}],"max_tokens":20}')
    echo "$resp" | grep -q '"content"' && ok "curl 取得 completion" || bad "curl 回應異常: $(echo "$resp" | head -c 200)"
else
    bad "serve 未就緒，錯誤: $(tr '\r' '\n' < "$SERVE_LOG" | grep -iE 'RuntimeError|EngineCore failed' | tail -1 | cut -c1-160)"
    echo "     完整 log: $SERVE_LOG"
fi
kill $serve_pid 2>/dev/null; sleep 3; kill -9 $serve_pid 2>/dev/null

note "結果"
if [ $fail -eq 0 ]; then
    echo "  全部通過 — 可以安全升配 8 卡"
    echo "  8 卡上執行: vllm serve $MODEL --served-model-name glm-5.2 \\"
    echo "                -tp 8 --max-model-len 131072 --gpu-memory-utilization 0.92"
else
    echo "  有項目失敗，先修掉再升配（8 卡 \$7.30/hr）"
fi
exit $fail

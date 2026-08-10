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
inc=$(ls "$MODEL"/.cache/huggingface/download/*.incomplete 2>/dev/null | wc -l)
[ "$inc" -eq 0 ] && ok "無未完成的 .incomplete 暫存檔" || bad "還有 $inc 個 .incomplete，下載尚未結束"

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
pkill -f "vllm serve" 2>/dev/null; sleep 2
nohup vllm serve Qwen/Qwen3-1.7B --max-model-len 8192 --port 8000 \
      > /root/autodl-tmp/serve-test.log 2>&1 &
for i in $(seq 1 90); do
    grep -q "Application startup complete" /root/autodl-tmp/serve-test.log 2>/dev/null && break
    sleep 5
done
if grep -q "Application startup complete" /root/autodl-tmp/serve-test.log 2>/dev/null; then
    resp=$(curl -s -m 60 http://localhost:8000/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{"model":"Qwen/Qwen3-1.7B","messages":[{"role":"user","content":"say hi"}],"max_tokens":20}')
    echo "$resp" | grep -q '"content"' && ok "curl 取得 completion" || bad "curl 回應異常: $(echo "$resp" | head -c 200)"
else
    bad "serve 未在 7.5 分鐘內就緒，見 /root/autodl-tmp/serve-test.log"
fi
pkill -f "vllm serve" 2>/dev/null

note "結果"
if [ $fail -eq 0 ]; then
    echo "  全部通過 — 可以安全升配 8 卡"
    echo "  8 卡上執行: vllm serve $MODEL --served-model-name glm-5.2 \\"
    echo "                -tp 8 --max-model-len 131072 --gpu-memory-utilization 0.92"
else
    echo "  有項目失敗，先修掉再升配（8 卡 \$7.30/hr）"
fi
exit $fail

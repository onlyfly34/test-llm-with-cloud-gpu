#!/bin/bash
# =============================================================================
# GLM-5.2 on RTX PRO 6000 Blackwell (sm_120) — 環境重建腳本
# 從 GPUhub/AutoDL 乾淨的 base image (Ubuntu 22.04 + miniconda py3.12 + driver 595) 重建
# 實測環境：Singapore-A, driver 595.58.03, CUDA 13.2, torch 預裝 2.12.1+cu130
#
# 用法: bash rebuild-env.sh 2>&1 | tee rebuild.log
# 耗時: 約 20-30 分鐘（大部分是下載）
# =============================================================================
set -euo pipefail

CONDA_SP=/root/miniconda3/lib/python3.12/site-packages
CU13=$CONDA_SP/nvidia/cu13
DATA=/root/autodl-tmp

echo "=== [1/8] 網路修正：IPv6 不通，強制優先 IPv4 ==="
# GPUhub SG 的 IPv6 完全不通 (curl -6 一律 code=000)，但 DNS 預設優先回 AAAA
# 不修的話每條連線都要先卡一次才 fallback，pip/uv 會慢到不可用
grep -q "^precedence ::ffff:0:0/96  100" /etc/gai.conf 2>/dev/null || \
    echo "precedence ::ffff:0:0/96  100" >> /etc/gai.conf

echo "=== [2/8] 快取導向 data disk（system disk 只有 30GB，會進 image）==="
mkdir -p $DATA/{pip-cache,hf-home,tmp}
export PIP_CACHE_DIR=$DATA/pip-cache TMPDIR=$DATA/tmp

echo "=== [3/8] torch 2.11.0+cu130（vLLM 0.26.0 硬性 pin）==="
# 關鍵：不要加 --extra-index-url pypi.org，否則 nvidia-* CUDA libs 會從
# files.pythonhosted.org (Fastly, ~0.2MB/s) 下載而不是 pypi.nvidia.com (~9MB/s)
pip install --index-url https://download.pytorch.org/whl/cu130 \
    torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0

echo "=== [4/8] vLLM 0.26.0 + HF 工具 ==="
pip install vllm==0.26.0 "huggingface_hub[cli]"

echo "=== [5/8] flashinfer（GLM-5.2 NVFP4 MoE 路徑必需）==="
# 務必用 0.6.14 — vLLM 0.26.0 硬性要求。降到 0.6.13 會把 torch 拖到 2.10.0
# 並拉進一整套 cu12 套件，導致 NCCL 衝突與 vLLM C 擴充載入失敗。
pip install "flashinfer-python==0.6.14"

echo "=== [6/8] nvcc（flashinfer 靠 JIT 編譯 kernel，vLLM 只帶 runtime 不帶 compiler）==="
# 版本必須對齊 cuda-toolkit 13.0.x，用 13.3 會報
# "CUDA compiler and CUDA toolkit headers are incompatible"
pip install --index-url https://pypi.nvidia.com --extra-index-url https://pypi.org/simple \
    "nvidia-cuda-nvcc==13.0.88" "nvidia-cuda-crt==13.0.88" "nvidia-nvvm==13.0.88"

echo "=== [7/8] 系統層設定（不依賴環境變數，因為 ~/.bashrc 非互動 shell 不執行）==="
# 7a. nvcc wrapper：直接 symlink 會讓 nvcc 用 /usr/local/bin/../include 找標頭檔而失敗，
#     必須用 wrapper 以真實路徑執行，並補上 -L 讓 -lcudart_static/-lcudadevrt 找得到
cat > /usr/local/bin/nvcc << EOF
#!/bin/sh
CU=$CU13
exec "\$CU/bin/nvcc" "\$@" -L"\$CU/lib"
EOF
chmod +x /usr/local/bin/nvcc

# 7b. 標準 CUDA 佈局
ln -sfn $CU13 /usr/local/cuda

# 7c. pip 的 CUDA 包只給 libcudart.so.13，缺 -lcudart 需要的無版本 symlink
ln -sf $CU13/lib/libcudart.so.13 $CU13/lib/libcudart.so
ln -sf $CU13/lib/libcudart.so.13 /usr/local/lib/libcudart.so

# 7d. 執行期函式庫路徑
cat > /etc/ld.so.conf.d/00-nvidia-cu13.conf << EOF
$CU13/lib
$CONDA_SP/nvidia/nccl/lib
EOF
ldconfig

# 7e. conda 的 libstdc++ 只到 GLIBCXX_3.4.29，系統的有 3.4.30。
#     系統 libnccl.so.2 需要 3.4.30 → vLLM 的 pynccl (ctypes.CDLL) 在 TP>1 時會直接崩。
#     這個錯誤單卡永遠測不出來，只有 8 卡才會炸。
if [ ! -L $CONDA_SP/../libstdc++.so.6 ]; then
    mv /root/miniconda3/lib/libstdc++.so.6 /root/miniconda3/lib/libstdc++.so.6.conda-backup
    ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /root/miniconda3/lib/libstdc++.so.6
fi

echo "=== [8/8] 環境變數（三層覆蓋）==="
# ~/.bashrc 第 6 行有 [ -z "$PS1" ] && return，非互動 shell 直接返回，
# 所以 export 必須放在那行「之前」，否則 vllm serve 從腳本啟動時完全不生效。
cat > /etc/profile.d/glm52-env.sh << EOF
export PIP_CACHE_DIR=$DATA/pip-cache
export HF_HOME=$DATA/hf-home
export HF_XET_HIGH_PERFORMANCE=1
export CUDA_HOME=$CU13
export LIBRARY_PATH="\$CUDA_HOME/lib:/usr/local/lib:\${LIBRARY_PATH:-}"
export VLLM_NCCL_SO_PATH=$CONDA_SP/nvidia/nccl/lib/libnccl.so.2
EOF
chmod +x /etc/profile.d/glm52-env.sh

grep -q HF_HOME /etc/environment || cat >> /etc/environment << EOF
HF_HOME=$DATA/hf-home
HF_XET_HIGH_PERFORMANCE=1
CUDA_HOME=$CU13
PIP_CACHE_DIR=$DATA/pip-cache
VLLM_NCCL_SO_PATH=$CONDA_SP/nvidia/nccl/lib/libnccl.so.2
EOF

echo
echo "=== 驗證 ==="
bash -c '
python -c "
import torch, vllm, ctypes
from vllm.utils.flashinfer import has_flashinfer
from vllm.model_executor.models.registry import ModelRegistry
cap = torch.cuda.get_device_capability(0)
ctypes.CDLL(\"libnccl.so.2\")
print(f\"torch          {torch.__version__}\")
print(f\"vllm           {vllm.__version__}\")
print(f\"device         {torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]}\")
print(f\"cuda avail     {torch.cuda.is_available()}\")
print(f\"nccl           {torch.cuda.nccl.version()} (CDLL OK)\")
print(f\"has_flashinfer {has_flashinfer()}\")
print(f\"GLM arch       {\"GlmMoeDsaForCausalLM\" in ModelRegistry.get_supported_archs()}\")
"'
echo
echo "全部通過即可用。下載權重："
echo "  hf download nvidia/GLM-5.2-NVFP4 --local-dir $DATA/glm-5.2-nvfp4"

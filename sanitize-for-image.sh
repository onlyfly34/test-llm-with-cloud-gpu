#!/bin/bash
# =============================================================================
# 公開 image 前的清理 — 移除所有個人身分與憑證痕跡
#
# ⚠️ 這支腳本會登出 Claude Code 並刪除對話紀錄。
#    必須從「純 SSH」執行，不要在 Claude Code 內跑（它會刪掉自己正在用的憑證）。
#
# 用法:  bash sanitize-for-image.sh          # 實際執行
#        bash sanitize-for-image.sh --dry-run # 只列出會刪什麼
# =============================================================================
set -uo pipefail
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1

gone() {
    if [ ! -e "$1" ] && [ ! -L "$1" ]; then echo "  ·  $1 (已無)"; return; fi
    if [ $DRY -eq 1 ]; then echo "  →  會刪除 $1 ($(du -sh "$1" 2>/dev/null | cut -f1))"
    else rm -rf "$1"; echo "  ✅ 已刪除 $1"; fi
}

echo "=== [1/4] Claude Code 憑證與對話紀錄 ==="
# .credentials.json 內含可直接使用你訂閱額度的 OAuth accessToken + refreshToken。
# .claude.json 內含 accountUuid / emailAddress / organizationName / billing 資訊。
# projects/ 是完整對話逐字稿；history.jsonl 是輸入過的 prompt。
# 注意：/logout 只清掉第一項，其餘都會留在 image 裡。
gone /root/.claude
gone /root/.claude.json
gone /root/.cache/claude-cli-nodejs
gone /root/.cache/claude

echo "=== [2/4] SSH 金鑰 ==="
# 私鑰若留在公開 image，任何人都能用它存取你註冊過 deploy key 的 repo。
# 刪除本機私鑰後，記得再到 GitHub → repo → Settings → Deploy keys 移除該公鑰。
gone /root/.ssh/id_ed25519
gone /root/.ssh/id_ed25519.pub
gone /root/.ssh/known_hosts.old
# known_hosts 只有 github.com 的公開 host key，留著無妨且方便使用者

echo "=== [3/4] git 身分 ==="
if [ $DRY -eq 1 ]; then
    echo "  →  會清除 /root/.gitconfig 的 user.name / user.email"
    echo "  →  會清除 repo .git/config 的 user 區段"
else
    git config --global --unset-all user.name 2>/dev/null
    git config --global --unset-all user.email 2>/dev/null
    git -C /root/test-llm-with-cloud-gpu config --local --remove-section user 2>/dev/null
    # remote 改回 HTTPS，這樣沒有你金鑰的人也能 pull
    git -C /root/test-llm-with-cloud-gpu remote set-url origin \
        https://github.com/onlyfly34/test-llm-with-cloud-gpu.git 2>/dev/null
    echo "  ✅ git 身分已清除，remote 改回 HTTPS"
fi

echo "=== [4/4] shell 與暫存 ==="
gone /root/.bash_history
gone /root/.python_history
gone /root/.viminfo
if [ $DRY -eq 1 ]; then
    echo "  →  會清除 /tmp ($(du -sh /tmp 2>/dev/null | cut -f1)) 與 pip cache"
else
    find /tmp -maxdepth 1 -mmin +5 -exec rm -rf {} + 2>/dev/null
    pip cache purge >/dev/null 2>&1
    echo "  ✅ /tmp 與 pip cache 已清"
fi

echo
echo "=== 保留（image 的價值所在，請勿刪）==="
echo "  /root/miniconda3          已驗證的 vLLM 環境"
echo "  /root/.cache/flashinfer   sm_120 JIT 產物，省開機時 ~10 分鐘編譯"
echo "  /etc/gai.conf             IPv4 優先（GPUhub SG 的 IPv6 不通）"
echo "  /etc/profile.d/glm52-env.sh, /etc/environment, /etc/ld.so.conf.d/"
echo "  /usr/local/bin/nvcc       flashinfer JIT 需要"
if [ $DRY -eq 1 ]; then
    echo
    echo "(dry-run，未實際刪除。加 --dry-run 以外的方式執行才會真正清理)"
    exit 0
fi

echo
echo "=== 殘留檢查 ==="
found=0
for p in /root/.claude /root/.claude.json /root/.ssh/id_ed25519 /root/.bash_history; do
    [ -e "$p" ] && { echo "  ❌ 仍存在: $p"; found=1; }
done
grep -rlEi "hf_[A-Za-z0-9]{20}|gh[pousr]_[A-Za-z0-9]{20}" /root/.bashrc /etc/environment /etc/profile.d/ 2>/dev/null \
    && { echo "  ❌ 設定檔含 token"; found=1; }
[ $found -eq 0 ] && echo "  ✅ 無殘留"
echo
echo "system disk: $(df -h / | tail -1 | awk '{print $3" / "$2}')"

#!/usr/bin/env bash
# =============================================================================
# 01_sanity_cold_warm.sh — Phase 1: cold-vs-warm LMCache hit smoke test
# =============================================================================
#
# SETTING
#   The smallest possible end-to-end check that the LMCache L2 (CPU-DRAM) tier is
#   actually storing and reusing KV cache. Talks to the running MiniMax-M2.5
#   server (start it with 00_serve_minimax.sh first).
#
# WHAT IT TESTS / WHY
#   Sends the SAME large prompt TWICE:
#     PASS 1 (cold):  prefix has never been seen -> LMCache hit tokens = 0,
#                     server stores the KV blocks to CPU DRAM ("Stored N tokens").
#     PASS 2 (warm):  identical prefix -> LMCache hit tokens > 0 (reused from L2).
#   This is the blog's "success gate": if warm hits == prompt prefix tokens, the
#   connector + hash keys + L2 tier are all wired correctly. At TP>1 it also
#   proves cross-worker key agreement (the hash fix in common.env).
#
# WHAT TO LOOK FOR
#   - Client side: PASS 2 reports cached_tokens > 0 (needs server's
#     --enable-prompt-tokens-details, which 00_serve_minimax.sh sets).
#   - AUTHORITATIVE signal: the SERVER log lines
#       "LMCache hit tokens: 0"   (pass 1)
#       "LMCache hit tokens: N"   (pass 2, N = prefix length)
#       "Stored N of N tokens, X GB, Y GB/s"  (offload bandwidth to CPU DRAM)
#     Tail it:  tail -f $LOG_DIR/server_minimax_*.log | grep -i lmcache
#
# USAGE
#   ./01_sanity_cold_warm.sh
#   PORT=8801 ./01_sanity_cold_warm.sh        # different server
#   PREFIX_REPS=400 ./01_sanity_cold_warm.sh  # bigger prefix (more tokens cached)
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/${COMMON_ENV:-common.env}"

EP="$ENDPOINT/v1/chat/completions"
PREFIX_REPS="${PREFIX_REPS:-200}"   # ~200 reps ≈ a few-thousand-token prefix

echo "Endpoint: $EP   served-name: $SERVED_NAME   prefix reps: $PREFIX_REPS"

# Build a long, FIXED prefix so there is a real, reusable prefix to cache.
PREFIX=$(python3 - "$PREFIX_REPS" <<'PY'
import sys
reps = int(sys.argv[1])
base = ("The quick brown fox jumps over the lazy dog. "
        "Caching key-value tensors avoids recomputation across turns. ")
print((base * reps).strip())
PY
)

req() {
  curl -s "$EP" -H 'Content-Type: application/json' -d "$(python3 - "$PREFIX" "$SERVED_NAME" <<'PY'
import json, sys
prefix, served = sys.argv[1], sys.argv[2]
print(json.dumps({
  "model": served,
  "messages": [
    {"role": "system", "content": prefix},
    {"role": "user",   "content": "In one sentence, what is a KV cache?"}
  ],
  "max_tokens": 40, "temperature": 0.0, "stream": False
}))
PY
)"
}

show() {
  python3 -c "
import sys, json
d = json.load(sys.stdin)
u = d.get('usage', {}) or {}
det = u.get('prompt_tokens_details') or {}
print('  prompt_tokens:', u.get('prompt_tokens'), '| cached_tokens:', det.get('cached_tokens'))
ch = d.get('choices', [{}])
msg = (ch[0].get('message') or {}).get('content', '')
print('  resp:', msg[:120])
"
}

echo "=== PASS 1 (cold) ==="
req | show 2>&1

sleep 2
echo
echo "=== PASS 2 (warm — expect cached_tokens > 0) ==="
req | show 2>&1

echo
echo "NOTE: confirm with the server log (authoritative):"
echo "  grep -i 'LMCache hit tokens\\|Stored' $LOG_DIR/server_minimax_*.log | tail -6"

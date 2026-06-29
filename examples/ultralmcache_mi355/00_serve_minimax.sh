#!/usr/bin/env bash
# =============================================================================
# 00_serve_minimax.sh — Launch vLLM (+/- LMCache) serving MiniMax-M2.5
# =============================================================================
#
# SETTING
#   Brings up the inference server that ALL benchmark phases (01-04) talk to.
#   This is the gfx950/MI355X port of the blog's MI300X serving stack
#   (blog Section 3 Step 4 + Section 4 "Common settings" + Section 8 "Reproduce").
#       vLLM (in-tree dev build, carries our KV kernels)
#         [+ LMCacheConnectorV1 + LMCache CPU-DRAM L2 tier]   (config C)
#   Model: MiniMax-M2.5, 256-expert MoE (top-8), 62 layers, 196k ctx.
#
# THE BLOG'S THREE CONFIGS (select with KV_CONFIG=A|B|C; the comparison IS the
# experiment — blog Section 2 "Three test configurations", Findings 1 & 5):
#   A  vanilla / no cache   -> --no-enable-prefix-caching        (lower bound)
#   B  HBM prefix cache     -> --enable-prefix-caching           (cheap baseline)
#   C  LMCache CPU-DRAM L2  -> --enable-prefix-caching
#                              + --kv-transfer-config '{LMCacheConnectorV1,kv_both}'
#                              + LMCACHE_* env (default)
#
# BLOG-EXACT KNOBS (from Section 4; defaults live in common.env)
#   TP=2, GPUS (2 free GPUs)         blog used TP=2 on 2x MI300X
#   GPU_MEM_UTIL=0.85 base / 0.78 stress   (pass GPU_MEM_UTIL=0.78 for stress)
#   LMCACHE_MAX_LOCAL_CPU_SIZE=64, LMCACHE_CHUNK_SIZE=256, LMCACHE_LOCAL_CPU=true
#   --tool-call-parser minimax_m2 --reasoning-parser minimax_m2
#       --enable-auto-tool-choice          (MINIMAX_PARSERS=1, blog Step 4)
#
# THE HASH FIX (mandatory at TP>1 — blog Step 4 mistake #1, Finding 2)
#   PYTHONHASHSEED=0 + (our addition) LMCACHE_PRE_CACHING_HASH_ALGORITHM=sha256 +
#   vLLM --prefix-caching-hash-algo sha256. See common.env / NOTES.md for why our
#   newer vLLM needs the explicit sha256 the blog's pinned vLLM did not.
#
# DEVIATIONS FROM THE BLOG (documented; see README.md "Differences")
#   - Hardware MI355X/gfx950 (blog MI300X/gfx942); ROCm 7.2.1 (blog 7.0.0).
#   - vLLM is our in-tree dev build 0.1.dev15956+gb076282a3 (blog: pinned 0.19.0).
#   - transformers 5.5.4 (blog 4.57.1); LMCache HEAD d9cab92 (blog main ~2026-04).
#   - Model precision is the SAME as the blog (public repo is block-FP8, ~230 GB).
#   - No `docker run` (already in a container).
#   - Explicit sha256 hash algo (blog's pinned vLLM accepted "builtin").
#
# USAGE
#   ./00_serve_minimax.sh                                  # config C, base (gmu 0.85)
#   KV_CONFIG=B ./00_serve_minimax.sh                      # HBM-only baseline
#   KV_CONFIG=A ./00_serve_minimax.sh                      # no-cache lower bound
#   GPU_MEM_UTIL=0.78 ./00_serve_minimax.sh                # STRESS memory pressure
#   GPUS=3,5 ./00_serve_minimax.sh                         # pick free GPUs
#
#   Wait for "Application startup complete" in the log, or:
#     source common.env && wait_for_server
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# COMMON_ENV selects the config overlay. Default = BF16 baseline (common.env).
# TQ44 runs set COMMON_ENV=common_tq44.env (which itself sources common.env and
# then exports the FlyDSL env block + KV_CACHE_DTYPE/BLOCK_SIZE/ATTN_BACKEND/
# COMPILATION_CONFIG so the KV_FLAGS pass-through below switches on TurboQuant).
source "$HERE/${COMMON_ENV:-common.env}"

TAG="${TAG:-${KV_CONFIG}_tp${TP}_gmu${GPU_MEM_UTIL}}"
LOG="$LOG_DIR/server_minimax_${TAG}.log"

# --- GPU mapping safety (MI300/MI355: rocm-smi card idx != HIP device idx) -----
# Preferred: pass CARDS=<physical card list> and let gpu_map.py translate to the
# correct HIP indices. Backward compat: GPUS is still interpreted as HIP indices.
# In all cases, refuse to launch onto a card another job already occupies
# (override with ALLOW_BUSY=1).
GMAP="$(dirname "${BASH_SOURCE[0]}")/gpu_map.py"
if [ -n "${CARDS:-}" ] && [ -f "$GMAP" ]; then
  _hip=$(python3 "$GMAP" cards2hip "$CARDS") || { echo "ERROR: cards2hip failed for CARDS=$CARDS" >&2; exit 1; }
  echo "   GPU map         : CARDS=$CARDS -> HIP=$_hip"
  GPUS="$_hip"
fi
if [ -f "$GMAP" ] && [ "${ALLOW_BUSY:-0}" != "1" ]; then
  if ! python3 "$GMAP" assertfree-hip "$GPUS"; then
    echo "ERROR: target GPUs (HIP=$GPUS) include a card used by another job." >&2
    echo "       Run 'python3 $GMAP table' to pick free GPUs, or set ALLOW_BUSY=1 to override." >&2
    exit 1
  fi
fi

# sanity: GPU list length must equal TP
NGPU=$(awk -F',' '{print NF}' <<<"$GPUS")
if [ "$NGPU" -ne "$TP" ]; then
  echo "ERROR: GPUS='$GPUS' has $NGPU entries but TP=$TP. They must match." >&2
  exit 1
fi
if [ ! -d "$MODEL_PATH" ]; then
  echo "ERROR: model not found at $MODEL_PATH" >&2
  exit 1
fi

# Assemble per-config flags (blog's A/B/C).
CACHE_FLAGS=()
case "$KV_CONFIG" in
  A) CACHE_FLAGS+=( --no-enable-prefix-caching ) ;;
  B) CACHE_FLAGS+=( --enable-prefix-caching --prefix-caching-hash-algo "$PREFIX_HASH_ALGO" ) ;;
  C) CACHE_FLAGS+=( --enable-prefix-caching --prefix-caching-hash-algo "$PREFIX_HASH_ALGO"
                    --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' ) ;;
  *) echo "ERROR: KV_CONFIG must be A, B, or C (got '$KV_CONFIG')" >&2; exit 1 ;;
esac

# MiniMax-M2 parser flags (blog Step 4). Drop with MINIMAX_PARSERS=0.
PARSER_FLAGS=()
if [ "$MINIMAX_PARSERS" = "1" ]; then
  PARSER_FLAGS+=( --tool-call-parser minimax_m2 --reasoning-parser minimax_m2 --enable-auto-tool-choice )
fi

# KV-cache / kernel flags. EMPTY by default => BF16 'auto' path (unchanged).
# common_tq44.env sets KV_CACHE_DTYPE/BLOCK_SIZE/ATTN_BACKEND/COMPILATION_CONFIG
# to switch on the TurboQuant-44 FlyDSL v4 kernel. The FlyDSL VLLM_TQ_* env vars
# are already exported by common_tq44.env and inherited by this exec.
KV_FLAGS=()
[ -n "${KV_CACHE_DTYPE:-}" ]     && KV_FLAGS+=( --kv-cache-dtype "$KV_CACHE_DTYPE" )
[ -n "${BLOCK_SIZE:-}" ]         && KV_FLAGS+=( --block-size "$BLOCK_SIZE" )
[ -n "${ATTN_BACKEND:-}" ]       && KV_FLAGS+=( --attention-backend "$ATTN_BACKEND" )
[ -n "${COMPILATION_CONFIG:-}" ] && KV_FLAGS+=( --compilation-config "$COMPILATION_CONFIG" )

echo "============================================================"
echo " MiniMax-M2.5 server  [KV_CONFIG=$KV_CONFIG]"
case "$KV_CONFIG" in
  A) echo "   strategy         : A = vanilla, NO cache (lower bound)";;
  B) echo "   strategy         : B = HBM prefix cache only (baseline)";;
  C) echo "   strategy         : C = LMCache CPU-DRAM L2 (${LMCACHE_MAX_LOCAL_CPU_SIZE} GB)";;
esac
echo "   model            : $MODEL_PATH"
echo "   TP / GPUs        : $TP / $GPUS"
echo "   gpu-mem-util     : $GPU_MEM_UTIL   max-model-len: $MAX_MODEL_LEN"
echo "   hash algo        : $PREFIX_HASH_ALGO (PYTHONHASHSEED=$PYTHONHASHSEED)"
echo "   minimax parsers  : $MINIMAX_PARSERS"
if [ "${#KV_FLAGS[@]}" -gt 0 ]; then
echo "   kv kernel        : dtype=${KV_CACHE_DTYPE:-auto} block=${BLOCK_SIZE:-default} backend=${ATTN_BACKEND:-default}"
echo "   tq flydsl        : V4=${VLLM_TQ_DECODE_V4:-} SOA_STORE=${VLLM_TQ_SOA_FUSION_STORE:-} BUTTERFLY=${VLLM_TQ_DECODE_V4_WHT_BUTTERFLY:-} AITER=${VLLM_ROCM_USE_AITER:-}"
fi
echo "   port / log       : $PORT / $LOG"
echo "============================================================"

# Record what this server was launched with so client scripts (e.g. Phase 3
# stress_kvmatched) can verify the running server's gmu/config matches the run
# they intend. Keyed by PORT so multiple servers don't clobber each other.
META="$LOG_DIR/server_meta_${PORT}.env"
cat > "$META" <<EOF
SERVER_PORT=$PORT
SERVER_KV_CONFIG=$KV_CONFIG
SERVER_TP=$TP
SERVER_GPUS=$GPUS
SERVER_GPU_MEM_UTIL=$GPU_MEM_UTIL
SERVER_MAX_MODEL_LEN=$MAX_MODEL_LEN
SERVER_STARTED=$(date -Iseconds)
EOF
echo "Wrote server metadata -> $META"

export HIP_VISIBLE_DEVICES="$GPUS"
export VLLM_USE_V1
# Blog Step 4 env (config C uses LMCACHE_*; harmless for A/B since no connector).
export VLLM_FLOAT32_MATMUL_PRECISION=high

exec python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name "$SERVED_NAME" \
  --tensor-parallel-size "$TP" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --trust-remote-code \
  --enable-prompt-tokens-details \
  "${CACHE_FLAGS[@]}" \
  "${PARSER_FLAGS[@]}" \
  "${KV_FLAGS[@]}" \
  --host "$HOST" --port "$PORT" \
  > "$LOG" 2>&1

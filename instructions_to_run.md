# Instructions to run — LMCache KV-offload on MI355X (BF16 / TQ44v4 / fp8g32)

A short, no-Docker guide to serve **MiniMax-M2.5** with **vLLM + LMCache CPU-DRAM KV
offload** on **AMD Instinct MI355X (gfx950, TP=2)** in three KV-cache formats:

| Format | KV cache | Needs |
|---|---|---|
| **BF16** (baseline) | stock 16-bit KV | vLLM branch only |
| **TQ44v4** | 4-bit TurboQuant packed KV | vLLM branch + **FlyDSL** kernels |
| **fp8g32** | UltraQuant FP4-g32 packed KV | vLLM branch only (pure-Triton kernel) |

---

## 1. Get the code

Two repos — the vLLM fork (carries the quantized **KV kernels**) and this LMCache fork
(the **group-aware packed connector** that offloads mixed bf16+4-bit KV):

```bash
# vLLM fork — KV kernels (TurboQuant / FP4 / fp8_g32)
git clone -b flydsl_fp8scalekernel https://github.com/aditi-amd/vllm.git
#   branch: flydsl_fp8scalekernel   (pinned commit 0136dd2ae)

# LMCache fork — this repo, group-aware packed connector
git clone -b feat/ultralmcache-mi355-packed-kv https://github.com/aditi-amd/LMCache.git
#   branch: feat/ultralmcache-mi355-packed-kv   (commit 6e931e7)
```

Branch link: <https://github.com/aditi-amd/vllm/tree/flydsl_fp8scalekernel>

---

## 2. Install (on the MI355X host; ROCm + PyTorch already present)

```bash
# (a) vLLM fork — editable, build against the host's ROCm torch
cd vllm
pip install -e . --no-build-isolation

# (b) LMCache fork — HIP build (PyPI wheel is CUDA-only)
cd ../LMCache
BUILD_WITH_HIP=1 ROCM_PATH=/opt/rocm pip install -e . --no-build-isolation

# (c) FlyDSL — TQ44v4 only (skip for BF16 / fp8g32)
#     prebuilt at /root/FlyDSL/build-fly/python_packages  (commit 41500b0)

# verify
python3 -c "import lmcache, vllm; print('lmcache + vllm OK')"
```

---

## 3. The three cache configs (A/B/C)

Each format is run in three configs; the comparison between them *is* the experiment:

| Config | What's cached | Server flags |
|---|---|---|
| **A — vanilla** | nothing (prefill every time) | `--no-enable-prefix-caching` |
| **B — HBM prefix cache** | KV in HBM (LRU) | `--enable-prefix-caching` |
| **C — LMCache DRAM** | HBM L1 + CPU-DRAM L2 | `--enable-prefix-caching` + LMCache connector |

The serve script `examples/ultralmcache_mi355/00_serve_minimax.sh` applies the
**mandatory hash fix** (`PYTHONHASHSEED=0` + `sha256`) automatically — without it you
get phantom 0% cache hits at TP>1. Select the config with `KV_CONFIG=A|B|C`.

> **The scripts are vendored in this repo** under
> [`examples/ultralmcache_mi355/`](./examples/ultralmcache_mi355/): the serve script,
> the cold/warm sanity check, the three env overlays (`common.env`,
> `common_tq44.env`, `common_fp8g32.env`), and the `gpu_map.py` HIP↔card helper.
> They are self-contained — `logs/` and `results*/` are written **beside the scripts**
> (`BENCH_DIR` defaults to that folder), and the only host-specific paths
> (`MODEL_PATH`, `VLLM_FLYDSL_ROOT`) are overridable env vars.

---

## 4. Launch (config C shown — swap `KV_CONFIG` for A/B)

All commands run from `examples/ultralmcache_mi355/` inside the LMCache fork.
Pick two **free** GPUs for `GPUS` (`len == TP=2`) — `python3 gpu_map.py table` lists
free HIP indices (rocm-smi card index ≠ HIP index on MI300/MI355).

### 4.1 BF16 baseline

```bash
cd examples/ultralmcache_mi355
KV_CONFIG=C GPUS=0,1 ./00_serve_minimax.sh
source common.env && wait_for_server
```

### 4.2 TQ44v4 (4-bit TurboQuant)

```bash
cd examples/ultralmcache_mi355
COMMON_ENV=common_tq44.env KV_CONFIG=C \
  GPU_MEM_UTIL="$(source common_tq44.env >/dev/null; echo $GPU_MEM_UTIL_KVMATCH)" \
  GPUS=0,1 ./00_serve_minimax.sh
source common_tq44.env && wait_for_server
```

Banner check: `kv kernel: dtype=turboquant_4bit_nc ...` and
`tq flydsl: V4=1 SOA_STORE=1 BUTTERFLY=0 AITER=0`.

### 4.3 fp8g32 (UltraQuant FP4-g32)

```bash
cd examples/ultralmcache_mi355
COMMON_ENV=common_fp8g32.env KV_CONFIG=C \
  GPU_MEM_UTIL="$(source common_fp8g32.env >/dev/null; echo $GPU_MEM_UTIL_KVMATCH)" \
  GPUS=0,1 ./00_serve_minimax.sh
source common_fp8g32.env && wait_for_server
```

Banner check: `kv kernel: dtype=fp8_kv_g32 block=32 backend=ROCM_AITER_UNIFIED_ATTN`.

> The only thing that changes between formats is the `COMMON_ENV` overlay (and the
> token-matched `gmu`/L2 it computes). Everything else — model, client, hash fix — is
> identical, which is what makes the three runs comparable.

---

## 5. Sanity check (always run first)

```bash
cd examples/ultralmcache_mi355
./01_sanity_cold_warm.sh
grep -iE 'LMCache hit tokens|Stored .* tokens' "$LOG_DIR"/server_minimax_*.log | tail
```

Expect: **1st pass** `hit tokens: 0` + `Stored N of N tokens`; **2nd pass** `hit tokens: N`.
If the 2nd pass is still `0`, the hash settings are wrong — fix before measuring.

---

## 6. Where results go

- **Authoritative cache signal = the server log** (`$LOG_DIR/server_minimax_*.log`):
  cold → `LMCache hit tokens: 0` + `Stored ...`; warm → `LMCache hit tokens: N`.
- Per-run tables/CSVs are namespaced so formats never overwrite each other:
  `results/` (BF16), `results_tq44/` (TQ44v4), `results_fp8g32/` (fp8g32).

---

## 7. Quick gotchas

- Keep `--enable-prefix-caching` **on** even with LMCache (LMCache reuses vLLM's hash fn).
- `PYTHONHASHSEED=0` + `sha256` are **mandatory** at TP>1 (handled by the serve script).
- **TQ44v4 only:** `VLLM_ROCM_USE_AITER=0`, butterfly flags OFF, `VLLM_TQ_SOA_FUSION_STORE=1`.
- **fp8g32 only:** `VLLM_FP8_G32_V3=1` is the enable flag (vLLM may log it as "unknown" — benign).
- Use **token-matched** `gmu`/L2 for the 4-bit formats (the env files compute these) — byte-matching never fills the tiers and makes B and C tie.
- BF16 is reproducible on stock upstream vLLM; **TQ44v4/fp8g32 need the vLLM fork above** (and FlyDSL for TQ44v4).

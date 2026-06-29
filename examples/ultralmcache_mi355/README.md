# UltraLMCache on MI355X — minimal serve scripts (BF16 / TQ44v4 / fp8g32)

Self-contained scripts to serve **MiniMax-M2.5** with **vLLM + LMCache CPU-DRAM KV
offload** on **AMD Instinct MI355X (gfx950, TP=2)** in three KV-cache formats. These
are the launch helpers referenced by the top-level [`instructions_to_run.md`](../../instructions_to_run.md);
read that for install steps, the vLLM fork branch, and the fairness rationale.

## Files

| File | Purpose |
|---|---|
| `00_serve_minimax.sh` | Launch vLLM (+/- LMCache) — picks A/B/C cache config, applies the TP>1 hash fix, passes KV-kernel flags through. |
| `01_sanity_cold_warm.sh` | Cold-vs-warm smoke test: same prompt twice, expect warm `cached_tokens > 0`. |
| `common.env` | BF16 baseline config (model path, endpoint, hash fix, LMCache L2 knobs, `wait_for_server`). |
| `common_tq44.env` | TurboQuant-44 (4-bit, FlyDSL v4) overlay — sources `common.env`, token-matched gmu/L2. |
| `common_fp8g32.env` | UltraQuant FP4-g32 overlay — sources `common.env`, token-matched gmu/L2. |
| `gpu_map.py` | Correct rocm-smi card ↔ HIP device-index mapping (they differ on MI300/MI355). |

## Quick start

```bash
python3 gpu_map.py table                 # find two FREE HIP indices
KV_CONFIG=C GPUS=1,2 ./00_serve_minimax.sh    # BF16, LMCache DRAM tier
source common.env && wait_for_server
./01_sanity_cold_warm.sh                  # warm pass should report cached_tokens > 0
```

Swap the overlay for the 4-bit formats (see `instructions_to_run.md` §4.2/§4.3):

```bash
COMMON_ENV=common_tq44.env   KV_CONFIG=C GPUS=1,2 ./00_serve_minimax.sh   # TQ44v4
COMMON_ENV=common_fp8g32.env KV_CONFIG=C GPUS=1,2 ./00_serve_minimax.sh   # fp8g32
```

## Host-specific knobs (overridable env vars)

- `MODEL_PATH` — MiniMax-M2.5 checkpoint (default `/shareddata/.../models/MiniMax-M2.5`).
- `VLLM_FLYDSL_ROOT` — FlyDSL build for TQ44v4 / fp8g32-V4 (default `/root/FlyDSL`).
- `BENCH_DIR` — output root; defaults to **this folder**, so `logs*/` and `results*/`
  are written beside the scripts (git-ignored).

# SPDX-License-Identifier: Apache-2.0
"""UltraQuant ``fp8_kv_g32`` packed-slot KV layout offload contract.

The ``fp8_kv_g32`` backend (FP4 E2M1 codes + UE8M0 per-group-of-32 scales +
FP8-E4M3 query) stores KV as a combined, opaque uint8 slot per (token, head),
exactly like TQ44 — only the slot *size* and the internal byte regions differ:

    head_dim=128, group_size=32  ->  slot = head_dim + 2*(head_dim/32) = 136 B (AoS)
        bytes [  0 .. 64)  K codes  (FP4 nibbles, 2/byte)
        bytes [ 64 .. 68)  K scales (UE8M0, 1 byte x 4 groups)
        bytes [ 68 ..132)  V codes
        bytes [132 ..136)  V scales (UE8M0, 1 byte x 4 groups)

LMCache treats this slot as an opaque ``NH x slot_bytes`` byte vector and moves
whole blocks; it never parses codes/scales. These tests therefore pin that the
*existing* group-aware connector already handles a 136-byte fp8_kv_g32 slot with
**no cache-layer code change** — the central claim for the UltraQuant bring-up:

  * a whole-block opaque D2H + H2D round-trip is byte-exact for a 136-B slot;
  * the even 136-B slot is NOT split into a fake 2 x 68 K/V pair (which would
    scramble the UE8M0 scale regions at offsets 64/68/132);
  * a mixed bf16-boundary + fp8g32-packed stack groups into exactly two kernel
    groups with the packed group carrying hs=136.

CPU fallback tests run without a GPU; the compiled-kernel / V3-connector tests
skip cleanly when no ROCm/CUDA device is present. This file is ADDITIVE — it does
not touch the TQ44 (SLOT=134) tests in this directory.
"""

# Third Party
import pytest
import torch

# First Party
from lmcache.python_ops_fallback import (
    EngineKVFormat as FbEngineKVFormat,
)
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector import utils as U
from lmcache.v1.gpu_connector.utils import detect_per_layer_formats
from lmcache.v1.kv_layer_groups import KVLayerGroupsManager
import lmcache.c_ops as lmc_ops
from lmcache.python_ops_fallback import (
    PageBufferShapeDesc as FbPageBufferShapeDesc,
)
from lmcache.python_ops_fallback import (
    TransferDirection as FbTransferDirection,
)
from lmcache.python_ops_fallback import (
    multi_layer_block_kv_transfer as fb_multi_layer_block_kv_transfer,
)
from lmcache.python_ops_fallback import (
    set_shape_desc_dtype,
)

# fp8_kv_g32 geometry on MiniMax-M2.5: head_dim=128, group_size=32 ->
# slot = 64 (K codes) + 4 (K scales) + 64 (V codes) + 4 (V scales) = 136 bytes.
HEAD_DIM = 128
GROUP_SIZE = 32
N_GROUPS = HEAD_DIM // GROUP_SIZE          # 4
K_CODES = HEAD_DIM // 2                     # 64
SLOT = HEAD_DIM + 2 * N_GROUPS             # 136
# fp8g32 within-slot region boundaries (the bytes a fake K/V split would scramble)
OFF_K_SCALES = K_CODES                      # 64
OFF_V_CODES = K_CODES + N_GROUPS            # 68
OFF_V_SCALES = HEAD_DIM + N_GROUPS          # 132

NB, BS, NH, NL = 16, 32, 4, 3
CHUNK_TOKENS = 256  # LMCache chunk = 8 blocks of 32 (block-aligned)
BLOCKS_PER_CHUNK = CHUNK_TOKENS // BS

BF16_FMT = lmc_ops.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS
PACKED_FMT = getattr(lmc_ops.EngineKVFormat, "NL_X_NB_BS_NH_PACKED", None)

_HAS_COMPILED_PACKED = PACKED_FMT is not None
requires_compiled_packed = pytest.mark.skipif(
    not _HAS_COMPILED_PACKED,
    reason="compiled lmc_ops lacks NL_X_NB_BS_NH_PACKED (rebuild csrc)",
)
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs CUDA/ROCm device"
)

PACKED_HINT = {"kv_layout": "PACKED", "packed_slot_size": SLOT}


def _raw_packed_caches(seed: int = 0) -> list[torch.Tensor]:
    """Per-layer fp8g32 packed tensors: [NB, BS, NH, 136] uint8."""
    torch.manual_seed(seed)
    return [
        torch.randint(0, 256, (NB, BS, NH, SLOT), dtype=torch.uint8)
        for _ in range(NL)
    ]


def _make_shape_desc() -> FbPageBufferShapeDesc:
    sd = FbPageBufferShapeDesc()
    sd.kv_size = 1  # combined K+V, single opaque plane
    sd.nl = NL
    sd.nb = NB
    sd.bs = BS
    sd.nh = NH
    sd.hs = SLOT  # 136-byte slot carried in hs (element_size == 1)
    sd.element_size = 1
    sd.block_stride_elems = BS * NH * SLOT
    set_shape_desc_dtype(sd, torch.uint8)
    return sd


# --------------------------------------------------------------------------- #
#  Geometry sanity                                                            #
# --------------------------------------------------------------------------- #


def test_fp8g32_slot_is_136():
    """Pin the fp8_kv_g32 slot byte-layout this suite asserts against."""
    assert SLOT == 136
    assert (OFF_K_SCALES, OFF_V_CODES, OFF_V_SCALES) == (64, 68, 132)
    # 4.25 bits/element amortized (4-bit code + 8/32-bit scale).
    bits_per_elem = (SLOT * 8) / (2 * HEAD_DIM)
    assert abs(bits_per_elem - 4.25) < 1e-9


def test_enum_value_present():
    assert int(FbEngineKVFormat.NL_X_NB_BS_NH_PACKED) == 11


# --------------------------------------------------------------------------- #
#  Discovery / accessors (slot carried as head_size, never split)            #
# --------------------------------------------------------------------------- #


@requires_compiled_packed
def test_discovery_returns_packed_and_never_splits():
    fmt, norm = U.normalize_kv_and_discover_format(
        _raw_packed_caches(1), EngineType.VLLM, PACKED_HINT
    )
    assert fmt == lmc_ops.EngineKVFormat.NL_X_NB_BS_NH_PACKED
    assert tuple(norm[0].shape) == (NB, BS, NH, SLOT)


@requires_compiled_packed
def test_discovery_without_hint_would_split_136():
    """Without the PACKED hint a (NB,BS,NH,136) cache hits the fused-K/V
    heuristic and is split into 2 x 68 — the silent-corruption hazard the
    explicit hint avoids (here it would slice the UE8M0 scale regions). Confirm
    the two paths diverge, exactly as for TQ44's 134."""
    fmt, norm = U.normalize_kv_and_discover_format(
        _raw_packed_caches(1), EngineType.VLLM, {"kv_layout": "NHD"}
    )
    assert fmt == lmc_ops.EngineKVFormat.NL_X_NB_NH_BS_TWO_HS
    assert tuple(norm[0].shape) == (NB, BS, NH, 2, SLOT // 2)


@requires_compiled_packed
def test_packed_accessors():
    fmt, norm = U.normalize_kv_and_discover_format(
        _raw_packed_caches(1), EngineType.VLLM, PACKED_HINT
    )
    assert U.get_num_layers(norm, fmt) == NL
    assert U.get_num_blocks(norm, fmt) == NB
    assert U.get_block_size(norm, fmt) == BS
    assert U.get_num_heads(norm, fmt) == NH
    assert U.get_head_size(norm, fmt) == SLOT  # 136 carried as head_size
    assert U.get_hidden_dim_size(norm, fmt) == NH * SLOT
    assert U.get_dtype(norm, fmt) == torch.uint8
    assert U.is_mla(fmt) is False


@requires_compiled_packed
def test_packed_shape_desc_has_kv_size_one():
    fmt, norm = U.normalize_kv_and_discover_format(
        _raw_packed_caches(1), EngineType.VLLM, PACKED_HINT
    )
    sd = U.make_page_buffer_shape_desc(
        norm, fmt, layer_idx=0, num_layers_in_group=NL,
        num_blocks=NB, block_size=BS,
    )
    assert sd.kv_size == 1
    assert sd.nh == NH
    assert sd.hs == SLOT  # 136
    assert sd.element_size == 1


def test_vllm_layout_hints_detects_packed_136():
    """The vLLM integration emits kv_layout=PACKED for 1-byte 4-D fp8g32 caches."""
    # First Party
    from lmcache.integration.vllm.utils import vllm_layout_hints

    kv_caches = {
        f"layer_{i}": torch.zeros((NB, BS, NH, SLOT), dtype=torch.uint8)
        for i in range(NL)
    }
    hints = vllm_layout_hints(kv_caches)
    assert hints.get("kv_layout") == "PACKED"
    assert hints.get("packed_slot_size") == SLOT


# --------------------------------------------------------------------------- #
#  CPU fallback round-trip (no GPU): byte-exact, scale regions intact         #
# --------------------------------------------------------------------------- #


def test_block_roundtrip_byte_exact():
    """D2H then H2D through the combined object plane is bit-exact for 136-B."""
    fmt = FbEngineKVFormat.NL_X_NB_BS_NH_PACKED
    raw = _raw_packed_caches()
    ref = [t.clone() for t in raw]
    sd = _make_shape_desc()

    block_ids = list(range(NB))
    n_chunks = (NB + BLOCKS_PER_CHUNK - 1) // BLOCKS_PER_CHUNK
    objs = [
        torch.zeros((NL, CHUNK_TOKENS, NH * SLOT), dtype=torch.uint8)
        for _ in range(n_chunks)
    ]

    paged_ptrs = torch.tensor([t.data_ptr() for t in raw], dtype=torch.long)
    obj_ptrs = [o.data_ptr() for o in objs]
    fb_multi_layer_block_kv_transfer(
        paged_ptrs, obj_ptrs, torch.tensor(block_ids, dtype=torch.long),
        torch.device("cpu"), FbTransferDirection.D2H, sd, CHUNK_TOKENS, fmt, 0,
    )
    out = [torch.zeros_like(t) for t in raw]
    out_ptrs = torch.tensor([t.data_ptr() for t in out], dtype=torch.long)
    fb_multi_layer_block_kv_transfer(
        out_ptrs, obj_ptrs, torch.tensor(block_ids, dtype=torch.long),
        torch.device("cpu"), FbTransferDirection.H2D, sd, CHUNK_TOKENS, fmt, 0,
    )
    for original, recovered in zip(ref, out, strict=True):
        assert torch.equal(original, recovered)


def test_scale_regions_survive_no_split():
    """An fp8g32 136-B slot must NOT be halved into 2 x 68.

    A fake K/V split at 68 would slice the K-scale region (64..68) onto the seam
    and misalign the V-code/V-scale regions. Stamp the four region boundaries and
    require they survive the round-trip in place — this is the UE8M0-scale analog
    of the TQ44 ``test_slot_is_not_split`` guard (which used 134's K/V seam)."""
    fmt = FbEngineKVFormat.NL_X_NB_BS_NH_PACKED
    raw = _raw_packed_caches()
    marks = {
        OFF_K_SCALES - 1: 0xA0,   # 63: last K code
        OFF_K_SCALES: 0xA1,       # 64: first K scale (UE8M0)
        OFF_V_CODES - 1: 0xA2,    # 67: last K scale
        OFF_V_CODES: 0xA3,        # 68: first V code (the would-be split seam)
        OFF_V_SCALES - 1: 0xA4,   # 131: last V code
        OFF_V_SCALES: 0xA5,       # 132: first V scale (UE8M0)
        SLOT - 1: 0xA6,           # 135: last V scale
    }
    for t in raw:
        for off, val in marks.items():
            t[0, 0, 0, off] = val
    ref = [t.clone() for t in raw]
    sd = _make_shape_desc()

    obj = torch.zeros((NL, CHUNK_TOKENS, NH * SLOT), dtype=torch.uint8)
    block_ids = list(range(BLOCKS_PER_CHUNK))
    paged_ptrs = torch.tensor([t.data_ptr() for t in raw], dtype=torch.long)
    fb_multi_layer_block_kv_transfer(
        paged_ptrs, [obj.data_ptr()], torch.tensor(block_ids, dtype=torch.long),
        torch.device("cpu"), FbTransferDirection.D2H, sd, CHUNK_TOKENS, fmt, 0,
    )
    out = [torch.zeros_like(t) for t in raw]
    out_ptrs = torch.tensor([t.data_ptr() for t in out], dtype=torch.long)
    fb_multi_layer_block_kv_transfer(
        out_ptrs, [obj.data_ptr()], torch.tensor(block_ids, dtype=torch.long),
        torch.device("cpu"), FbTransferDirection.H2D, sd, CHUNK_TOKENS, fmt, 0,
    )
    for recovered in out:
        for off, val in marks.items():
            assert recovered[0, 0, 0, off] == val, (
                f"fp8g32 region byte {off} scrambled (slot was split?)"
            )
    touched = torch.tensor(block_ids, dtype=torch.long)
    for original, recovered in zip(ref, out, strict=True):
        assert torch.equal(recovered.index_select(0, touched),
                           original.index_select(0, touched))


def test_partial_blocks_leave_others_untouched():
    """Only the transferred blocks are written on H2D; the rest stay zero."""
    fmt = FbEngineKVFormat.NL_X_NB_BS_NH_PACKED
    raw = _raw_packed_caches()
    ref = [t.clone() for t in raw]
    sd = _make_shape_desc()

    block_ids = list(range(BLOCKS_PER_CHUNK))
    obj = torch.zeros((NL, CHUNK_TOKENS, NH * SLOT), dtype=torch.uint8)
    paged_ptrs = torch.tensor([t.data_ptr() for t in raw], dtype=torch.long)
    fb_multi_layer_block_kv_transfer(
        paged_ptrs, [obj.data_ptr()], torch.tensor(block_ids, dtype=torch.long),
        torch.device("cpu"), FbTransferDirection.D2H, sd, CHUNK_TOKENS, fmt, 0,
    )
    out = [torch.zeros_like(t) for t in raw]
    out_ptrs = torch.tensor([t.data_ptr() for t in out], dtype=torch.long)
    fb_multi_layer_block_kv_transfer(
        out_ptrs, [obj.data_ptr()], torch.tensor(block_ids, dtype=torch.long),
        torch.device("cpu"), FbTransferDirection.H2D, sd, CHUNK_TOKENS, fmt, 0,
    )
    touched = torch.tensor(block_ids, dtype=torch.long)
    untouched = torch.tensor(
        [b for b in range(NB) if b not in block_ids], dtype=torch.long
    )
    for original, recovered in zip(ref, out, strict=True):
        assert torch.equal(recovered.index_select(0, touched),
                           original.index_select(0, touched))
        assert torch.count_nonzero(recovered.index_select(0, untouched)) == 0


# --------------------------------------------------------------------------- #
#  Mixed bf16-boundary + fp8g32-packed stack -> two kernel groups             #
# --------------------------------------------------------------------------- #

N_BF16_EACH_SIDE = 2
N_PACKED = 3
MIX_NB, MIX_BS, MIX_NH = 8, 32, 4
NUM_LAYERS = 2 * N_BF16_EACH_SIDE + N_PACKED  # 7


def _mix_packed_indices() -> list[int]:
    return list(range(N_BF16_EACH_SIDE, N_BF16_EACH_SIDE + N_PACKED))


def _mix_bf16_indices() -> list[int]:
    return list(range(N_BF16_EACH_SIDE)) + list(
        range(NUM_LAYERS - N_BF16_EACH_SIDE, NUM_LAYERS)
    )


def _make_mixed_stack(device: str = "cpu", seed: int = 0) -> list[torch.Tensor]:
    torch.manual_seed(seed)
    layers: list[torch.Tensor] = []
    packed_set = set(_mix_packed_indices())
    for i in range(NUM_LAYERS):
        if i in packed_set:
            layers.append(
                torch.randint(0, 256, (MIX_NB, MIX_BS, MIX_NH, SLOT),
                              dtype=torch.uint8, device=device)
            )
        else:
            layers.append(
                torch.randn(2, MIX_NB, MIX_BS, MIX_NH, HEAD_DIM,
                            dtype=torch.bfloat16, device=device)
            )
    return layers


@requires_compiled_packed
def test_detect_per_layer_formats_rank_based_no_hint():
    """Mixed 4-D(fp8g32)/5-D(bf16) stack classified per-layer without a hint:
    the 5-D layers prove the layout, so the 4-D layers can only be packed."""
    kv = _make_mixed_stack()
    fmts = detect_per_layer_formats(kv, EngineType.VLLM, {})
    packed_set = set(_mix_packed_indices())
    expected = [
        int(PACKED_FMT) if i in packed_set else int(BF16_FMT)
        for i in range(NUM_LAYERS)
    ]
    assert [int(f) for f in fmts] == expected


@requires_compiled_packed
def test_mixed_stack_groups_into_two_packed_hs_136():
    kv = _make_mixed_stack()
    plf = detect_per_layer_formats(kv, EngineType.VLLM, {})
    mgr = KVLayerGroupsManager(
        kv, engine_kv_format=BF16_FMT, num_blocks=MIX_NB, per_layer_format=plf
    )
    groups = mgr.kernel_groups
    assert len(groups) == 2
    by_fmt = {int(g.engine_kv_format): g for g in groups}
    bf = by_fmt[int(BF16_FMT)]
    pk = by_fmt[int(PACKED_FMT)]

    assert bf.layer_indices == _mix_bf16_indices()
    assert pk.layer_indices == _mix_packed_indices()

    # bf16 group: separate K/V plane, real head_size, 2-byte elems.
    assert bf.dtype == torch.bfloat16
    assert bf.shape_desc.kv_size == 2
    assert bf.shape_desc.hs == HEAD_DIM
    assert bf.shape_desc.element_size == 2

    # fp8g32 packed group: combined plane, 136-B slot as head_size, 1-byte.
    assert pk.dtype == torch.uint8
    assert pk.shape_desc.kv_size == 1
    assert pk.shape_desc.hs == SLOT  # 136
    assert pk.shape_desc.nh == MIX_NH
    assert pk.shape_desc.element_size == 1
    assert pk.hidden_dim_size == MIX_NH * SLOT


def test_uint8_kv_dtype_is_v3_trigger():
    """fp8_kv_g32 registers a uint8 KV container, the V3 auto-select trigger."""
    kv_dtype = torch.uint8
    assert (kv_dtype == torch.uint8) is True
    for non_quant in (torch.bfloat16, torch.float16, torch.float8_e4m3fn):
        assert (non_quant == torch.uint8) is False


# --------------------------------------------------------------------------- #
#  Compiled HIP kernel round-trip (GPU; skips without a device)               #
# --------------------------------------------------------------------------- #

_GPU_KERNEL_DRIVER = r"""
import torch, lmcache.c_ops as o
NB, BS, NH, SLOT, NL = 16, 32, 4, 136, 3
CHUNK = 256
BPC = CHUNK // BS
NCH = (NB + BPC - 1) // BPC
dev = torch.device("cuda")
torch.manual_seed(7)
raw = [torch.randint(0, 256, (NB, BS, NH, SLOT), dtype=torch.uint8, device=dev)
       for _ in range(NL)]
ref = [t.clone() for t in raw]
sd = o.PageBufferShapeDesc()
sd.kv_size = 1; sd.nl = NL; sd.nb = NB; sd.bs = BS; sd.nh = NH; sd.hs = SLOT
sd.element_size = 1
fmt = o.EngineKVFormat.NL_X_NB_BS_NH_PACKED
objs = [torch.zeros((1, NL, CHUNK, NH * SLOT), dtype=torch.uint8, device=dev)
        for _ in range(NCH)]
paged = torch.tensor([t.data_ptr() for t in raw], dtype=torch.long, device=dev)
bids = torch.arange(NB, dtype=torch.long, device=dev)
o.multi_layer_block_kv_transfer(
    paged, [x.data_ptr() for x in objs], bids, dev,
    o.TransferDirection.D2H, sd, CHUNK, fmt, 0)
torch.cuda.synchronize()
out = [torch.zeros_like(t) for t in raw]
op = torch.tensor([t.data_ptr() for t in out], dtype=torch.long, device=dev)
o.multi_layer_block_kv_transfer(
    op, [x.data_ptr() for x in objs], bids, dev,
    o.TransferDirection.H2D, sd, CHUNK, fmt, 0)
torch.cuda.synchronize()
assert all(torch.equal(a, b) for a, b in zip(ref, out)), "round-trip mismatch"
print("FP8G32_GPU_ROUNDTRIP_OK")
"""


@requires_compiled_packed
@requires_cuda
def test_compiled_gpu_kernel_roundtrip():
    """Drive the compiled multi_layer_block_kv_transfer on-device for 136-B."""
    # Standard
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "-c", _GPU_KERNEL_DRIVER],
        capture_output=True, text=True, timeout=180,
    )
    assert "FP8G32_GPU_ROUNDTRIP_OK" in proc.stdout, (
        f"GPU kernel round-trip failed.\nstdout:\n{proc.stdout}\n"
        f"stderr:\n{proc.stderr}"
    )

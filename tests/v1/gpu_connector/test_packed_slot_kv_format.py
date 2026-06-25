# SPDX-License-Identifier: Apache-2.0
"""Combined-KV packed-slot KV layout (EngineKVFormat.NL_X_NB_BS_NH_PACKED).

Quantized KV backends (TurboQuant TQ44 FlyDSL v4, FP4-g32, FP8-g32) pack BOTH
K and V codes plus their fp16 metadata (norm / scale / zero) into one opaque
byte slot per (token, head). The per-layer physical tensor is
``[NB, BS, NH, slot_size]`` of ``uint8``; ``slot_size`` is NOT ``2 * head_size``
and must never be split into a K/V pair.

These tests pin the pure-Python fallback transfer for that layout:
  * a whole-block opaque D2H + H2D round-trip is byte-exact, including the
    bytes that (under the SoA store) hold the per-block metadata tail;
  * blocks outside the transfer set are left untouched.

The fallback path is exercised directly (no GPU / no compiled extension needed),
mirroring tests/v1/gpu_connector/test_blocks_first_fused_kv_format.py.
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

# TQ44 (turboquant_4bit_nc) geometry on MiniMax-M2.5: head_dim=128 -> packed
# slot of 134 bytes (K 64 codes + 2 norm | V 64 codes + 4 scale/zero).
NB, BS, NH, SLOT, NL = 16, 32, 4, 134, 3
CHUNK_TOKENS = 256  # LMCache chunk = 8 blocks of 32 (block-aligned)
BLOCKS_PER_CHUNK = CHUNK_TOKENS // BS


def _raw_packed_caches() -> list[torch.Tensor]:
    """Per-layer packed tensors as registered: [NB, BS, NH, SLOT] uint8."""
    torch.manual_seed(0)
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
    sd.hs = SLOT  # packed slot size carried in hs (element_size == 1)
    sd.element_size = 1
    sd.block_stride_elems = BS * NH * SLOT
    set_shape_desc_dtype(sd, torch.uint8)
    return sd


PACKED_HINT = {"kv_layout": "PACKED", "packed_slot_size": SLOT}

_HAS_COMPILED_PACKED = hasattr(
    lmc_ops.EngineKVFormat, "NL_X_NB_BS_NH_PACKED"
)
requires_compiled_packed = pytest.mark.skipif(
    not _HAS_COMPILED_PACKED,
    reason="compiled lmc_ops lacks NL_X_NB_BS_NH_PACKED (rebuild csrc)",
)


def _packed_layers() -> list[torch.Tensor]:
    """Per-layer packed tensors [NB, BS, NH, SLOT] uint8 (CPU is fine)."""
    torch.manual_seed(1)
    return [
        torch.randint(0, 256, (NB, BS, NH, SLOT), dtype=torch.uint8)
        for _ in range(NL)
    ]


def test_enum_value_present():
    assert int(FbEngineKVFormat.NL_X_NB_BS_NH_PACKED) == 11


@requires_compiled_packed
def test_discovery_returns_packed_and_never_splits():
    fmt, norm = U.normalize_kv_and_discover_format(
        _packed_layers(), EngineType.VLLM, PACKED_HINT
    )
    assert fmt == lmc_ops.EngineKVFormat.NL_X_NB_BS_NH_PACKED
    # Discovery must keep the trailing dim intact (no reshape to [..., 2, 67]).
    assert tuple(norm[0].shape) == (NB, BS, NH, SLOT)


@requires_compiled_packed
def test_discovery_without_hint_would_split_134():
    """Without the PACKED hint, a 4-D (NB,BS,NH,134) cache hits the fused-K/V
    heuristic and is split — which is exactly the silent-corruption bug the
    explicit hint avoids. Confirm the two paths diverge."""
    fmt, norm = U.normalize_kv_and_discover_format(
        _packed_layers(), EngineType.VLLM, {"kv_layout": "NHD"}
    )
    # The legacy heuristic splits the even trailing dim into [..., 2, 67].
    assert fmt == lmc_ops.EngineKVFormat.NL_X_NB_NH_BS_TWO_HS
    assert tuple(norm[0].shape) == (NB, BS, NH, 2, SLOT // 2)


@requires_compiled_packed
def test_packed_accessors():
    fmt, norm = U.normalize_kv_and_discover_format(
        _packed_layers(), EngineType.VLLM, PACKED_HINT
    )
    assert U.get_num_layers(norm, fmt) == NL
    assert U.get_num_blocks(norm, fmt) == NB
    assert U.get_block_size(norm, fmt) == BS
    assert U.get_num_heads(norm, fmt) == NH
    assert U.get_head_size(norm, fmt) == SLOT  # slot carried as head_size
    assert U.get_hidden_dim_size(norm, fmt) == NH * SLOT
    assert U.get_page_buffer_size(norm, fmt) == NB * BS
    assert U.get_tokens_per_layer(norm, fmt) == NB * BS
    assert U.get_elements_per_layer(norm, fmt) == NB * BS * NH * SLOT
    assert U.get_dtype(norm, fmt) == torch.uint8
    # Combined-KV packed is non-MLA but kv_size == 1.
    assert U.is_mla(fmt) is False


def test_vllm_layout_hints_detects_packed():
    """The vLLM integration emits kv_layout=PACKED for 1-byte 4-D caches."""
    # First Party
    from lmcache.integration.vllm.utils import vllm_layout_hints

    kv_caches = {
        f"layer_{i}": torch.zeros((NB, BS, NH, SLOT), dtype=torch.uint8)
        for i in range(NL)
    }
    hints = vllm_layout_hints(kv_caches)
    assert hints.get("kv_layout") == "PACKED"
    assert hints.get("packed_slot_size") == SLOT


def test_vllm_layout_hints_ignores_unquantized():
    """A normal fp16 5-D KV cache must NOT be flagged as packed."""
    # First Party
    from lmcache.integration.vllm.utils import vllm_layout_hints

    # 2-byte 5-D flash-attention cache: not packed.
    kv_caches = {
        "layer_0": torch.zeros((2, NB, BS, NH, 128), dtype=torch.float16)
    }
    hints = vllm_layout_hints(kv_caches)
    assert hints.get("kv_layout") != "PACKED"
    assert "packed_slot_size" not in hints


@requires_compiled_packed
def test_packed_shape_desc_has_kv_size_one():
    fmt, norm = U.normalize_kv_and_discover_format(
        _packed_layers(), EngineType.VLLM, PACKED_HINT
    )
    sd = U.make_page_buffer_shape_desc(
        norm,
        fmt,
        layer_idx=0,
        num_layers_in_group=NL,
        num_blocks=NB,
        block_size=BS,
    )
    assert sd.kv_size == 1  # combined K+V plane
    assert sd.nh == NH  # real heads (not absorbed like MLA)
    assert sd.hs == SLOT  # opaque slot size
    assert sd.bs == BS
    assert sd.nb == NB
    assert sd.nl == NL
    assert sd.element_size == 1


def test_block_roundtrip_byte_exact():
    """D2H then H2D through the combined object plane must be bit-exact."""
    fmt = FbEngineKVFormat.NL_X_NB_BS_NH_PACKED
    raw = _raw_packed_caches()
    ref = [t.clone() for t in raw]
    sd = _make_shape_desc()

    # All NB blocks transferred -> chunks covering the full buffer.
    block_ids = list(range(NB))
    n_chunks = (NB + BLOCKS_PER_CHUNK - 1) // BLOCKS_PER_CHUNK

    # Combined object plane: [NL, chunk_tokens, NH * SLOT] uint8, one per chunk.
    objs = [
        torch.zeros((NL, CHUNK_TOKENS, NH * SLOT), dtype=torch.uint8)
        for _ in range(n_chunks)
    ]

    # --- D2H: paged -> objects ---
    paged_ptrs = torch.tensor([t.data_ptr() for t in raw], dtype=torch.long)
    obj_ptrs = [o.data_ptr() for o in objs]
    fb_multi_layer_block_kv_transfer(
        paged_ptrs,
        obj_ptrs,
        torch.tensor(block_ids, dtype=torch.long),
        torch.device("cpu"),
        FbTransferDirection.D2H,
        sd,
        CHUNK_TOKENS,
        fmt,
        0,
    )

    # --- H2D: objects -> fresh paged buffers ---
    out = [torch.zeros_like(t) for t in raw]
    out_ptrs = torch.tensor([t.data_ptr() for t in out], dtype=torch.long)
    fb_multi_layer_block_kv_transfer(
        out_ptrs,
        obj_ptrs,
        torch.tensor(block_ids, dtype=torch.long),
        torch.device("cpu"),
        FbTransferDirection.H2D,
        sd,
        CHUNK_TOKENS,
        fmt,
        0,
    )

    for original, recovered in zip(ref, out, strict=True):
        assert torch.equal(original, recovered)


# Standalone driver for the compiled HIP kernel round-trip. Run in a
# subprocess (see test below) so any ROCm coredump-handler abort at process
# teardown cannot take down the pytest process. The kernel itself completes
# and validates byte-exactness before printing the sentinel.
_GPU_KERNEL_DRIVER = r"""
import torch, lmcache.c_ops as o
NB, BS, NH, SLOT, NL = 16, 32, 4, 134, 3
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
print("PACKED_GPU_ROUNDTRIP_OK")
"""


@requires_compiled_packed
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="GPU kernel round-trip requires a CUDA/ROCm device",
)
def test_compiled_gpu_kernel_roundtrip():
    """Drive the COMPILED lmc_ops.multi_layer_block_kv_transfer on-device.

    Exercises the newly-built HIP kernel path for the packed format
    (page_buffer_offset MLA-style branch + k_or_v_size == 1 grid), not the
    pure-Python fallback. Runs in a subprocess to isolate ROCm teardown.
    """
    # Standard
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "-c", _GPU_KERNEL_DRIVER],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert "PACKED_GPU_ROUNDTRIP_OK" in proc.stdout, (
        f"GPU kernel round-trip failed.\nstdout:\n{proc.stdout}\n"
        f"stderr:\n{proc.stderr}"
    )


def test_partial_blocks_leave_others_untouched():
    """Only the transferred blocks are written on H2D; the rest stay zero."""
    fmt = FbEngineKVFormat.NL_X_NB_BS_NH_PACKED
    raw = _raw_packed_caches()
    ref = [t.clone() for t in raw]
    sd = _make_shape_desc()

    # Transfer exactly one chunk's worth of blocks (first BLOCKS_PER_CHUNK).
    block_ids = list(range(BLOCKS_PER_CHUNK))
    obj = torch.zeros((NL, CHUNK_TOKENS, NH * SLOT), dtype=torch.uint8)

    paged_ptrs = torch.tensor([t.data_ptr() for t in raw], dtype=torch.long)
    fb_multi_layer_block_kv_transfer(
        paged_ptrs,
        [obj.data_ptr()],
        torch.tensor(block_ids, dtype=torch.long),
        torch.device("cpu"),
        FbTransferDirection.D2H,
        sd,
        CHUNK_TOKENS,
        fmt,
        0,
    )

    out = [torch.zeros_like(t) for t in raw]
    out_ptrs = torch.tensor([t.data_ptr() for t in out], dtype=torch.long)
    fb_multi_layer_block_kv_transfer(
        out_ptrs,
        [obj.data_ptr()],
        torch.tensor(block_ids, dtype=torch.long),
        torch.device("cpu"),
        FbTransferDirection.H2D,
        sd,
        CHUNK_TOKENS,
        fmt,
        0,
    )

    touched = torch.tensor(block_ids, dtype=torch.long)
    untouched = torch.tensor(
        [b for b in range(NB) if b not in block_ids], dtype=torch.long
    )
    for original, recovered in zip(ref, out, strict=True):
        assert torch.equal(recovered.index_select(0, touched),
                           original.index_select(0, touched))
        # Untouched blocks must still be zero (never written).
        assert torch.count_nonzero(recovered.index_select(0, untouched)) == 0


def test_slot_is_not_split():
    """A packed slot whose size is even (e.g. 134) must NOT be halved.

    Regression guard: the legacy fused-KV path would reshape a 4-D trailing
    dim into [..., 2, slot // 2]. For 134 that silently yields 2 x 67, slicing
    K codes / V codes / fp16 scales across a fake K/V seam. The packed format
    must keep the slot intact, so a single byte changed at the K/V boundary
    survives the round-trip in place.
    """
    fmt = FbEngineKVFormat.NL_X_NB_BS_NH_PACKED
    raw = _raw_packed_caches()
    # Stamp a recognizable value straddling the real K/V boundary (byte 66 is
    # the first V byte; byte 65 is the last K-norm byte).
    for t in raw:
        t[0, 0, 0, 64] = 200  # last K code region
        t[0, 0, 0, 65] = 201  # K-norm fp16 low byte
        t[0, 0, 0, 66] = 202  # first V code
    ref = [t.clone() for t in raw]
    sd = _make_shape_desc()

    obj = torch.zeros((NL, CHUNK_TOKENS, NH * SLOT), dtype=torch.uint8)
    block_ids = list(range(BLOCKS_PER_CHUNK))

    paged_ptrs = torch.tensor([t.data_ptr() for t in raw], dtype=torch.long)
    fb_multi_layer_block_kv_transfer(
        paged_ptrs, [obj.data_ptr()],
        torch.tensor(block_ids, dtype=torch.long),
        torch.device("cpu"), FbTransferDirection.D2H, sd, CHUNK_TOKENS, fmt, 0,
    )
    out = [torch.zeros_like(t) for t in raw]
    out_ptrs = torch.tensor([t.data_ptr() for t in out], dtype=torch.long)
    fb_multi_layer_block_kv_transfer(
        out_ptrs, [obj.data_ptr()],
        torch.tensor(block_ids, dtype=torch.long),
        torch.device("cpu"), FbTransferDirection.H2D, sd, CHUNK_TOKENS, fmt, 0,
    )

    # Block 0 is in the transferred set; the stamped boundary bytes must
    # survive in place (no K/V split would have scrambled them).
    for recovered in out:
        assert recovered[0, 0, 0, 64] == 200
        assert recovered[0, 0, 0, 65] == 201
        assert recovered[0, 0, 0, 66] == 202
    # Compare only the transferred blocks (the rest of `out` stays zero).
    touched = torch.tensor(block_ids, dtype=torch.long)
    for original, recovered in zip(ref, out, strict=True):
        assert torch.equal(recovered.index_select(0, touched),
                           original.index_select(0, touched))

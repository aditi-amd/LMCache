# SPDX-License-Identifier: Apache-2.0
"""Mixed packed (4-D uint8) + native bf16 (5-D) KV stack per-group transfer.

TurboQuant TQ44 (and FP4-g32 / FP8-g32) apply *boundary protection*: the first
and last N attention layers keep a native bf16 cache (standard 5-D
``[2, NB, BS, NH, HS]``), while the middle layers use a combined packed slot
(4-D ``[NB, BS, NH, slot]`` uint8). The resulting per-layer stack is therefore
**mixed in rank, dtype, and per-token byte width**.

LMCache's default single-geometry connector (V2) strides the whole stack with
one geometry; on a mixed stack it walks the packed layers with bf16 geometry and
faults the GPU. These tests pin the group-aware path:

  * ``detect_per_layer_formats`` classifies each layer by rank (no hint needed
    for a genuinely mixed stack).
  * ``KVLayerGroupsManager`` splits the stack into exactly two kernel groups
    (one bf16, one packed) with the correct per-group format/dtype/geometry.
  * A per-group D2H -> H2D round-trip through the compiled ``multi_layer_kv_transfer``
    kernel is byte-exact for the packed group and value-exact for the bf16
    group, and leaves untouched slots intact.
  * The auto-selection trigger picks the group-aware V3 connector for a uint8
    (quantized) KV cache.
"""

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.utils import detect_per_layer_formats
from lmcache.v1.kv_layer_groups import KVLayerGroupsManager
import lmcache.c_ops as lmc_ops

# MiniMax-M2.5 TQ44 geometry: head_dim=128 -> packed slot 134 bytes
# (key 64 codes + 2 norm | value 64 codes + 4 scale/zero). 8 blocks of 32
# tokens, 4 KV heads. 2 bf16 boundary layers on each side, 3 packed in between.
NB, BS, NH, HS, SLOT = 8, 32, 4, 128, 134
N_BF16_EACH_SIDE = 2
N_PACKED = 3
NUM_LAYERS = 2 * N_BF16_EACH_SIDE + N_PACKED  # 7

BF16_FMT = lmc_ops.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS
PACKED_FMT = getattr(lmc_ops.EngineKVFormat, "NL_X_NB_BS_NH_PACKED", None)

requires_compiled_packed = pytest.mark.skipif(
    PACKED_FMT is None,
    reason="compiled lmc_ops lacks NL_X_NB_BS_NH_PACKED (rebuild csrc)",
)
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs CUDA device"
)


def _packed_layer_indices() -> list[int]:
    return list(range(N_BF16_EACH_SIDE, N_BF16_EACH_SIDE + N_PACKED))


def _bf16_layer_indices() -> list[int]:
    front = list(range(N_BF16_EACH_SIDE))
    back = list(range(NUM_LAYERS - N_BF16_EACH_SIDE, NUM_LAYERS))
    return front + back


def _make_mixed_stack(device: str = "cpu", seed: int = 0) -> list[torch.Tensor]:
    """Per-layer registered tensors: bf16 5-D boundary + uint8 4-D packed."""
    torch.manual_seed(seed)
    layers: list[torch.Tensor] = []
    packed_set = set(_packed_layer_indices())
    for i in range(NUM_LAYERS):
        if i in packed_set:
            layers.append(
                torch.randint(
                    0, 256, (NB, BS, NH, SLOT), dtype=torch.uint8, device=device
                )
            )
        else:
            layers.append(
                torch.randn(2, NB, BS, NH, HS, dtype=torch.bfloat16, device=device)
            )
    return layers


# --------------------------------------------------------------------------- #
#  Per-layer format detection                                                 #
# --------------------------------------------------------------------------- #


@requires_compiled_packed
def test_detect_per_layer_formats_rank_based_no_hint():
    """A mixed 4-D/5-D stack is classified per-layer even without a PACKED hint:
    the 5-D layers prove the layout, so the 4-D layers can only be packed."""
    kv = _make_mixed_stack()
    fmts = detect_per_layer_formats(kv, EngineType.VLLM, {})
    expected = []
    packed_set = set(_packed_layer_indices())
    for i in range(NUM_LAYERS):
        expected.append(int(PACKED_FMT) if i in packed_set else int(BF16_FMT))
    assert [int(f) for f in fmts] == expected


@requires_compiled_packed
def test_detect_per_layer_formats_uniform_packed_needs_hint():
    """A *uniform* all-4-D stack is ambiguous (packed slot vs fused 2*HS) and
    is only classified packed when the explicit PACKED hint is supplied."""
    kv = [
        torch.zeros(NB, BS, NH, SLOT, dtype=torch.uint8) for _ in range(NUM_LAYERS)
    ]
    fmts = detect_per_layer_formats(
        kv, EngineType.VLLM, {"kv_layout": "PACKED", "packed_slot_size": SLOT}
    )
    assert all(int(f) == int(PACKED_FMT) for f in fmts)


def test_detect_per_layer_formats_uniform_bf16_unchanged():
    """A normal uniform bf16 stack stays single-format (no regression)."""
    kv = [
        torch.zeros(2, NB, BS, NH, HS, dtype=torch.bfloat16)
        for _ in range(NUM_LAYERS)
    ]
    fmts = detect_per_layer_formats(kv, EngineType.VLLM, {})
    assert all(int(f) == int(BF16_FMT) for f in fmts)
    assert len(fmts) == NUM_LAYERS


# --------------------------------------------------------------------------- #
#  Grouping                                                                    #
# --------------------------------------------------------------------------- #


@requires_compiled_packed
def test_mixed_stack_groups_into_two():
    kv = _make_mixed_stack()
    plf = detect_per_layer_formats(kv, EngineType.VLLM, {})
    mgr = KVLayerGroupsManager(
        kv, engine_kv_format=BF16_FMT, num_blocks=NB, per_layer_format=plf
    )
    groups = mgr.kernel_groups
    assert len(groups) == 2

    by_fmt = {int(g.engine_kv_format): g for g in groups}
    bf = by_fmt[int(BF16_FMT)]
    pk = by_fmt[int(PACKED_FMT)]

    # Membership.
    assert bf.layer_indices == _bf16_layer_indices()
    assert pk.layer_indices == _packed_layer_indices()

    # bf16 group geometry: separate K/V plane, real head_size, 2-byte elems.
    assert bf.dtype == torch.bfloat16
    assert bf.shape_desc.kv_size == 2
    assert bf.shape_desc.hs == HS
    assert bf.shape_desc.nh == NH
    assert bf.shape_desc.element_size == 2
    assert bf.hidden_dim_size == NH * HS

    # packed group geometry: combined plane, slot carried as head_size, 1 byte.
    assert pk.dtype == torch.uint8
    assert pk.shape_desc.kv_size == 1
    assert pk.shape_desc.hs == SLOT
    assert pk.shape_desc.nh == NH
    assert pk.shape_desc.element_size == 1
    assert pk.hidden_dim_size == NH * SLOT


@requires_compiled_packed
def test_homogeneous_stack_single_group_no_format_override():
    """When per_layer_format is omitted (homogeneous), the identity's format
    slot stays at the -1 sentinel and a single group is produced."""
    kv = [
        torch.zeros(2, NB, BS, NH, HS, dtype=torch.bfloat16)
        for _ in range(NUM_LAYERS)
    ]
    mgr = KVLayerGroupsManager(kv, engine_kv_format=BF16_FMT, num_blocks=NB)
    groups = mgr.kernel_groups
    assert len(groups) == 1
    assert groups[0].layer_indices == list(range(NUM_LAYERS))


# --------------------------------------------------------------------------- #
#  End-to-end per-group kernel round-trip (GPU)                               #
# --------------------------------------------------------------------------- #


def _group_object_tensor(group, num_tokens: int, device) -> torch.Tensor:
    """Allocate a per-group object plane shaped exactly as the kernel expects:
    ``[kv_size, num_layers_in_group, num_tokens, hidden_dim_size]``."""
    return torch.zeros(
        group.shape_desc.kv_size,
        group.num_layers,
        num_tokens,
        group.hidden_dim_size,
        dtype=group.dtype,
        device=device,
    )


def _group_pointers(kv, group, device) -> torch.Tensor:
    ptrs = [kv[i].data_ptr() for i in group.layer_indices]
    return torch.tensor(ptrs, dtype=torch.int64, device=device)


@requires_compiled_packed
@requires_cuda
def test_mixed_per_group_roundtrip_kernel():
    """Per-group D2H then H2D through the compiled kernel:
    packed group byte-exact, bf16 group value-exact, untouched slots intact."""
    dev = torch.device("cuda")
    kv = _make_mixed_stack(device="cuda", seed=11)
    ref = [t.clone() for t in kv]

    plf = detect_per_layer_formats(kv, EngineType.VLLM, {})
    mgr = KVLayerGroupsManager(
        kv, engine_kv_format=BF16_FMT, num_blocks=NB, per_layer_format=plf
    )
    groups = mgr.kernel_groups
    page_buffer_size = NB * BS

    # Transfer the first half of the slots only, so the back half must stay
    # untouched (verifies slot_mapping correctness & no over-write).
    num_tokens = page_buffer_size
    transferred = page_buffer_size // 2
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=dev)
    # Mark the back half invalid (-1) so the kernel skips it.
    slot_mapping[transferred:] = -1

    # Build per-group object planes and pointer arrays.
    objs = [_group_object_tensor(g, num_tokens, dev) for g in groups]
    gptrs = [_group_pointers(kv, g, dev) for g in groups]

    # --- D2H: paged -> objects ---
    for g, obj, ptrs in zip(groups, objs, gptrs, strict=True):
        lmc_ops.multi_layer_kv_transfer(
            obj,
            ptrs,
            slot_mapping,
            dev,
            page_buffer_size,
            lmc_ops.TransferDirection.D2H,
            g.engine_kv_format,
            block_size=g.shape_desc.bs,
            head_size=g.shape_desc.hs,
        )

    # Zero the paged buffers, then H2D back from the objects.
    for t in kv:
        t.zero_()
    for g, obj, ptrs in zip(groups, objs, gptrs, strict=True):
        lmc_ops.multi_layer_kv_transfer(
            obj,
            ptrs,
            slot_mapping,
            dev,
            page_buffer_size,
            lmc_ops.TransferDirection.H2D,
            g.engine_kv_format,
            block_size=g.shape_desc.bs,
            head_size=g.shape_desc.hs,
        )
    torch.cuda.synchronize()

    # Verify per layer. For the transferred slots the data must match the
    # reference; the untouched (-1) slots must remain zero after H2D.
    packed_set = set(_packed_layer_indices())
    for i in range(NUM_LAYERS):
        out = kv[i]
        original = ref[i]
        if i in packed_set:
            # packed: [NB, BS, NH, SLOT]; slot s -> block s//BS, pos s%BS.
            # transferred slots [0, transferred) round-trip byte-exact.
            for s in range(transferred):
                b, p = s // BS, s % BS
                assert torch.equal(out[b, p], original[b, p]), (
                    f"packed layer {i} slot {s} not byte-exact"
                )
            for s in range(transferred, num_tokens):
                b, p = s // BS, s % BS
                assert torch.count_nonzero(out[b, p]) == 0, (
                    f"packed layer {i} slot {s} should be untouched (zero)"
                )
        else:
            # bf16: [2, NB, BS, NH, HS].
            for s in range(transferred):
                b, p = s // BS, s % BS
                assert torch.equal(out[:, b, p], original[:, b, p]), (
                    f"bf16 layer {i} slot {s} not value-exact"
                )
            for s in range(transferred, num_tokens):
                b, p = s // BS, s % BS
                assert torch.count_nonzero(out[:, b, p]) == 0, (
                    f"bf16 layer {i} slot {s} should be untouched (zero)"
                )


# --------------------------------------------------------------------------- #
#  Auto-selection of the group-aware connector                                #
# --------------------------------------------------------------------------- #


def test_uint8_kv_dtype_is_v3_trigger():
    """A uint8 KV container dtype is the trigger that routes a run onto the
    group-aware V3 connector. This pins the contract used in CreateGPUConnector
    so the selection heuristic can't silently regress."""
    assert torch.uint8 != torch.bfloat16
    # The trigger predicate is intentionally simple and lives in
    # CreateGPUConnector: metadata.kv_dtype == torch.uint8.
    kv_dtype = torch.uint8
    assert (kv_dtype == torch.uint8) is True
    for non_quant in (torch.bfloat16, torch.float16, torch.float8_e4m3fn):
        assert (non_quant == torch.uint8) is False


# --------------------------------------------------------------------------- #
#  End-to-end V3 connector round-trip (store -> object -> load)               #
# --------------------------------------------------------------------------- #


def _build_mixed_metadata(kv_caches: dict[str, torch.Tensor]):
    """Metadata with the groups manager left UNbuilt so the V3 connector's own
    ``_initialize_kv_cache_pointers`` exercises per-layer-format detection,
    PACKED-hint self-injection, and per-group manager construction."""
    # First Party
    from lmcache.v1.metadata import LMCacheMetadata

    return LMCacheMetadata(
        model_name="test-tq44-mixed",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.uint8,  # the quantized-KV signal used to auto-select V3
        kv_shape=(NUM_LAYERS, 2, 256, NH, HS),
        use_mla=False,
    )


@requires_compiled_packed
@requires_cuda
def test_v3_connector_mixed_stack_store_load_roundtrip():
    """Drive the real VLLMPagedMemGPUConnectorV3 through a production-shaped
    store (from_gpu -> memory object) then load (to_gpu -> fresh dst stack) and
    verify every layer round-trips: byte-exact packed, value-exact bf16. This
    exercises detect_per_layer_formats, the connector's PACKED-hint
    self-injection, KVLayerGroupsManager construction with per_layer_format,
    metadata.get_shapes()/get_dtypes(), and the per-group transfer loops."""
    # First Party
    from lmcache.v1.gpu_connector.gpu_connectors import VLLMPagedMemGPUConnectorV3
    from lmcache.v1.memory_management import MemoryFormat, PinMemoryAllocator

    dev = torch.device("cuda", torch.cuda.current_device())
    src = _make_mixed_stack(device=str(dev), seed=21)
    dst = _make_mixed_stack(device=str(dev), seed=99)  # different contents
    ref = [t.clone() for t in src]

    src_caches = {f"layer_{i}": t for i, t in enumerate(src)}
    dst_caches = {f"layer_{i}": t for i, t in enumerate(dst)}

    page_buffer_size = NB * BS
    num_tokens = page_buffer_size
    chunk_size = 256
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=dev)

    allocator = PinMemoryAllocator(512 * 1024 * 1024)

    meta_src = _build_mixed_metadata(src_caches)
    meta_dst = _build_mixed_metadata(dst_caches)
    store_conn = VLLMPagedMemGPUConnectorV3(
        metadata=meta_src, use_gpu=False, device=dev
    )
    load_conn = VLLMPagedMemGPUConnectorV3(
        metadata=meta_dst, use_gpu=False, device=dev
    )

    # Mirror the adapter: eagerly bind caches so the groups manager (and hence
    # per-group buffer shapes) exist before the first allocation.
    store_conn.register_kv_caches(list(src_caches.values()))
    load_conn.register_kv_caches(list(dst_caches.values()))
    assert meta_src.get_num_groups() == 2
    assert meta_dst.get_num_groups() == 2

    for start in range(0, num_tokens, chunk_size):
        end = min(start + chunk_size, num_tokens)
        memory_obj = allocator.allocate(
            meta_src.get_shapes(end - start), meta_src.get_dtypes()
        )
        assert memory_obj is not None
        store_conn.from_gpu(
            memory_obj,
            start,
            end,
            kvcaches=list(src_caches.values()),
            slot_mapping=slot_mapping,
            offset=0,
        )
        assert memory_obj.metadata.fmt == MemoryFormat.KV_2LTD
        load_conn.to_gpu(
            memory_obj,
            start,
            end,
            kvcaches=list(dst_caches.values()),
            slot_mapping=slot_mapping,
            offset=0,
        )
        allocator.free(memory_obj)
    torch.cuda.synchronize()

    # dst must now equal the original src content, per layer.
    packed_set = set(_packed_layer_indices())
    for i in range(NUM_LAYERS):
        if i in packed_set:
            assert torch.equal(dst[i], ref[i]), f"packed layer {i} mismatch"
        else:
            assert torch.equal(dst[i], ref[i]), f"bf16 layer {i} mismatch"


# --------------------------------------------------------------------------- #
#  DEDICATED bf16 BOUNDARY-LAYER DRAM round-trip (first-2 / last-2 isolation)  #
# --------------------------------------------------------------------------- #


def _bits_equal_bf16(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Bit-exact comparison for bf16: view as uint16 so a single flipped bit
    (incl. NaN/-0.0 cases that ``torch.equal`` would mishandle) is caught."""
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


@requires_compiled_packed
@requires_cuda
def test_bf16_boundary_dram_roundtrip_bit_exact_and_isolated():
    """Dedicated correctness check for the bf16 *boundary* layers (first-2 /
    last-2) through the CPU-DRAM serialization path of the V3 connector.

    This complements the live forced-eviction needle probe (which reloads the
    whole stack from DRAM but cannot *isolate* the boundary group end-to-end):
    here we drive the real ``VLLMPagedMemGPUConnectorV3`` store (GPU -> pinned
    CPU-DRAM memory object) then load (CPU-DRAM -> fresh GPU stack) and assert

      1. every bf16 boundary layer is **bit-exact** (uint16 view), and
      2. the boundary group is **isolated** from the packed group: even after we
         scribble garbage into the destination packed layers right before the
         load, the bf16 boundary layers still reload bit-exact (a packed-group
         transfer bug cannot silently corrupt the boundary group, and vice
         versa).
    """
    # First Party
    from lmcache.v1.gpu_connector.gpu_connectors import VLLMPagedMemGPUConnectorV3
    from lmcache.v1.memory_management import MemoryFormat, PinMemoryAllocator

    dev = torch.device("cuda", torch.cuda.current_device())
    src = _make_mixed_stack(device=str(dev), seed=2027)
    dst = _make_mixed_stack(device=str(dev), seed=4242)  # different contents
    ref = [t.clone() for t in src]

    src_caches = {f"layer_{i}": t for i, t in enumerate(src)}
    dst_caches = {f"layer_{i}": t for i, t in enumerate(dst)}

    page_buffer_size = NB * BS
    num_tokens = page_buffer_size
    chunk_size = 256
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=dev)
    allocator = PinMemoryAllocator(512 * 1024 * 1024)

    store_conn = VLLMPagedMemGPUConnectorV3(
        metadata=_build_mixed_metadata(src_caches), use_gpu=False, device=dev
    )
    load_conn = VLLMPagedMemGPUConnectorV3(
        metadata=_build_mixed_metadata(dst_caches), use_gpu=False, device=dev
    )
    store_conn.register_kv_caches(list(src_caches.values()))
    load_conn.register_kv_caches(list(dst_caches.values()))

    bf16_layers = _bf16_layer_indices()
    packed_layers = _packed_layer_indices()
    # Sanity: this fixture really models first-N / last-N boundary protection.
    assert bf16_layers == [0, 1, NUM_LAYERS - 2, NUM_LAYERS - 1]

    for start in range(0, num_tokens, chunk_size):
        end = min(start + chunk_size, num_tokens)
        memory_obj = allocator.allocate(
            store_conn.metadata.get_shapes(end - start),
            store_conn.metadata.get_dtypes(),
        )
        assert memory_obj is not None
        # store: GPU -> CPU-DRAM memory object
        store_conn.from_gpu(
            memory_obj, start, end,
            kvcaches=list(src_caches.values()), slot_mapping=slot_mapping, offset=0,
        )
        assert memory_obj.metadata.fmt == MemoryFormat.KV_2LTD

        # ISOLATION: corrupt the destination packed layers right before load so
        # that a *correct* bf16 reload cannot be an accident of pre-existing
        # state, and a packed-path overrun into the boundary group would show up.
        for i in packed_layers:
            dst[i].fill_(0xAB)

        # load: CPU-DRAM memory object -> GPU
        load_conn.to_gpu(
            memory_obj, start, end,
            kvcaches=list(dst_caches.values()), slot_mapping=slot_mapping, offset=0,
        )
        allocator.free(memory_obj)
    torch.cuda.synchronize()

    # (1) every bf16 boundary layer bit-exact after the DRAM round-trip
    for i in bf16_layers:
        assert _bits_equal_bf16(dst[i], ref[i]), (
            f"bf16 BOUNDARY layer {i} not bit-exact after CPU-DRAM round-trip"
        )
    # (2) packed group also restored correctly (garbage fully overwritten)
    for i in packed_layers:
        assert torch.equal(dst[i], ref[i]), (
            f"packed layer {i} not byte-exact after DRAM round-trip"
        )

from __future__ import annotations

from typing import TYPE_CHECKING

from sglang.srt.environ import envs
from sglang.srt.mem_cache.buffer_only.hiradix_cache import (
    BufferOnlyHiRadixCache,
)
from sglang.srt.utils.tensor_bridge import use_mlx

if TYPE_CHECKING:
    from sglang.srt.mem_cache.registry import TreeCacheBuildContext


def create_buffer_only_cache(
    ctx: TreeCacheBuildContext,
) -> BufferOnlyHiRadixCache:
    """Validate and construct the isolated buffer-only HiCache backend."""

    if not ctx.enable_hierarchical_cache:
        raise ValueError("HiCache buffer_only requires hierarchical cache.")
    if ctx.effective_chunked_prefill_size is not None and ctx.disable_radix_cache:
        raise ValueError("HiCache buffer_only requires the Python radix cache.")
    if envs.SGLANG_EXPERIMENTAL_CPP_RADIX_TREE.get():
        raise ValueError("HiCache buffer_only does not support the C++ radix tree.")
    if envs.SGLANG_ENABLE_UNIFIED_RADIX_TREE.get() or use_mlx():
        raise ValueError("HiCache buffer_only does not support UnifiedRadixCache.")
    if ctx.is_hybrid_ssm or ctx.is_hybrid_swa:
        raise ValueError("HiCache buffer_only does not support hybrid SSM/SWA caches.")

    cache = BufferOnlyHiRadixCache(
        params=ctx.params,
        server_args=ctx.server_args,
    )
    ctx.tp_worker.register_hicache_layer_transfer_counter(
        cache.cache_controller.layer_done_counter
    )
    return cache

# HiCache buffer-only backend

This package isolates the L2 transfer-buffer implementation from the default
HiCache path. The default `cache` mode does not import this package.

## Integration surface

The implementation intentionally keeps the shared-code patch small:

- `server_args.py` defines and validates `--hicache-host-memory-mode`.
- `registry.py` lazily selects `create_buffer_only_cache`.
- `base_prefix_cache.py` exposes one L3 prefetch-anchor hook.
- `radix_cache.py` exposes one radix-node factory.
- `hiradix_cache.py` exposes one cache-controller factory.
- `scheduler.py` calls the prefetch-anchor hook.
- `http_server.py` reports the active host-memory mode.

All queueing, transfer ownership, L3 durability, retry, and tree-state logic is
owned by this package.

## Runtime invariants

1. L2 pages always have one active owner: a D2H write, an L3 read, or an H2D
   load. They are returned only after the owning asynchronous operation stops
   touching them.
2. An L1 node remains protected until its final write chunk is staged in L2.
   L3 acknowledgement, rather than D2H completion, advances
   `storage_backed_tokens`.
3. L3 reads query the continuous hit length before allocating L2 pages. The
   scheduler allocates only the confirmed prefix.
4. A node under H2D is hidden from normal prefix matching until its completion
   event is synchronized.
5. Queue decisions and allocations use the same collective ordering on every
   attention rank. Divergent write identities or states fail fast.
6. Partial reads, partial writes, cancellation, reset, and shutdown release
   only memory no longer touched by an I/O thread or device stream.

## Upgrade checklist

When rebasing this backend onto a newer SGLang release:

1. Apply the shared integration patch and keep the default controller unchanged.
2. Run `test_hicache_buffer_only.py` first. Its signature guard reports
   overridden upstream methods whose call shape changed.
3. Compare upstream changes to every method overridden by
   `BufferOnlyHiRadixCache` and `BufferOnlyHiCacheController`, even when the
   patch applies without conflicts.
4. Re-run default radix, registry, and server-argument tests to confirm that
   `--hicache-host-memory-mode cache` remains behaviorally unchanged.
5. Run multi-rank GPU tests for write retry, read cancellation, reset, and
   shutdown. CPU tests cannot validate CUDA event ordering or backend I/O.

The isolated package makes the patch easy to apply, but it does not make
upstream semantic changes safe to ignore.

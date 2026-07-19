# 15. 模块与函数导航地图

逐函数附录当前覆盖 **1178 个 Python 函数、132 个类声明**，以及 hash、通用搬运、staged write-back 三份完整 C++/CUDA/ROCm 源文件。每个 Python 函数都列出完整代码、签名、用途、状态读写和直接调用；数据类还列出字段契约，帮助先看懂状态再进入方法。

机器可读覆盖清单：[coverage.json](appendix/coverage.json)。附录总索引：[appendix/README.md](appendix/README.md)。

## 15.1 覆盖表

| 附录 | 函数 | 类 | 模块职责 |
|---|---:|---:|---|
| [A01 Registry](appendix/A01-registry.md) | 6 | 1 | 缓存实现注册和选择。 |
| [A02 Builder](appendix/A02-kv-cache-builder.md) | 3 | 1 | device pools 与 tree cache 构建。 |
| [A02b Prefix contract](appendix/A02b-prefix-cache-contract.md) | 39 | 12 | 公共参数/结果、抽象接口和 HiCache hooks。 |
| [A02c Init params](appendix/A02c-cache-init-params.md) | 0 | 1 | 构建阶段传入缓存的完整参数对象。 |
| [A02d L1 pools](appendix/A02d-l1-memory-pools.md) | 192 | 18 | 请求行/Mamba/Hybrid 映射及全部 device KV pool。 |
| [A02e Allocator base](appendix/A02e-device-allocator-base.md) | 18 | 1 | L1 allocator 抽象契约、回退与指标。 |
| [A02f Token allocator](appendix/A02f-token-allocator.md) | 7 | 1 | `page_size=1` 的 token slot 分配。 |
| [A02g Paged allocator](appendix/A02g-paged-allocator.md) | 9 | 1 | page 分配、token slot 展开与释放。 |
| [A02h Request allocation](appendix/A02h-request-allocation-flow.md) | 22 | 0 | prefill/decode 分配、请求行写入和释放。 |
| [A02i HiSparse allocator](appendix/A02i-hisparse-allocator.md) | 43 | 2 | 完整/压缩/稀疏 KV 的复合分配。 |
| [A02j SWA allocator](appendix/A02j-swa-allocator.md) | 44 | 2 | Full/SWA 页映射、分配、释放和回滚。 |
| [A02k Mamba allocator](appendix/A02k-mamba-allocator.md) | 9 | 1 | recurrent state 槽位生命周期。 |
| [A03 Radix](appendix/A03-radix-cache.md) | 45 | 3 | key、node、match、insert、split、lock、L1 eviction。 |
| [A03b Eviction policy](appendix/A03b-eviction-policy.md) | 9 | 8 | LRU/LFU/FIFO/MRU/FILO/priority/SLRU。 |
| [A03c Cache events](appendix/A03c-cache-events.md) | 4 | 1 | GPU/CPU store/remove/clear 事件。 |
| [A03d Session Radix](appendix/A03d-session-radix.md) | 6 | 1 | session leaf 与引用释放。 |
| [A04 HiRadix](appendix/A04-hiradix-cache.md) | 67 | 1 | L2/L3、读写、回收、ACK、rank 同步。 |
| [A05 Controller](appendix/A05-cache-controller.md) | 59 | 7 | operation、stream/event、L3 threads/queues。 |
| [A06 Host base](appendix/A06-host-pool-base.md) | 23 | 1 | L2 sizing、allocator、抽象搬运接口。 |
| [A07 Host common](appendix/A07-host-pool-common.md) | 7 | 1 | host tensor allocator 适配。 |
| [A08 MHA host](appendix/A08-host-pool-mha.md) | 43 | 3 | MHA/K-only/非对称 K/V layout 与搬运。 |
| [A09 Specialized pools](appendix/A09-model-specific-host-pools.md) | 118 | 8 | MLA、Mamba、DSA、DeepSeek V4、多 pool。 |
| [A09b HiSparse host](appendix/A09b-hisparse-host.md) | 4 | 1 | 稀疏 pool 的 page/token 分配换算。 |
| [A10 Storage](appendix/A10-storage-contract.md) | 44 | 11 | L3 contract、V2 transfer、文件后端。 |
| [A11 Factory](appendix/A11-storage-factory.md) | 6 | 1 | 内置/动态 backend 延迟加载。 |
| [A12 JIT Python](appendix/A12-jit-python.md) | 11 | 0 | 搬运 kernel 编译与调用。 |
| [A12b Native hash](appendix/A12b-native-hash-loader.md) | 4 | 0 | C++ hash 扩展加载和输入规范化。 |
| [A12c Hash utils](appendix/A12c-hash-utils.md) | 5 | 0 | 淘汰策略选择、链式页 hash 与节点分裂。 |
| [A13 Unified](appendix/A13-unified-radix-cache.md) | 128 | 6 | 组件化树、LRU、HiCache 复合状态机。 |
| [A13b Component contract](appendix/A13b-tree-component.md) | 27 | 6 | 组件数据、匹配、锁、淘汰和 transfer 抽象。 |
| [A13c Full component](appendix/A13c-full-component.md) | 12 | 1 | Full KV 组件策略。 |
| [A13d SWA component](appendix/A13d-swa-component.md) | 27 | 1 | 滑动窗口组件及 host/device 生命周期。 |
| [A13e Mamba component](appendix/A13e-mamba-component.md) | 19 | 1 | recurrent/checkpoint 组件生命周期。 |
| [A14 Hybrid controller](appendix/A14-hybrid-controller.md) | 33 | 4 | 多 pool operation。 |
| [A15 Assembler](appendix/A15-hybrid-pool-assembler.md) | 34 | 8 | 模型策略和 pool 装配。 |
| [A16 PD decode](appendix/A16-decode-hicache.md) | 13 | 5 | decode 侧本地缓存恢复。 |
| [A17 Scheduler](appendix/A17-scheduler-integration.md) | 6 | 1 | 预取、入批、abort、管理 API 接入点。 |
| [A18 Policy](appendix/A18-schedule-policy.md) | 2 | 1 | token 预算和 load-back。 |
| [A19 Req/Batch](appendix/A19-request-batch-state.md) | 7 | 2 | 请求命中/分配/提交字段和 H2D consumer。 |
| [A19b Result commit](appendix/A19b-result-cache-commit.md) | 2 | 1 | prefill/decode 结果提交到 Tree 的实际调用点。 |
| [A19c Metrics](appendix/A19c-hicache-metrics.md) | 12 | 4 | 请求缓存来源及内置 Prometheus storage/Radix 指标。 |
| [A19d Cache reporting](appendix/A19d-cache-hit-reporting.md) | 6 | 2 | 请求级 device/host/storage 命中来源输出。 |
| [A20 Args](appendix/A20-server-args.md) | 3 | 1 | layout/backend 兼容性归一化。 |
| [A21 C++](appendix/A21-kernel-cpp.md) | 完整文件 | - | SHA-256 hash、L1/L2 transfer、staged write-back。 |

Scheduler、schedule policy、schedule batch、L1 memory pool 和 server args 是大型通用模块，附录只抽取与 HiCache 生命周期直接相关的函数；核心 HiCache 文件则覆盖全部函数，包括局部嵌套函数。

## 15.2 启动调用链

```text
ServerArgs._handle_hicache
├── _resolve_layout_io_compatibility
└── _resolve_storage_layout_compatibility

build_kv_cache
└── create_tree_cache
    └── default_radix_cache_factory
        ├── HiRadixCache.__init__
        │   ├── HostKVCache subclass.__init__
        │   └── HiCacheController.__init__
        │       └── attach_storage_backend
        │           ├── StorageBackendFactory.create_backend
        │           └── _start_storage_threads
        └── _create_unified_radix_cache
            └── UnifiedRadixCache.init_hicache
                └── attach_hybrid_pool_to_unified_cache
```

按此顺序阅读 A20 -> A02 -> A01 -> A04/A13 -> A05/A14 -> A06/A08/A09 -> A11。

## 15.3 Radix 算法函数组

### Key

```text
RadixKey.__init__
RadixKey.__len__ / __getitem__ / raw_token_ids
RadixKey.page_aligned
RadixKey.maybe_to_bigram_view
RadixKey._check_compatible
RadixKey.match
RadixKey.child_key
RadixKey.hash_page
```

### Node

```text
TreeNode.__init__
TreeNode.evicted / backuped
TreeNode.protect_host / release_host
TreeNode.get_last_hash_value / get_prefix_hash_values
```

### Match/insert/split

```text
RadixCache.match_prefix
└── _match_prefix_helper
    └── _split_node

RadixCache.insert
└── _insert_helper
    └── _split_node
```

### 请求提交

```text
cache_unfinished_req
├── insert
├── match_prefix
├── req_to_token_pool.write
├── dec_lock_ref(old)
└── inc_lock_ref(new)

cache_finished_req
├── insert
├── allocator.free(non-owned suffix)
└── dec_lock_ref(last_node)
```

### L1 分配、物理写入与提交入口

```text
ScheduleBatch.prepare_for_extend
└── alloc_for_extend
    ├── alloc_req_slots
    ├── alloc_token_slots / alloc_paged_token_slots_extend
    │   ├── allocator.alloc
    │   └── evict_from_tree_cache（空间不足时）
    └── write_cache_indices
        └── req_to_token_pool.write

attention/model forward
└── device KV pool.set_kv_buffer
    └── 把当前 layer 的 K/V 字节写入已分配 slots

SchedulerBatchResultProcessor.process_batch_result_prefill/decode
└── maybe_cache_unfinished_req / release_kv_cache
    └── cache_unfinished_req / cache_finished_req
```

第一段只建立“请求位置 -> L1 slot”的映射，第二段才写真实 KV 字节，第三段在 forward 结果可提交后把 slots 的所有权转给 Radix 或释放。详细状态变化见 [03a](03a-l1-memory-and-request-mapping.md)。

### 淘汰

```text
evict
├── eviction strategy priority
├── _delete_leaf
├── _update_leaf_status
└── allocator.free
```

## 15.4 经典本地读链

```text
Req.init_next_round_input
└── match_prefix_for_req
    └── HiRadixCache.match_prefix
        └── HiRadixCache._match_prefix_helper

PrefillAdder.add_one_req
└── HiRadixCache.init_load_back
    └── HiRadixCache.load_back
        ├── inc_lock_ref
        ├── TreeNode.protect_host
        ├── HiCacheController.load
        │   └── device allocator.alloc
        └── publish node.value

Scheduler._get_new_batch_prefill_raw
└── ready_to_load_host_cache
    └── HiCacheController.start_loading
        ├── CacheOperation.merge_ops
        ├── move_indices
        ├── HostPool.load_to_device_per_layer
        └── LayerLoadingEvent.complete
```

## 15.5 L3 预取链

```text
Scheduler._prefetch_kvcache
└── HiRadixCache.prefetch_from_storage
    ├── TreeNode.protect_host(anchor)
    ├── HostKVCache.alloc
    ├── evict_host (retry)
    └── HiCacheController.prefetch
        └── prefetch_queue.put

prefetch_thread_func
├── _storage_hit_query
│   ├── get_hash_str
│   │   └── get_native_hash
│   └── storage_backend.batch_exists
├── _all_reduce_prefetch_groups(MIN)
├── prefetch_revoke_queue.put OR
└── prefetch_buffer.put

prefetch_io_aux_func
└── _page_transfer
    ├── page_get_func
    ├── PrefetchOperation.increment
    └── append_host_mem_release

Scheduler._get_new_batch_prefill_raw
└── HiRadixCache.check_prefetch_progress
    ├── can_terminate_prefetch
    ├── controller.terminate_prefetch
    ├── _insert_helper_host
    └── HostKVCache.free(duplicate/tail)
```

## 15.6 写链

```text
HiRadixCache._inc_hit_count / eviction
└── write_backup
    ├── HiCacheController.write
    │   ├── HostKVCache.alloc
    │   ├── write_queue.append(CacheOperation)
    │   └── start_writing（返回前立即调用）
    │       ├── CacheOperation.merge_ops（经典路径通常只有当前 op）
    │       ├── move_indices
    │       ├── HostPool.backup_from_device_all_layer
    │       └── ack_write_queue.append
    ├── node.host_value = allocated indices
    ├── _track_write_through_node
    └── inc_lock_ref

HiRadixCache.writing_check
└── _finish_write_through_ack
    ├── clear pending
    ├── write_backup_storage
    │   ├── _concat_split_chain (if split)
    │   ├── HiCacheController.write_storage
    │   └── TreeNode.protect_host
    └── dec_lock_ref

backup_thread_func
└── _page_backup
    ├── page_set_func
    └── ack_backup_queue.put

drain_storage_control_queues
└── release ongoing_backup host ref
```

## 15.7 HiRadix 生命周期函数组

### 分布式同步

```text
_all_reduce_attn_groups
_barrier_attn_groups
_all_reduce
_pp_sync
_drain_async_work
```

### Storage 生命周期

```text
_parse_storage_backend_extra_config
_apply_storage_runtime_config
attach_storage_backend
detach_storage_backend
_force_release_pending_storage_ops
shutdown
clear_storage_backend
```

### Control queues

```text
_drain_storage_control_queues_local
_drain_storage_control_queues_impl
drain_storage_control_queues
check_hicache_events
```

### Prefetch 生命周期

```text
prefetch_from_storage
_prefetch_timeout_check_linear_func
can_terminate_prefetch
check_prefetch_progress
terminate_prefetch
pop_prefetch_loaded_tokens
release_aborted_request
```

## 15.8 Controller 类型与 operation

### 小数据对象

```text
LayerLoadingEvent
LayerDoneCounter
CacheOperation
StorageOperation
PrefetchOperation
```

先读这些类的字段，再读 Controller。它们就是跨阶段状态载体。

### L1/L2

```text
write -> start_writing
load -> move_indices -> start_loading
evict_device / evict_host
set_draft_kv_pool
```

### L3

```text
attach/detach/reset
prefetch/terminate_prefetch
_storage_hit_query
prefetch_thread_func
prefetch_io_aux_func
_page_transfer
write_storage
backup_thread_func
_page_backup
```

### Backend adapter

```text
_generic_page_get / _generic_page_set
_page_get_zero_copy / _page_set_zero_copy
_draft_page_get* / _draft_page_set*
```

## 15.9 Host pool 函数组

每个具体 pool 都按同一接口读：

```text
get_size_per_token / get_ksize_per_token
init_kv_buffer
clear / available_size / alloc / free
load_to_device_per_layer
backup_from_device_all_layer
get_data_page
get_dummy_flat_data_page
set_from_flat_data_page
get_page_buffer_meta
is_stride_page_aligned
```

对不同模型比较这些函数，比顺序阅读 `memory_pool_host.py` 更高效。

## 15.10 Storage 函数组

### Contract

```text
register_mem_pool_host / register_mem_host_pool_v2
batch_exists / batch_exists_v2
batch_get / batch_get_v1 / batch_get_v2
batch_set / batch_set_v1 / batch_set_v2
get / set / exists / clear / get_stats
```

### Transfer/result

```text
PoolTransfer
SidecarPoolSpec
PoolTransferResult.empty/update_*
```

### File backend

```text
key/path helpers
MetadataCache
get/set/exists
_read_page/_write_page/_batch_io_v2
batch_exists_v2/batch_get_v2/batch_set_v2
clear
```

### 内置观测

```text
Req.cached_tokens_device/host/storage
SchedulerOutputStreamer.get_cached_tokens_details
TokenizerMetricsCollector.observe_one_finished_request
StorageMetrics
StorageMetricsCollector.log_prefetched_tokens / log_backuped_tokens
StorageMetricsCollector.log_storage_metrics
RadixCacheMetricsCollector.increment_*_num_tokens
RadixCacheMetricsCollector.observe_*_duration
```

这组函数覆盖请求最终有效缓存 token 的来源归因、成功 prefetch/backup token、backend 提供的页数/带宽，以及 eviction/load-back；完整诊断漏斗仍需额外插桩，见 [13.13](13-performance.md#1313-建议的窗口与累积指标需要插桩)。

## 15.11 Unified/Hybrid 入口

```text
UnifiedRadixCache.init_hicache
├── attach_hybrid_pool_to_unified_cache
│   ├── _select_strategy
│   ├── strategy.build
│   └── _apply_stack_result
└── _apply_storage_runtime_config

Unified write_backup/load_back/prefetch_from_storage
├── component.build_hicache_transfers
├── _build_sidecar_transfers
├── HybridCacheController.write/load/prefetch
└── component.commit_hicache_transfer
```

先用 [A13b](appendix/A13b-tree-component.md) 建立组件接口，再按模型读 [Full](appendix/A13c-full-component.md)、[SWA](appendix/A13d-swa-component.md) 或 [Mamba](appendix/A13e-mamba-component.md)，最后跳到对应 assembler strategy。普通 MHA 部署无需顺序阅读所有混合组件。

## 15.12 如何使用自动附录

每个函数条目结构：

```text
位置 / 同步形式 / 签名
作用
读取的 self 状态
写入的 self 状态
直接调用
控制结构
初学者注意
完整源码折叠块
```

类条目会额外列出继承关系、decorator 和静态可提取字段。对 `CacheInitParams`、operation、ACK/result 等状态对象，应先读类契约，再读消费它们的函数；否则很容易把“字段非空”误当成“异步数据已经可读”。

源码、行号、签名、调用和状态访问由 AST 机械提取；“作用”和“初学者注意”是启发式导航摘要，关键路径做了人工覆盖，但仍不替代语义判断。例如 `self.method` 会被列在“读取对象状态”中，因为语法上是属性读取；正文负责解释它是方法还是数据字段。

源码改变后运行：

```bash
python3 docs/hicache_source_guide/tools/generate_function_appendix.py --check
```

若失败，先看 `coverage.json` 的 source SHA 和函数行号变化，再重新生成并复核相关正文。

下一章：[循序渐进的源码阅读练习](16-reading-labs.md)。

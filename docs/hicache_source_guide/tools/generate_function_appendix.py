#!/usr/bin/env python3
"""Generate deterministic, source-pinned HiCache function reference appendices.

The hand-written chapters explain the important execution paths.  This generator
provides the exhaustive layer: every function in the selected HiCache runtime
files is copied verbatim and accompanied by a small AST-derived fact sheet.

Usage:
    python3 docs/hicache_source_guide/tools/generate_function_appendix.py
    python3 docs/hicache_source_guide/tools/generate_function_appendix.py --check
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = ROOT / "docs/hicache_source_guide/appendix"


@dataclass(frozen=True)
class SourceSpec:
    output_name: str
    source_path: str
    title: str
    role: str
    include_names: Optional[frozenset[str]] = None
    include_classes: Optional[frozenset[str]] = None


SPECS = [
    SourceSpec(
        "A01-registry.md",
        "python/sglang/srt/mem_cache/registry.py",
        "Radix Cache 注册与实现选择",
        "决定启动时创建经典 Radix、HiRadix 还是 UnifiedRadixCache。",
    ),
    SourceSpec(
        "A02-kv-cache-builder.md",
        "python/sglang/srt/mem_cache/kv_cache_builder.py",
        "KV Cache 构建入口",
        "把模型 KV 池、请求索引池和服务参数组装成树缓存。",
    ),
    SourceSpec(
        "A02b-prefix-cache-contract.md",
        "python/sglang/srt/mem_cache/base_prefix_cache.py",
        "Prefix Cache 公共契约",
        "定义 match/insert/evict/lock/HiCache hook 的参数、结果和所有缓存实现共用接口。",
    ),
    SourceSpec(
        "A02c-cache-init-params.md",
        "python/sglang/srt/mem_cache/cache_init_params.py",
        "CacheInitParams 启动数据契约",
        "集中承载 tree cache 创建时需要的 pool、并行组、页大小和功能开关。",
        frozenset(),
        frozenset({"CacheInitParams"}),
    ),
    SourceSpec(
        "A02d-l1-memory-pools.md",
        "python/sglang/srt/mem_cache/memory_pool.py",
        "L1 请求映射与设备 KV Pool",
        "完整覆盖请求槽、Mamba/Hybrid 请求映射，以及 MHA/MLA/FP4/page-major/稀疏 L1 buffer 的读写接口。",
    ),
    SourceSpec(
        "A02e-device-allocator-base.md",
        "python/sglang/srt/mem_cache/allocator/base.py",
        "L1 KV 槽位分配器契约",
        "定义设备 KV allocator 的容量口径、批量释放、状态回滚和分页扩展接口。",
    ),
    SourceSpec(
        "A02f-token-allocator.md",
        "python/sglang/srt/mem_cache/allocator/token.py",
        "逐 Token L1 Allocator",
        "管理 page_size=1 时的 GPU KV 槽位 free/release 集合。",
    ),
    SourceSpec(
        "A02g-paged-allocator.md",
        "python/sglang/srt/mem_cache/allocator/paged.py",
        "分页 L1 Allocator",
        "管理 page_size>1 时的页分配，并为 extend/decode 补齐或新建物理页。",
    ),
    SourceSpec(
        "A02h-request-allocation-flow.md",
        "python/sglang/srt/mem_cache/common.py",
        "请求槽、KV 槽与提交释放流程",
        "把 Scheduler batch 预算落实为请求行和设备 KV slots，并在结束时把索引交给 Radix Tree 或 allocator。",
    ),
    SourceSpec(
        "A02i-hisparse-allocator.md",
        "python/sglang/srt/mem_cache/allocator/hisparse.py",
        "HiSparse L1 Allocator",
        "协调完整 KV、压缩 KV、稀疏 device buffer 与 SWA 映射的复合分配和释放。",
    ),
    SourceSpec(
        "A02j-swa-allocator.md",
        "python/sglang/srt/mem_cache/allocator/swa.py",
        "SWA L1 Allocator",
        "管理 Full Attention 与 Sliding Window Attention 的共享页、映射、扩展和回滚。",
    ),
    SourceSpec(
        "A02k-mamba-allocator.md",
        "python/sglang/srt/mem_cache/allocator/mamba.py",
        "Mamba 状态槽 Allocator",
        "为每个请求分配和释放 recurrent state 槽，并支持 batch 级暂存提交。",
    ),
    SourceSpec(
        "A03-radix-cache.md",
        "python/sglang/srt/mem_cache/radix_cache.py",
        "基础 Radix Cache",
        "定义键、树节点、前缀匹配、插入、分裂、锁和 L1 淘汰算法。",
    ),
    SourceSpec(
        "A03b-eviction-policy.md",
        "python/sglang/srt/mem_cache/evict_policy.py",
        "淘汰优先级策略",
        "把 LRU/LFU/FIFO/MRU/FILO/请求优先级等策略转换成可比较的 leaf priority。",
    ),
    SourceSpec(
        "A03c-cache-events.md",
        "python/sglang/srt/mem_cache/events.py",
        "KV Cache 事件发布",
        "在节点进入/离开 GPU 或 CPU 层时产生供下游索引和路由消费的事件。",
    ),
    SourceSpec(
        "A03d-session-radix.md",
        "python/sglang/srt/mem_cache/session_radix_cache.py",
        "Session Radix 生命周期",
        "跟踪 session 持有的 leaf 和关闭状态，并在 session 释放时归还树引用。",
    ),
    SourceSpec(
        "A04-hiradix-cache.md",
        "python/sglang/srt/mem_cache/hiradix_cache.py",
        "经典 HiRadixCache",
        "在 RadixCache 上增加 L2、L3、异步读写、跨 rank 同步和分层淘汰。",
    ),
    SourceSpec(
        "A05-cache-controller.md",
        "python/sglang/srt/managers/cache_controller.py",
        "经典 HiCacheController",
        "管理 CUDA stream、事件、工作队列、后台线程以及 L1/L2/L3 数据搬运。",
    ),
    SourceSpec(
        "A06-host-pool-base.md",
        "python/sglang/srt/mem_cache/pool_host/base.py",
        "L2 Host Pool 基类",
        "计算 L2 容量，预分配固定页槽，并提供线程安全的分配和释放。",
    ),
    SourceSpec(
        "A07-host-pool-common.md",
        "python/sglang/srt/mem_cache/pool_host/common.py",
        "L2 内存分配器适配",
        "选择 pinned/普通/外部 host allocator，并统一 host tensor 的分配接口。",
    ),
    SourceSpec(
        "A08-host-pool-mha.md",
        "python/sglang/srt/mem_cache/pool_host/mha.py",
        "MHA L2 数据布局与搬运",
        "实现 MHA、K-only 和非对称 K/V 的各类 L2 布局、页视图与搬运分派。",
    ),
    SourceSpec(
        "A09-model-specific-host-pools.md",
        "python/sglang/srt/mem_cache/memory_pool_host.py",
        "模型特化 L2 Pool",
        "实现 MLA、Mamba、DeepSeek V4、DSA 以及多 pool 组合的 host 缓存。",
    ),
    SourceSpec(
        "A09b-hisparse-host.md",
        "python/sglang/srt/mem_cache/pool_host/hisparse.py",
        "HiSparse Host 分配辅助",
        "把 token 请求按页换算为 page slots，并为稀疏注意力 pool 暴露一致的 host indices。",
    ),
    SourceSpec(
        "A10-storage-contract.md",
        "python/sglang/srt/mem_cache/hicache_storage.py",
        "L3 接口与文件后端",
        "定义 L3 后端契约、跨 pool 传输描述、命中语义和参考文件后端。",
    ),
    SourceSpec(
        "A11-storage-factory.md",
        "python/sglang/srt/mem_cache/storage/backend_factory.py",
        "L3 后端工厂",
        "延迟加载内置或动态 L3 实现，并传入与模型/rank/布局有关的配置。",
    ),
    SourceSpec(
        "A12-jit-python.md",
        "python/sglang/jit_kernel/hicache.py",
        "HiCache JIT Python 包装",
        "按元素宽度编译并调用 L1/L2 搬运内核，失败时允许上层选择备用实现。",
    ),
    SourceSpec(
        "A12b-native-hash-loader.md",
        "python/sglang/srt/mem_cache/cpp_utils/native_hash.py",
        "页 Hash 原生扩展加载器",
        "检查平台与 CPU 能力，JIT 编译 C++/OpenSSL hash 扩展，并整理普通/bigram token 输入。",
    ),
    SourceSpec(
        "A12c-hash-utils.md",
        "python/sglang/srt/mem_cache/utils.py",
        "页 Hash 工具函数",
        "生成链式页 hash，并在 Radix 节点分裂时同步切分 hash 元数据。",
        frozenset(
            {
                "get_hash_str",
                "get_eviction_strategy",
                "hash_str_to_int64",
                "compute_node_hash_values",
                "split_node_hash_value",
            }
        ),
    ),
    SourceSpec(
        "A13-unified-radix-cache.md",
        "python/sglang/srt/mem_cache/unified_radix_cache.py",
        "UnifiedRadixCache",
        "用组件化节点同时管理 Full KV、SWA、Mamba 等状态，并接入 HiCache。",
    ),
    SourceSpec(
        "A13b-tree-component.md",
        "python/sglang/srt/mem_cache/unified_cache_components/tree_component.py",
        "Unified TreeComponent 契约",
        "定义组件类型、组件数据、匹配/插入/锁/淘汰与 HiCache transfer 的策略接口。",
    ),
    SourceSpec(
        "A13c-full-component.md",
        "python/sglang/srt/mem_cache/unified_cache_components/full_component.py",
        "Unified Full KV Component",
        "实现完整 attention KV 的匹配、锁、分裂、设备/Host 淘汰和分层传输提交。",
    ),
    SourceSpec(
        "A13d-swa-component.md",
        "python/sglang/srt/mem_cache/unified_cache_components/swa_component.py",
        "Unified SWA Component",
        "实现滑动窗口 KV 的索引翻译、窗口锁、尾部状态、L2/L3 传输和独立回收。",
    ),
    SourceSpec(
        "A13e-mamba-component.md",
        "python/sglang/srt/mem_cache/unified_cache_components/mamba_component.py",
        "Unified Mamba Component",
        "实现 recurrent state/checkpoint 的匹配、槽位生命周期、分裂、传输和回收。",
    ),
    SourceSpec(
        "A14-hybrid-controller.md",
        "python/sglang/srt/mem_cache/hybrid_cache/hybrid_cache_controller.py",
        "HybridCacheController",
        "把经典控制器扩展为一次操作携带多个 pool 的复合搬运。",
    ),
    SourceSpec(
        "A15-hybrid-pool-assembler.md",
        "python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py",
        "混合模型 Pool 装配器",
        "根据模型组件选择策略，构造 host pool、sidecar pool 和 Unified 组件。",
    ),
    SourceSpec(
        "A16-decode-hicache.md",
        "python/sglang/srt/disaggregation/decode_hicache_mixin.py",
        "PD Decode HiCache 路径",
        "在 prefill/decode 分离场景查询、预取、回载并提交本地缓存恢复结果。",
    ),
    SourceSpec(
        "A17-scheduler-integration.md",
        "python/sglang/srt/managers/scheduler.py",
        "Scheduler 接入点",
        "展示普通服务路径何时发起 L3 预取、轮询完成、触发 L2 到 L1 回载及清理。",
        frozenset(
            {
                "Scheduler._prefetch_kvcache",
                "Scheduler._get_new_batch_prefill_raw",
                "Scheduler.clear_hicache_storage_wrapped",
                "Scheduler.attach_hicache_storage_wrapped",
                "Scheduler.detach_hicache_storage_wrapped",
                "Scheduler.abort_request",
            }
        ),
    ),
    SourceSpec(
        "A18-schedule-policy.md",
        "python/sglang/srt/managers/schedule_policy.py",
        "调度预算与回载",
        "展示命中结果如何变成 token 预算，以及请求入批前如何调用 init_load_back。",
        frozenset({"match_prefix_for_req", "PrefillAdder.add_one_req"}),
    ),
    SourceSpec(
        "A19-request-batch-state.md",
        "python/sglang/srt/managers/schedule_batch.py",
        "请求与 Batch 中的 HiCache 状态",
        "展示请求记录 L1/L2/L3 命中长度、重新匹配以及 forward 前等待回载的方式。",
        frozenset(
            {
                "Req.needs_host_load_back",
                "Req._cache_commit_len",
                "Req.pop_committed_kv_cache",
                "Req.pop_overallocated_kv_cache",
                "Req.init_next_round_input",
                "ScheduleBatch.prepare_for_extend",
                "ScheduleBatch.prepare_for_decode",
            }
        ),
    ),
    SourceSpec(
        "A19b-result-cache-commit.md",
        "python/sglang/srt/managers/scheduler_components/batch_result_processor.py",
        "Forward 结果提交到 Prefix Cache",
        "展示 prefill/decode 结果返回 Scheduler 后，何时缓存未完成请求、提交完成请求并释放请求资源。",
        frozenset(
            {
                "SchedulerBatchResultProcessor.process_batch_result_prefill",
                "SchedulerBatchResultProcessor.process_batch_result_decode",
            }
        ),
        frozenset({"SchedulerBatchResultProcessor"}),
    ),
    SourceSpec(
        "A19c-hicache-metrics.md",
        "python/sglang/srt/observability/metrics_collector.py",
        "HiCache 内置 Prometheus 指标",
        "定义当前主分支已有的请求缓存来源、L3 prefetch/backup 与 Radix eviction/load-back counter 和 histogram。",
        frozenset(
            {
                "TokenizerMetricsCollector.__init__",
                "TokenizerMetricsCollector.observe_one_finished_request",
                "StorageMetricsCollector.__init__",
                "StorageMetricsCollector.log_prefetched_tokens",
                "StorageMetricsCollector.log_backuped_tokens",
                "StorageMetricsCollector._log_histogram",
                "StorageMetricsCollector.log_storage_metrics",
                "RadixCacheMetricsCollector.__init__",
                "RadixCacheMetricsCollector.increment_eviction_num_tokens",
                "RadixCacheMetricsCollector.increment_load_back_num_tokens",
                "RadixCacheMetricsCollector.observe_eviction_duration",
                "RadixCacheMetricsCollector.observe_load_back_duration",
            }
        ),
        frozenset(
            {
                "TokenizerMetricsCollector",
                "StorageMetrics",
                "StorageMetricsCollector",
                "RadixCacheMetricsCollector",
            }
        ),
    ),
    SourceSpec(
        "A19d-cache-hit-reporting.md",
        "python/sglang/srt/managers/scheduler_components/output_streamer.py",
        "请求级缓存命中来源上报",
        "把 Req 中的 device/host/storage 缓存 token 拆分装入输出消息，供 API、benchmark 与请求指标消费。",
        frozenset(
            {
                "SchedulerOutputStreamer._get_storage_backend_type",
                "SchedulerOutputStreamer.get_cached_tokens_details",
                "SchedulerOutputStreamer._stream_output_generation",
                "SchedulerOutputStreamer._stream_output_embedding",
                "_GenerationStreamAccumulator.accept",
                "_GenerationStreamAccumulator.to_payload",
            }
        ),
        frozenset(
            {
                "SchedulerOutputStreamer",
                "_GenerationStreamAccumulator",
            }
        ),
    ),
    SourceSpec(
        "A20-server-args.md",
        "python/sglang/srt/server_args.py",
        "HiCache 参数归一化",
        "展示 layout、I/O backend 与 L3 backend 的兼容性修正。",
        frozenset(
            {
                "ServerArgs._handle_hicache",
                "ServerArgs._resolve_layout_io_compatibility",
                "ServerArgs._resolve_storage_layout_compatibility",
            }
        ),
    ),
]


CPP_SPECS = [
    (
        "python/sglang/srt/mem_cache/cpp_utils/hash_binding.cpp",
        "链式 SHA-256 页 Hash 的 PyBind/OpenSSL 实现",
    ),
    (
        "python/sglang/jit_kernel/csrc/kvcacheio/hicache.cuh",
        "通用 L1/L2 向量化搬运内核",
    ),
    (
        "python/sglang/jit_kernel/csrc/kvcacheio/staged_write_back.cuh",
        "layer-first 到 page-first 的分阶段写回内核",
    ),
]


PURPOSE_OVERRIDES = {
    "default_radix_cache_factory": "按服务参数选择基础 RadixCache、HiRadixCache 或 UnifiedRadixCache，并完成 HiCache 附件装配。",
    "create_tree_cache": "调用当前注册的工厂创建树缓存；这是 builder 与具体缓存实现之间的稳定边界。",
    "build_kv_cache": "KV 缓存启动总入口：发现模型所需的池，构造 CacheInitParams，再创建 tree cache。",
    "RadixKey.match": "计算两个 RadixKey 的最长公共前缀；先跳跃探测再二分定位，并按 page_size 向下对齐。",
    "RadixKey.hash_page": "把上一页 hash 与当前页 token 串联后计算链式页 hash，作为 L3 的内容地址；当前实现不会把 RadixKey.extra_key 传给 native hash。",
    "RadixCache.match_prefix": "把请求 token 映射到 Radix 树，返回可直接复用的 L1 索引和最后命中的节点。",
    "RadixCache._match_prefix_helper": "逐层走树；完全命中继续，部分命中则分裂节点，并拼接命中节点的 value。",
    "RadixCache._split_node": "把一个压缩边拆成公共前缀节点和剩余后缀节点，同时维护父子关系、锁和叶集合。",
    "RadixCache._insert_helper": "递归插入键；复用公共前缀，必要时分裂，最后创建只保存新后缀的节点。",
    "RadixCache.cache_unfinished_req": "把正在执行请求新算出的 KV 插入树，再重新匹配并重写 req_to_token_pool 映射。",
    "RadixCache.inc_lock_ref": "沿当前节点到根增加引用锁，使正在使用的前缀不能被 L1 淘汰。",
    "RadixCache.dec_lock_ref": "沿祖先链释放引用锁，并把重新可淘汰的 token 计回容量。",
    "HiRadixCache.write_backup": "为节点分配 L2 槽，提交 L1 到 L2 异步复制，并登记 ACK 到来前的节点锁。",
    "HiRadixCache._finish_write_through_ack": "消费 D2H 完成 ACK；发布 L2 可用状态、释放 L1 锁，并按配置继续写 L3。",
    "HiRadixCache.write_backup_storage": "用节点页 hash 和 L2 索引创建 L3 写操作，并持有 host 引用直到 L3 ACK。",
    "HiRadixCache.load_back": "收集 L2 命中链，检查阈值与 L1 配额，分配 GPU 槽并异步执行 H2D。",
    "HiRadixCache.init_load_back": "Scheduler 入批前的回载入口；成功时返回刚恢复到 L1 的连续索引。",
    "HiRadixCache.prefetch_from_storage": "为尚未本地命中的后缀预留 L2 页，并把链式 hash 查询/读取交给后台线程。",
    "HiRadixCache.check_prefetch_progress": "在调度线程收割 L3 预取结果，把成功页插入 host Radix 树并释放未使用预留页。",
    "HiRadixCache.evict": "根据写策略和节点状态选择 L1 删除、降级到 L2 或先写回再淘汰。",
    "HiRadixCache.evict_host": "只从可安全释放的 host 叶节点回收 L2 槽，避免破坏连续前缀和在途 I/O。",
    "HiRadixCache.check_hicache_events": "每个 scheduler step 的维护入口：轮询 D2H/H2D、L3 控制队列和指标。",
    "LayerLoadingEvent.wait": "模型执行到某层前等待该层 H2D 事件，实现回载与逐层计算重叠。",
    "LayerDoneCounter": "三个环形计数槽协调连续多个 batch 的逐层加载生产者和消费者。",
    "CacheOperation.merge_ops": "把多个同方向 L1/L2 小操作合并成一次批量搬运，同时保留每个节点 ACK。",
    "HiCacheController.write": "分配 L2 索引、追加 D2H operation，并在当前实现中立即调用 start_writing；返回索引仍只表示异步复制已提交，不表示复制完成。",
    "HiCacheController.start_writing": "合并当前 write_queue，在独立 stream 上搬运所有层，并把 CUDA event 和 ACK 放入完成队列；由于 write 会立即调用它，经典路径通常只有单个 operation。",
    "HiCacheController.load": "分配 L1 索引并排队 H2D；真正的批量复制由 start_loading 启动。",
    "HiCacheController.start_loading": "合并 H2D 操作，为每层记录事件，使模型可按层等待并与计算流水重叠。",
    "HiCacheController._storage_hit_query": "按链式页 hash 批量询问 L3，遇到第一个缺页就截断为最长连续命中前缀。",
    "HiCacheController.prefetch_thread_func": "L3 查询线程主循环：同步各 rank 命中长度，裁剪预留页，再交给 I/O 线程读取。",
    "HiCacheController.prefetch_io_aux_func": "L3 数据读取线程主循环：分批把成功页填入 L2，并把未使用尾部送回释放队列。",
    "HiCacheController._page_transfer": "按 STORAGE_BATCH_SIZE 分批读取 L3；只把连续且完整成功的 batch 计为可用。",
    "HiCacheController._page_backup": "把一个 L2 节点按页写入 L3，并记录完整成功的 token 数供 ACK/指标使用。",
    "HostKVCache.alloc": "从固定 L2 页槽集合中原子取出连续数量的逻辑 token 槽，并标记为占用。",
    "HostKVCache.free": "把 L2 槽原子归还；包含重复释放检查，以尽早暴露生命周期错误。",
    "MHATokenToKVPoolHost.init_kv_buffer": "根据 layout 创建 L2 K/V 大张量；不同维度顺序决定 D2H、H2D 与 L3 I/O 的连续性。",
    "MHATokenToKVPoolHost.load_to_device_per_layer": "按 layout 和 io_backend 分派某一层的 L2 到 L1 复制。",
    "MHATokenToKVPoolHost.backup_from_device_all_layer": "按 layout 和 io_backend 分派全部层的 L1 到 L2 复制。",
    "HiCacheStorage.batch_exists": "默认逐页 exists，返回从第一页开始连续存在的页数，而不是所有命中页的总数。",
    "HiCacheFile.batch_exists_v2": "同时检查 KV 与辅助 pool，按各 pool 命中策略收缩为可安全复用的最长前缀。",
    "HiCacheFile.set": "以临时文件加原子 replace 写一页，并通过 LRU evictor 预留/提交容量。",
    "StorageBackendFactory.create_backend": "按名字延迟加载 L3 类，区分内置后端和用户动态后端后创建实例。",
    "transfer_hicache_one_layer": "整理张量视图并调用 JIT 内核复制一个 layer 的离散 token 槽。",
    "transfer_hicache_all_layer": "传入各 layer 基址数组，一次 kernel 调度复制所有层的离散 token 槽。",
    "_load_native_hash_module": "在受支持的平台 JIT 编译并缓存 C++ hash 扩展；失败会保留原异常作为 cause。",
    "_native_hash_input": "把 list/array/RadixKey/bigram 输入规范化为原生扩展需要的连续整数 buffer 和逻辑宽度；RadixKey.extra_key 在这里不会进入 hash 输入。",
    "get_native_hash": "调用原生扩展生成整段或逐页链式 SHA-256 十六进制 hash。",
    "get_hash_str": "把可选上一页十六进制 hash 转为 digest，并统一调用原生 hash 实现；调用参数没有独立 namespace 字段。",
    "ReqToTokenPool.alloc": "为本批请求分配 req_to_token 矩阵行；chunked/已提交请求可复用已有 req_pool_idx。",
    "ReqToTokenPool.free": "归还请求矩阵行并把 req.req_pool_idx 清空；它不直接释放该行指向的 KV slots。",
    "alloc_for_extend": "为 prefill/extend 批次分配请求行和新 KV slots，并把缓存前缀与新 slots 写入 req_to_token 映射。",
    "alloc_for_decode": "为每个 decode 请求分配新 KV slot/page，并把当前位置写入 req_to_token 映射。",
    "release_kv_cache": "请求结束时先让 tree cache 接管可缓存 KV，再释放 speculative/未提交尾部和请求行。",
    "SchedulerBatchResultProcessor.process_batch_result_prefill": "处理 prefill forward 结果；完成请求走 release_kv_cache，未完成请求走 maybe_cache_unfinished_req。",
    "SchedulerBatchResultProcessor.process_batch_result_decode": "处理 decode forward 结果并更新请求；完成、retract 和继续运行分支决定 KV 的提交或释放时机。",
    "split_node_hash_value": "按节点 token 分裂点换算页数，同步切开 prefix/suffix 的 hash 列表。",
    "UnifiedRadixCache.init_hicache": "为组件化 Radix 树装配多 pool host cache、HybridController、sidecar 与 L3 参数。",
    "UnifiedRadixCache.write_backup": "为 Full KV 和所有辅助组件构造复合 D2H 操作，并原子提交各组件 host 索引。",
    "UnifiedRadixCache.load_back": "为 Full KV 和辅助状态构造复合 H2D 操作，保证组件恢复结果与树节点一致。",
    "UnifiedRadixCache.prefetch_from_storage": "为 Full KV 及辅助 pool 分配 L2 资源并发起组件感知的 L3 预取。",
    "HybridCacheController._resolve_pool_transfers_allocation": "解析复合传输中每个 pool 的索引来源并分配独立 host 槽。",
    "attach_hybrid_pool_to_unified_cache": "选择匹配模型的装配策略，并把构建结果连接到 UnifiedRadixCache。",
    "Scheduler._prefetch_kvcache": "请求进入等待队列时，先做本地匹配，再对 L1/L2 后面的 token 后缀发起 L3 预取。",
    "Scheduler._get_new_batch_prefill_raw": "构建 prefill batch；在此轮询预取、重新匹配、触发 L2 回载并启动批量 H2D。",
    "PrefillAdder.add_one_req": "执行入批容量核算；L2 命中只减少计算 token，但回载仍需预留相应 L1 容量。",
    "Req.init_next_round_input": "重新匹配树并把 L1、L2 及各组件命中长度写回请求状态。",
    "ScheduleBatch.prepare_for_extend": "准备 forward 输入，并让消费方绑定本批 H2D 的层级完成计数。",
    "TokenizerMetricsCollector.__init__": "创建请求级 Prometheus 指标；其中 cached_tokens_total 通过 cache_source label 区分 device、host 和具体 storage backend。",
    "TokenizerMetricsCollector.observe_one_finished_request": "请求结束时累计 prompt/generation/cached token；有缓存来源明细时分别写入 device、host 和 storage_<backend> 标签。",
    "StorageMetricsCollector.__init__": "创建 L3 成功 token、每批页数和读写带宽的 Prometheus Counter/Histogram。",
    "StorageMetricsCollector.log_prefetched_tokens": "把 Scheduler 已收割并新增发布到 L2 树的 L3 token 数累加到 Prometheus Counter；本函数不发起预取。",
    "StorageMetricsCollector.log_backuped_tokens": "把 L3 backup ACK 报告的已完成 token 数累加到 Prometheus Counter；本函数不执行写入。",
    "StorageMetricsCollector._log_histogram": "把一个后端统计样本写入指定 Prometheus Histogram。",
    "StorageMetricsCollector.log_storage_metrics": "消费 backend get_stats 返回的页数和带宽样本，并写入对应 Prometheus Histogram。",
    "RadixCacheMetricsCollector.__init__": "创建 L1 eviction/load-back token Counter 和 host 侧方法耗时 Histogram。",
    "RadixCacheMetricsCollector.increment_eviction_num_tokens": "累计 Tree eviction 本次处理的 token 数；本函数不执行淘汰。",
    "RadixCacheMetricsCollector.increment_load_back_num_tokens": "累计 load_back 已成功分配并提交 H2D 的 token 数；本函数不执行回载。",
    "RadixCacheMetricsCollector.observe_eviction_duration": "记录一次 Tree eviction 方法的 host 侧调用时长。",
    "RadixCacheMetricsCollector.observe_load_back_duration": "记录一次 Tree load_back 方法的 host 侧调用时长。",
    "SchedulerOutputStreamer._get_storage_backend_type": "从当前 tree cache 的 controller 读取 L3 backend 类名，作为请求缓存来源标签。",
    "SchedulerOutputStreamer.get_cached_tokens_details": "把 Req 的 device/host/storage token 计数整理为请求级缓存来源明细，并兼容只有总命中数的旧路径。",
    "SchedulerOutputStreamer._stream_output_generation": "生成请求输出时把缓存来源提取函数交给累加器，使每个完成请求携带 cached_tokens_details。",
    "SchedulerOutputStreamer._stream_output_embedding": "构造 embedding 输出时逐请求附带 device/host/storage 缓存来源明细。",
    "_GenerationStreamAccumulator.accept": "接收一个待输出请求并收集文本、token、结束原因以及 cached_tokens_details 等字段。",
    "_GenerationStreamAccumulator.to_payload": "把累积的请求输出字段组装成 BatchTokenIDOutput，其中包含逐请求 cached_tokens_details。",
    "DecodeHiCacheTransferMixin._try_hicache_queue_load_back": "PD decode 侧等待 L3 预取后重新匹配，并把本地 L2 命中排入 H2D 恢复。",
    "TreeComponent.refresh_lru": "按 WALKDOWN/MATCH_END/INSERT_END 阶段刷新当前组件的 LRU，而不改变其他组件的淘汰顺序。",
    "SWAComponent._translate_full_to_swa": "通过专用 allocator 把 Full KV slots 翻译为对应的 SWA slots。",
    "SWAComponent._restore_device_value": "把已恢复的 SWA device indices 发布到节点，并同步 device/host LRU 与可淘汰计数。",
    "SWAComponent._restore_device_value_with_locked_full": "在 Full KV 仍受锁保护时重建 Full-to-SWA 映射，释放临时 Full slots，再发布 SWA device value。",
    "SWAComponent._maybe_split_leaf_for_swa_lock": "必要时按滑动窗口边界分裂 leaf，使 SWA 锁只保护仍需保留的窗口范围。",
    "SWAComponent._release_swa_host": "释放 SWA host slots；没有独立 host pool 时该索引可能来自共享/派生路径。",
    "SWAComponent._attach_swa_host_value": "把预取或备份得到的 SWA host indices 挂到组件数据，并维护 host LRU/容量。",
    "SWAComponent._commit_prefetch": "根据主 KV 的插入结果裁剪并提交 SWA 预取页，释放重复或未使用的 host slots。",
    "MambaComponent._alloc_mamba_slot": "从 Mamba 状态 allocator 获取一个请求级 recurrent state slot，失败时明确报错。",
    "MambaComponent._alloc_int8_ckpt_slot": "为可选 int8 Mamba checkpoint 分配一个槽位。",
    "MambaComponent._commit_int8_checkpoint": "把 active Mamba state 压缩提交到 checkpoint pool，并返回可由树持有的 checkpoint slots。",
    "MambaComponent._free_mamba_value": "按 value 编码区分并释放普通 Mamba state 或 checkpoint slot。",
    "eviction_priority": "计算当前组件节点的淘汰优先级；它只给策略排序，不在此函数中释放数据。",
    "create_match_validator": "创建带组件语义的节点校验闭包，决定 Full/SWA/Mamba 数据是否允许继续扩展共同命中前缀。",
    "finalize_match_result": "在树遍历结束后按当前组件状态裁剪、补充或校验最终命中结果。",
    "redistribute_on_node_split": "节点分裂时同步切分或重挂当前组件的 device/host 数据、锁和 LRU 元数据。",
    "build_hicache_transfers": "按 BACKUP_HOST、LOAD_BACK、PREFETCH 或 BACKUP_STORAGE 阶段构造当前组件的 PoolTransfer 描述。",
    "commit_hicache_transfer": "在复合操作资源已分配或 I/O 已确认后，把当前组件的 transfer 结果提交到节点并处理未使用资源。",
}


CPP_SYMBOLS = [
    ("checked_u32", "把 token id 安全收窄为 uint32，越界时拒绝生成不一致 hash。"),
    ("hash_page", "使用 OpenSSL SHA-256，把 prior digest 与当前页规范化字节串联计算。"),
    ("fill_regular_page / fill_bigram_page", "把普通 token 或 bigram 逻辑单元编码成稳定的小端 uint32 页字节。"),
    ("hash_pages_to_hex_blob", "逐页链式计算 digest，并暂存为连续十六进制字符块。"),
    ("hash_all / pages_hashes", "分别生成单个整体 hash 或按 page_size 生成 hash 列表。"),
    ("get_hash", "PyBind 暴露入口，根据 page_size 是否为空选择整体/分页模式。"),
    ("get_mem_package", "按单个元素字节数选择向量化 load/store 的打包类型。"),
    ("store_nc", "CUDA 路径使用 non-coherent/global streaming store；ROCm 路径使用兼容写法。"),
    ("load_vec / store_vec", "每个 warp 协作搬运一个 KV 元素，并按 unroll 展开向量访问。"),
    ("hicache_transfer_per_layer", "一个 worker 处理一个 token 槽，对 K/V 执行同层 src/dst 索引映射。"),
    ("hicache_transfer_all_layer", "一个 worker 处理一个 token 槽，并在内核内循环全部 layer。"),
    ("HiCacheKernel.run_one", "检查 TVM FFI 参数、计算 stride/block 数并启动 per-layer kernel。"),
    ("HiCacheKernel.run_all", "接收每层指针数组并启动 all-layer kernel。"),
    ("run_one_mla / run_all_mla", "复用同一模板但不搬运独立 V，适配 MLA 单缓冲结构。"),
    ("get_cuda_memcpy_batch_async", "运行时解析批量异步 memcpy API；不可用时返回空指针。"),
    ("copy_page_first_pages_fallback", "批量 API 不可用时逐页执行异步 device-to-host memcpy。"),
    ("try_copy_page_first_pages_batch", "构造多页 src/dst/size 数组并尝试一次批量异步 memcpy。"),
    ("run_staged_impl", "先在 GPU staging 中整理 layer-first 数据，再批量复制到 page-first host 页。"),
    ("run_all_lf_pf_staged", "MHA K/V 的 staged write-back 入口。"),
    ("run_all_mla_lf_pf_staged", "MLA 单缓冲的 staged write-back 入口。"),
]


@dataclass
class FunctionInfo:
    qualified_name: str
    node: ast.FunctionDef | ast.AsyncFunctionDef
    start_line: int
    end_line: int
    code: str


@dataclass
class ClassInfo:
    qualified_name: str
    node: ast.ClassDef
    start_line: int
    end_line: int
    code: str


def collect_classes(source: str) -> list[ClassInfo]:
    tree = ast.parse(source)
    lines = source.splitlines()
    result: list[ClassInfo] = []

    def walk(nodes: Iterable[ast.stmt], prefix: str = "") -> None:
        for node in nodes:
            if not isinstance(node, ast.ClassDef):
                continue
            qualified_name = f"{prefix}{node.name}"
            decorator_lines = [d.lineno for d in node.decorator_list]
            start = min([node.lineno, *decorator_lines])
            end = node.end_lineno or node.lineno
            code = "\n".join(lines[start - 1 : end])
            result.append(ClassInfo(qualified_name, node, start, end, code))
            walk(node.body, f"{qualified_name}.")

    walk(tree.body)
    return result


def class_fields(node: ast.ClassDef) -> list[tuple[str, str, str]]:
    fields: list[tuple[str, str, str]] = []
    for child in node.body:
        if isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
            annotation = ast.unparse(child.annotation)
            default = ast.unparse(child.value) if child.value is not None else "必填"
            fields.append((child.target.id, annotation, default))
        elif isinstance(child, ast.Assign):
            for target in child.targets:
                if isinstance(target, ast.Name) and not target.id.startswith("__"):
                    fields.append((target.id, "未标注", ast.unparse(child.value)))
    return fields


def render_class_contract(info: ClassInfo, spec: SourceSpec) -> str:
    node = info.node
    bases = ", ".join(ast.unparse(base) for base in node.bases) or "object"
    decorators = ", ".join(ast.unparse(d) for d in node.decorator_list) or "无"
    fields = class_fields(node)
    direct_methods = [
        child.name
        for child in node.body
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    body = [
        f"\n### `{info.qualified_name}`\n\n",
        f"- **位置**：`{spec.source_path}:{info.start_line}-{info.end_line}`\n",
        f"- **基类**：`{bases}`\n",
        f"- **装饰器**：`{decorators}`\n",
        f"- **直接方法**：{concise(direct_methods)}\n",
    ]
    if fields:
        body.extend(
            [
                "\n| 声明字段/常量 | 类型 | 默认值 |\n",
                "|---|---|---|\n",
                *[
                    f"| `{name}` | `{annotation}` | `{default}` |\n"
                    for name, annotation, default in fields
                ],
            ]
        )
    else:
        body.append("\n该类没有类体级字段声明；主要运行状态在 `__init__` 中写入，可结合后面的状态读写清单阅读。\n")

    if not direct_methods:
        body.extend(
            [
                "\n<details>\n<summary>展开完整数据声明</summary>\n\n",
                f"```python\n{info.code}\n```\n\n",
                "</details>\n",
            ]
        )
    return "".join(body)


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def dotted_name(node: ast.AST) -> Optional[str]:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        return ".".join(reversed(parts))
    return None


def collect_functions(source: str) -> list[FunctionInfo]:
    tree = ast.parse(source)
    lines = source.splitlines()
    result: list[FunctionInfo] = []

    def walk(nodes: Iterable[ast.stmt], prefix: str = "") -> None:
        for node in nodes:
            if isinstance(node, ast.ClassDef):
                walk(node.body, f"{prefix}{node.name}.")
                continue
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            qualified_name = f"{prefix}{node.name}"
            decorator_lines = [d.lineno for d in node.decorator_list]
            start = min([node.lineno, *decorator_lines])
            end = node.end_lineno or node.lineno
            code = "\n".join(lines[start - 1 : end])
            result.append(FunctionInfo(qualified_name, node, start, end, code))
            walk(node.body, f"{qualified_name}.<local>.")

    walk(tree.body)
    return result


def function_facts(node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, object]:
    calls: set[str] = set()
    reads: set[str] = set()
    writes: set[str] = set()
    control = {"if": 0, "for": 0, "while": 0, "try": 0, "with": 0, "assert": 0}

    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            name = dotted_name(child.func)
            if name:
                calls.add(name)
        elif isinstance(child, ast.Attribute):
            name = dotted_name(child)
            if name and name.startswith("self."):
                if isinstance(child.ctx, (ast.Store, ast.Del)):
                    writes.add(name)
                else:
                    reads.add(name)
        elif isinstance(child, ast.If):
            control["if"] += 1
        elif isinstance(child, (ast.For, ast.AsyncFor)):
            control["for"] += 1
        elif isinstance(child, ast.While):
            control["while"] += 1
        elif isinstance(child, ast.Try):
            control["try"] += 1
        elif isinstance(child, (ast.With, ast.AsyncWith)):
            control["with"] += 1
        elif isinstance(child, ast.Assert):
            control["assert"] += 1

    reads -= writes
    return {
        "calls": sorted(calls),
        "reads": sorted(reads),
        "writes": sorted(writes),
        "control": {k: v for k, v in control.items() if v},
    }


def concise(items: Iterable[str], limit: int = 14) -> str:
    values = list(items)
    if not values:
        return "无"
    shown = values[:limit]
    suffix = f"，另有 {len(values) - limit} 项" if len(values) > limit else ""
    return "、".join(f"`{x}`" for x in shown) + suffix


def function_name_tokens(qualified_name: str) -> tuple[str, ...]:
    """Split a function name without matching short words inside other words."""
    short_name = qualified_name.split(".")[-1].strip("_")
    snake_name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", short_name).lower()
    return tuple(token for token in snake_name.split("_") if token)


def infer_purpose(info: FunctionInfo, role: str) -> str:
    direct = PURPOSE_OVERRIDES.get(info.qualified_name)
    if direct:
        return direct
    short_name = info.qualified_name.split(".")[-1]
    direct = PURPOSE_OVERRIDES.get(short_name)
    if direct:
        return direct

    tokens = set(function_name_tokens(info.qualified_name))
    patterns = [
        (("init", "initialize"), "初始化对象及其运行状态"),
        (("reset", "clear", "cleanup", "destroy", "shutdown"), "重置或释放相关资源与状态"),
        (("alloc",), "分配当前模块管理的槽位或资源"),
        (("free", "release"), "释放当前模块持有的槽位、锁或资源"),
        (("attach", "register"), "注册或连接一个运行时组件"),
        (("detach",), "断开运行时组件并处理遗留状态"),
        (("restore", "recover", "unevict"), "恢复此前淘汰、暂存或低层保存的数据与元数据"),
        (("translate", "mapping", "map"), "在不同 pool 的索引语义之间建立或查询映射"),
        (("match", "query", "exists", "contains"), "查询数据或计算可复用的连续前缀"),
        (("insert", "set", "add", "commit"), "写入或提交数据及其元数据"),
        (("prefetch",), "发起、推进或结束 L3 预取"),
        (("load", "loading", "get", "read"), "读取数据或把低层缓存恢复到高层"),
        (("backup", "write", "writing", "store"), "写出数据或把高层缓存备份到低层"),
        (("evict", "delete", "drop", "remove"), "淘汰数据并维护容量与树结构"),
        (("inc", "increment", "protect", "lock"), "增加引用/保护状态，防止在途数据被回收"),
        (("dec", "unlock"), "减少引用/保护状态，使资源重新可回收"),
        (("check", "can", "is", "has", "ready"), "检查条件或异步操作状态"),
        (("make", "build", "create", "generate", "resolve"), "构造或解析当前流程需要的对象/配置"),
        (("update", "refresh", "redistribute", "record", "track", "log"), "更新内部状态、关系或统计信息"),
    ]
    doc = ast.get_docstring(info.node, clean=True)
    doc_suffix = ""
    if doc:
        first = " ".join(doc.splitlines()).strip()
        if len(first) > 150:
            first = first[:147] + "..."
        doc_suffix = f" 源码注释摘要：{first}"
    for needles, text in patterns:
        if any(needle in tokens for needle in needles):
            return f"{text}；它位于“{role}”这一职责范围内。{doc_suffix}"
    return (
        f"完成 `{short_name}` 对应的局部职责；结合调用者理解其在“{role}”中的位置。"
        f"{doc_suffix}"
    )


def beginner_note(info: FunctionInfo, facts: dict[str, object]) -> str:
    name = info.qualified_name.lower()
    tokens = set(function_name_tokens(info.qualified_name))
    is_metrics_helper = "metricscollector." in name or bool(
        tokens.intersection({"log", "observe"})
    )
    notes = []
    if is_metrics_helper:
        notes.append("这里只更新本地统计对象，不发起缓存 I/O，也不构成数据完成屏障")
    if tokens.intersection({"alloc", "free", "evict"}):
        notes.append("重点核对资源所有权：成功、失败和提前返回三条路径都必须成对分配/释放")
    if tokens.intersection({"lock", "protect", "ref"}):
        notes.append("这里的引用计数是生命周期保护，不是 Python 对象引用计数")
    if not is_metrics_helper and tokens.intersection(
        {"prefetch", "load", "loading", "write", "writing", "backup"}
    ):
        notes.append("函数返回通常只表示已排队或已预留，真正可见性要以 event/ACK 为准")
    if "hash" in tokens:
        notes.append(
            "当前页 hash 的直接输入是 token/bigram 字节与 prior digest；RadixKey.extra_key 不会自动进入 native hash，L3 隔离还要核对 backend namespace"
        )
    elif "key" in tokens:
        notes.append(
            "RadixKey.extra_key 会影响本地树兼容性和 child key，但不要据此推断它也进入了 L3 page hash"
        )
    if facts["control"].get("assert", 0):
        notes.append("assert 表达内部不变量；触发时应追查更早的状态破坏，而不是简单移除断言")
    if isinstance(info.node, ast.AsyncFunctionDef):
        notes.append("这是 Python async 协程；不要与控制器内部 OS 线程或 CUDA 异步混为一谈")
    if not notes:
        notes.append("先看调用者传入什么，再看它修改哪些状态，最后确认返回值代表“已完成”还是“已提交”")
    return "；".join(notes[:3]) + "。"


def render_function(info: FunctionInfo, spec: SourceSpec) -> str:
    node = info.node
    facts = function_facts(node)
    args = ast.unparse(node.args)
    return_type = ast.unparse(node.returns) if node.returns is not None else "未标注"
    control = facts["control"]
    control_text = "、".join(f"{k}={v}" for k, v in control.items()) or "直线流程"
    kind = "异步协程" if isinstance(node, ast.AsyncFunctionDef) else "同步函数"
    purpose = infer_purpose(info, spec.role)

    return f"""\n## `{info.qualified_name}`\n\n- **位置**：`{spec.source_path}:{info.start_line}-{info.end_line}`\n- **形式**：{kind}\n- **签名**：`({args}) -> {return_type}`\n- **作用**：{purpose}\n- **读取的对象状态**：{concise(facts['reads'])}\n- **写入的对象状态**：{concise(facts['writes'])}\n- **直接调用**：{concise(facts['calls'])}\n- **控制结构速览**：{control_text}\n- **初学者注意**：{beginner_note(info, facts)}\n\n<details>\n<summary>展开完整源码</summary>\n\n```python\n{info.code}\n```\n\n</details>\n"""


def render_python_spec(spec: SourceSpec, revision: str) -> tuple[str, dict[str, object]]:
    source_file = ROOT / spec.source_path
    source = source_file.read_text(encoding="utf-8")
    all_functions = collect_functions(source)
    all_classes = collect_classes(source)
    if spec.include_names is None:
        functions = all_functions
    else:
        functions = [f for f in all_functions if f.qualified_name in spec.include_names]
        missing = spec.include_names - {f.qualified_name for f in functions}
        if missing:
            raise RuntimeError(f"Missing selected functions in {spec.source_path}: {missing}")

    if spec.include_classes is not None:
        classes = [c for c in all_classes if c.qualified_name in spec.include_classes]
        missing_classes = spec.include_classes - {c.qualified_name for c in classes}
        if missing_classes:
            raise RuntimeError(
                f"Missing selected classes in {spec.source_path}: {missing_classes}"
            )
    elif spec.include_names is None:
        classes = all_classes
    else:
        classes = [
            c
            for c in all_classes
            if any(
                name == c.qualified_name
                or name.startswith(f"{c.qualified_name}.")
                for name in spec.include_names
            )
        ]

    scope = (
        "本文件的全部函数（包括局部嵌套函数）"
        if spec.include_names is None
        else "与 HiCache 直接相关的选定集成函数；该大文件的其他函数不属于本书逐函数范围"
    )
    body = [
        "<!-- Generated by tools/generate_function_appendix.py. Do not edit manually. -->\n",
        f"# {spec.title}\n\n",
        f"> 源码快照：`{revision}`<br>\n",
        f"> 源文件：`{spec.source_path}`<br>\n",
        f"> SHA-256：`{sha256_text(source)}`<br>\n",
        f"> 覆盖范围：{scope}，共 **{len(functions)}** 个函数、**{len(classes)}** 个类声明。\n\n",
        f"**模块职责**：{spec.role}\n\n",
        "阅读方法：先读“作用”和“状态读写”，再展开源码。源码、行号、签名、调用和状态访问由 AST 机械提取；“作用”和“初学者注意”是启发式导航摘要，关键路径做了人工覆盖，但不替代正文或源码语义。局部函数会同时出现在外层函数代码块中并单独列出，这是有意保留的，便于完整覆盖。\n",
    ]
    if classes:
        body.append("\n## 类与状态契约\n")
        for class_info in classes:
            body.append(render_class_contract(class_info, spec))
        if functions:
            body.append("\n## 函数逐项说明\n")
    for function in functions:
        body.append(render_function(function, spec))

    metadata = {
        "output": spec.output_name,
        "source": spec.source_path,
        "source_sha256": sha256_text(source),
        "scope": "all" if spec.include_names is None else "selected",
        "class_count": len(classes),
        "classes": [
            {
                "name": c.qualified_name,
                "start_line": c.start_line,
                "end_line": c.end_line,
            }
            for c in classes
        ],
        "function_count": len(functions),
        "functions": [
            {
                "name": f.qualified_name,
                "start_line": f.start_line,
                "end_line": f.end_line,
            }
            for f in functions
        ],
    }
    return "".join(body), metadata


def render_cpp_appendix(revision: str) -> tuple[str, dict[str, object]]:
    body = [
        "<!-- Generated by tools/generate_function_appendix.py. Do not edit manually. -->\n",
        "# HiCache C++/CUDA/ROCm 内核\n\n",
        f"> 源码快照：`{revision}`。本附录保留两个内核文件的完整代码。\n\n",
        "## 符号导读\n\n",
        "| 符号 | 作用 |\n|---|---|\n",
    ]
    for symbol, explanation in CPP_SYMBOLS:
        body.append(f"| `{symbol}` | {explanation} |\n")

    files = []
    for source_path, title in CPP_SPECS:
        source = (ROOT / source_path).read_text(encoding="utf-8")
        files.append(
            {
                "source": source_path,
                "source_sha256": sha256_text(source),
                "line_count": len(source.splitlines()),
            }
        )
        body.extend(
            [
                f"\n## {title}\n\n",
                f"- 源文件：`{source_path}`\n",
                f"- SHA-256：`{sha256_text(source)}`\n\n",
                "<details>\n<summary>展开完整源码</summary>\n\n",
                f"```cpp\n{source.rstrip()}\n```\n\n",
                "</details>\n",
            ]
        )
    return "".join(body), {
        "output": "A21-kernel-cpp.md",
        "scope": "complete_files",
        "files": files,
        "symbol_count": len(CPP_SYMBOLS),
    }


def render_index(revision: str, entries: list[dict[str, object]]) -> str:
    function_total = sum(int(e.get("function_count", 0)) for e in entries)
    class_total = sum(int(e.get("class_count", 0)) for e in entries)
    rows = []
    for entry in entries:
        if "function_count" in entry:
            coverage = (
                f"{entry['function_count']} 个函数，{entry['class_count']} 个类声明"
            )
            source = f"`{entry['source']}`"
        else:
            coverage = f"{entry['symbol_count']} 个核心符号，完整内核文件"
            source = "、".join(f"`{x['source']}`" for x in entry["files"])
        rows.append(f"| [{entry['output']}]({entry['output']}) | {source} | {coverage} |")
    return "\n".join(
        [
            "<!-- Generated by tools/generate_function_appendix.py. Do not edit manually. -->",
            "# 逐函数源码附录",
            "",
            f"> 快照提交：`{revision}`。Python 附录共覆盖 **{function_total}** 个函数、**{class_total}** 个类声明。",
            "",
            "正文负责解释系统和关键路径，附录负责逐函数穷举。阅读时不要从 A01 顺序硬啃到 A21；应从正文链接跳到具体函数。",
            "",
            "| 附录 | 源文件 | 覆盖 |",
            "|---|---|---:|",
            *rows,
            "",
            "## 一致性检查",
            "",
            "```bash",
            "python3 docs/hicache_source_guide/tools/generate_function_appendix.py --check",
            "```",
            "",
            "如果源码函数、行号或内容发生变化，检查会失败。重新运行不带 `--check` 的命令即可更新附录和 `coverage.json`。",
            "",
        ]
    )


def build_outputs() -> tuple[dict[Path, str], dict[str, object]]:
    revision = git_revision()
    outputs: dict[Path, str] = {}
    entries = []
    for spec in SPECS:
        content, metadata = render_python_spec(spec, revision)
        outputs[OUTPUT_DIR / spec.output_name] = content
        entries.append(metadata)
    cpp_content, cpp_metadata = render_cpp_appendix(revision)
    outputs[OUTPUT_DIR / "A21-kernel-cpp.md"] = cpp_content
    entries.append(cpp_metadata)
    outputs[OUTPUT_DIR / "README.md"] = render_index(revision, entries)

    coverage = {
        "revision": revision,
        "generator": "docs/hicache_source_guide/tools/generate_function_appendix.py",
        "python_function_total": sum(
            int(e.get("function_count", 0)) for e in entries
        ),
        "python_class_total": sum(int(e.get("class_count", 0)) for e in entries),
        "entries": entries,
    }
    outputs[OUTPUT_DIR / "coverage.json"] = json.dumps(
        coverage, ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n"
    return outputs, coverage


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if generated appendices differ from the current source tree.",
    )
    args = parser.parse_args()

    outputs, coverage = build_outputs()
    stale = []
    for path, content in outputs.items():
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != content:
                stale.append(path.relative_to(ROOT).as_posix())
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

    if stale:
        print("Generated HiCache guide files are stale:", file=sys.stderr)
        for path in stale:
            print(f"  {path}", file=sys.stderr)
        return 1

    verb = "Verified" if args.check else "Generated"
    print(
        f"{verb} {len(outputs)} files covering "
        f"{coverage['python_function_total']} Python functions."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

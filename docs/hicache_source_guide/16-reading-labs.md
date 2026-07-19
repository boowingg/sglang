# 16. 循序渐进的源码阅读练习

这些练习按难度递增。每个练习都要求自己写出状态变化，不只是看答案。

## Lab 1：区分 token、index 与 KV 字节

### Lab 1 目标

看到任意 tensor 能判断它装的是内容、索引还是 KV。

### Lab 1 步骤

1. 打开 [A03](appendix/A03-radix-cache.md) 的 `TreeNode.__init__`。
2. 打开 [A02d](appendix/A02d-l1-memory-pools.md) 的 `ReqToTokenPool` 与 `set_kv_buffer`。
3. 打开 [A08](appendix/A08-host-pool-mha.md) 的 `init_kv_buffer`。
4. 打开 [A05](appendix/A05-cache-controller.md) 的 `CacheOperation.__init__`。
5. 画出 `req_to_token -> L1 slot`、`node.key -> node.value -> device pool` 和 `node.host_value -> host pool`。

### Lab 1 检查答案

- key 保存 token ids；
- value/host_value 保存整数 slots；
- `req_to_token` 的单元格也保存 L1 slot id，但按请求序列位置组织；
- KV bytes 位于 pool buffer；
- operation 只引用 indices，不复制整份 KV 到 Python 对象。

## Lab 2：手算一次 Radix 分裂

### Lab 2 输入

```text
已有 child.key = [10,11,12,13,14,15,16,17]
新 key          = [10,11,12,13,20,21,22,23]
page_size = 4
```

### Lab 2 任务

1. `RadixKey.match()` 返回多少？
2. 分裂后有哪些节点？
3. old child 的 value/host_value/hash 如何切？
4. parent.children 的 key 是什么？

### Lab 2 检查答案

公共前缀 4 token；创建 `[10..13]` 中间节点，旧后缀 `[14..17]` 和新后缀 `[20..23]` 成为两个 children。三个数据字段都按同一 page 边界切分。

## Lab 3：跟踪一个 L3 prefetch

### Lab 3 源码顺序

1. `Scheduler._prefetch_kvcache`
2. `HiRadixCache.prefetch_from_storage`
3. `HiCacheController.prefetch`
4. `prefetch_thread_func`
5. `_storage_hit_query`
6. `prefetch_io_aux_func`
7. `_page_transfer`
8. `HiRadixCache.check_prefetch_progress`

### Lab 3 记录表

| 时点 | host indices 谁拥有 | completed tokens | tree 是否可见 | anchor ref |
|---|---|---:|---|---:|
| alloc 后 |  |  |  |  |
| query 后 |  |  |  |  |
| 第一批 read 后 |  |  |  |  |
| Scheduler 收割后 |  |  |  |  |

### Lab 3 自问

为什么 read thread 不能直接调用 `_insert_helper_host()`？为什么未命中尾部通过 queue 释放？

## Lab 4：跟踪一个 D2H + L3 write

### Lab 4 场景

节点 1024 token，D2H 排队后被分裂成 768 + 256。

### Lab 4 任务

沿以下函数写出 pending 数据变化：

```text
write_backup
_track_write_through_node
_split_node
_replace_pending_write_through_node
writing_check
_finish_write_through_ack
_concat_split_chain
write_backup_storage
```

### Lab 4 检查点

- ACK id 为什么仍可用旧 node id？
- `publish_nodes` 为什么必须包含两个碎片？
- `backup_len` 为什么不能改成最后一个碎片长度？
- L3 写的 host indices 如何恢复为原 1024 token 顺序？

## Lab 5：计算 L2 容量

### Lab 5 参数

```text
layers=32
kv_heads=8
head_dim=128
dtype=FP16
K+V
page_size=64
hicache_size=30（源码按十进制 GB，即 30,000,000,000 bytes）
```

### Lab 5 任务

1. 计算 size per token；
2. 计算理论 token slots；
3. 按 page_size 对齐；
4. 算能保存多少个 4096-token 前缀；
5. 把 prefetch admission watermark 设为 50% 后，思考 resident/write staging 的空间，并解释为什么这不等于物理预留一半 L2。

### Lab 5 检查答案

```text
size/token = 32×8×128×2 bytes×2 = 131072 bytes
raw slots = floor(30,000,000,000 / 131072) = 228881
源码 page_num = floor(228881/64)+1 = 3577
最终 slots = 3577×64 = 228928（约 30.006 GB）
可完整保存约 55 个 4096-token 前缀（未计其他占用）
```

实际还受系统 reserve、PP 同步容量、辅助 pool 和 allocator 约束。

50% 只是在检查时阻止更多 speculative prefetch 入场；各类数据仍共享同一 allocator，单个大 operation 也可能跨过检查点。不能把剩余 50% 当成写路径不可借用、必然可用的静态分区。

## Lab 6：实现一个内存中的最小 L3 后端测试替身

### Lab 6 目标

理解接口，不用于生产。

### Lab 6 要求

- 继承 `HiCacheStorage`；
- key -> immutable bytes/tensor clone；
- `batch_exists` 返回连续前缀；
- `batch_get_v2/set_v2` 精确 per-page bool；
- 支持注入“第 N 页失败”；
- 支持并发同 key set；
- set 返回后不再读取 source buffer。

### Lab 6 测试

```text
all hit
middle miss
partial read
partial write
same-key concurrent write
source/target immediately reused
```

做完后再阅读 `HiCacheFile._batch_io_v2`，比较它如何验证 indices 长度。

## Lab 7：分析 `non-leaf`

### Lab 7 场景

```text
parent: value exists, host_value=None
child A: value=None, host_value exists
child B: value=None, host_value exists
```

### Lab 7 任务

1. `all(c.evicted for c in parent.children)` 是什么？
2. parent 是 device leaf 还是 structural leaf？
3. `_promote_parent` 是否会把它放入 heap？
4. 若走 `_evict_regular` 会发生什么？
5. 安全动作应是仅释放 parent L1、删除 subtree，还是先备份 parent？各自需要什么前提？

这个练习用于理解“同一个 leaf 词在不同层级可能有不同定义”。

## Lab 8：从指标判断并发退化

### Lab 8 数据

```text
并发 A: backend hit 75%, L2 allocation 99%, read completion 100%, effective 74%
并发 B: backend hit 69%, L2 allocation 80%, read completion 100%, effective 55%
```

### Lab 8 任务

按影响从大到小列出原因，并写出下一轮需要增加/查看的五个指标。

### Lab 8 参考方向

最大损失在 L2 admission；其次是 backend hit。查看 L2 occupancy 分解、prefetch reserved、backup-protected tokens、allocation retry/shrink、L3 write queue age。Read completion 已是 100%，不应先优化 get 错误率。

## Lab 9：理解逐层 H2D

### Lab 9 任务

读：

```text
LayerLoadingEvent
LayerDoneCounter
HiCacheController.start_loading
ScheduleBatch.prepare_for_extend
```

画出 batch N 和 batch N+1 的 producer/consumer slot。回答：

- 为什么是环形多个 slot，而不是一组 event？
- 为什么 node.value 可先发布但模型仍安全？
- 如果 consumer index 错一位，会出现什么？

## Lab 10：审核自研 L3 的 buffer 生命周期

选择后端的 set/get 各一条真实代码路径，回答：

```text
方法返回时 DMA/RDMA 是否完成？
谁持有 completion handle？
HiCache 何时 release_host？
断连/timeout 时 callback 是否还会访问 pointer？
detach 如何 cancel/drain？
```

若任一答案不明确，先解决生命周期，再做带宽优化。

## Lab 11：静态一致性检查

运行：

```bash
python3 docs/hicache_source_guide/tools/generate_function_appendix.py --check
```

再查看：

```bash
git diff --stat
git diff -- docs/hicache_source_guide
```

理解：生成附录固定了 source SHA 和 commit。源码变化后，正文中的时序结论也可能需要人工复核，不能只机械重新生成。

## Lab 12：测试分层

按环境选择，不要在无 GPU/依赖环境盲目跑全部：

```bash
pytest -q test/registered/unit/mem_cache/test_hiradix_cache_unit.py
pytest -q test/registered/unit/mem_cache/test_hiradix_pp_sync_drain.py
pytest -q test/registered/unit/mem_cache/test_unified_radix_hicache_dispatch.py
pytest -q test/registered/unit/mem_cache/test_hicache_file_lru_unit.py
```

GPU/JIT/集成测试需要相应设备、后端和多 rank 环境。每次改生命周期代码，至少新增一个能复现原交错顺序的定向测试。

## Lab 13：跟踪一次 prefill 的 L1 所有权转移

按下面顺序阅读 [A02h](appendix/A02h-request-allocation-flow.md)、[A02d](appendix/A02d-l1-memory-pools.md)、[A19](appendix/A19-request-batch-state.md) 和 [A19b](appendix/A19b-result-cache-commit.md)：

```text
ScheduleBatch.prepare_for_extend
alloc_for_extend
write_cache_indices
device pool.set_kv_buffer
SchedulerBatchResultProcessor.process_batch_result_prefill
cache_unfinished_req / cache_finished_req
```

画一张表，逐步记录 `req_pool_idx`、`prefix_indices`、`out_cache_loc`、`kv_allocated_len`、`kv_committed_len`、`cache_protected_len` 和 Tree `lock_ref`。重点回答：哪些 slots 只是预分配，哪些已写 KV，哪些已提交给 Tree，结束请求时哪一段必须释放。

## Lab 14：验证本地 namespace 与 L3 hash 不相同

构造 token ids 相同、`extra_key` 不同的两个 `RadixKey`，分别检查：

1. `_check_compatible()`/本地 match 是否允许共享；
2. `child_key(page_size)` 是否相同；
3. 相同 `prior_hash` 下 `hash_page()` 是否相同；
4. 自研后端最终 key 的 namespace 从哪里来。

预期：本地 child key 不同、两个 key 不兼容，但当前原生页 hash 相同。这个实验直接证明 adapter/salt 隔离不能只依赖本地 `extra_key`；生产验证应再检查 backend key 是否编码 model/rank/layout/version 和需要的动态语义。

## 完成标准

完成这些练习后，应能不看书回答：

1. 请求从 L3 hit 到 GPU 真正可用经过哪些状态？
2. 为什么 L2 容量小会让 L3 hit 失效？
3. 为什么 write ACK 不等于 L3 全部成功？
4. pending D2H 期间节点分裂如何提交？
5. 哪些引用保护 L1、L2 和索引 tensor？
6. 为什么多 rank 频繁使用 MIN？
7. 自研 L3 返回 True 需要承诺什么？
8. 高并发 cache-hit 下降应怎样按 token 漏斗定位？
9. 请求位置、L1 slot、Radix `value` 和真实 KV 字节如何关联？
10. 为什么本地 `extra_key` 不等同于 L3 namespace？

返回[全书目录](README.md)或进入[逐函数源码附录](appendix/README.md)。

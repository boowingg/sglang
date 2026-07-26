# HiCache 读路径 L1+L2 真实叶子回收

> 状态：V1 已实现，待真实负载验证
>
> 代码基线：`main` / `f1c247edf997e932ad6fa35d479e98b41a18ce6e`
>
> 开发分支：`hicache-l2-leaf-reclaim-design`
>
> 主要实现：
> [`HiRadixCache`](../../python/sglang/srt/mem_cache/hiradix_cache.py)

## 1. V1 范围

这一版只解决一个问题：

> L3 预取需要申请 L2 页面时，如果原有 L2-only 回收仍不能提供足够空间，
> 允许继续删除冷的 L1+L2 真实叶子，为 L3 到 L2 的读取腾出空间。

明确包含：

- 保留原有 L2-only 回收为第一优先级。
- 第一阶段完成后重新读取 L2 allocator 的可用容量。
- 仍有缺口时，从冷到热扫描 L1+L2 候选。
- 只删除 RadixTree 的真实拓扑叶子。
- 在一个 Scheduler 操作内释放 L1、释放 L2并解除整个节点。
- 清理 Session 索引、候选集合、容量会计和 KV cache events。
- 保护正在被请求、D2H、H2D 或 L3 异步操作使用的节点。
- 第一版通过 capability gate 限制在能够安全工作的部署形态。

明确不包含：

- 不实现 query-first。
- 不改变 L3 查询、读取和超时状态机。
- 不改变 `write_backup()` 的分配或回收策略。
- 不给 `TreeNode` 增加永久 L3 驻留状态。
- 不增加 Restore Arena、旁路传输 buffer 或流式恢复通道。
- 不修改 UnifiedRadixCache、HiMambaRadixCache 或 Hybrid cache。
- 不增加新的命令行参数和统计指标。

因此，这是一版刻意收敛的改动。它验证“放宽读路径 L2 回收是否能提高 L3
有效读入量”，不同时承担预取状态机重构。

## 2. 要解决的问题

HiCache 的本地层级可以简化为：

```text
L1: GPU KV
L2: Host KV
L3: 外部或分布式存储
```

节点通过两个字段表达本地状态：

| `node.value` | `node.host_value` | 状态 |
| --- | --- | --- |
| 有 | 无 | L1-only |
| 有 | 有 | L1+L2 |
| 无 | 有 | L2-only |
| 无 | 无 | 不应长期留在树中的空节点 |

原来的 Host 回收只处理 `evictable_host_leaves`。这个集合主要包含已经没有 L1、
但仍持有 L2 的节点，也就是 L2-only 候选。

高并发时可能出现：

```text
L2 被大量 L1+L2 重复副本占据
-> L3 预取申请 L2 失败
-> L2-only 候选很少
-> 原有 evict_host() 释放不出足够空间
-> 预取被截短或完全放弃
-> 已经存在于 L3 的 KV 无法进入本地缓存
-> 请求重算，端到端 cache hit 下降
```

V1 的核心变化是：

```text
L2-only 不足
-> 允许牺牲冷的 L1+L2 真实叶子
-> 同时得到 L1 和 L2 空间
-> L3 预取重新申请完整 L2
```

## 3. 请求读路径

当前预取仍然是 eager allocation，顺序没有改成 query-first：

```text
1. Scheduler 构造完整且页对齐的 prefetch_key
2. 检查 storage、阈值和预取限流
3. protect_host(last_host_node)
4. 尝试按完整 prefetch_length 申请 L2
5. 申请失败，进入两级 Host 回收
6. 再次申请完整 L2
7. 仍失败，按当前剩余 L2 尝试部分预取
8. 创建并提交 L3 prefetch operation
9. 后台查询 L3 并读取连续命中页
10. Scheduler 将实际完成部分插入 Host RadixTree
11. 后续 load-back 将命中 KV 从 L2 传到 L1
```

第 5 步是本分支唯一改变的业务阶段。

### 3.1 两级回收顺序

[`HiRadixCache.evict_host()`](../../python/sglang/srt/mem_cache/hiradix_cache.py)
新增关键字参数：

```python
evict_host(num_tokens, *, allow_l1_l2=False)
```

执行流程：

```text
第一阶段
    完整执行原有 evictable_host_leaves 回收
    只释放 L2-only 节点

容量复核
    available_size = mem_pool_host.available_size()
    shortage = max(0, requested_tokens - available_size)

第二阶段
    仅当 allow_l1_l2=True 且 shortage > 0
    扫描 L1+L2 真实叶子
    只回收到覆盖 shortage
```

容量复核非常重要。不能简单使用：

```text
requested_tokens - first_stage_evicted_tokens
```

因为 L2 在回收前可能已经有一部分空闲页。例如需要 4096 token，原本空闲
1024，第一阶段又释放 3072，此时已经可以完整分配，不应再删除任何 L1。

### 3.2 调用边界

只有
[`prefetch_from_storage()`](../../python/sglang/srt/mem_cache/hiradix_cache.py)
调用：

```python
self.evict_host(prefetch_length, allow_l1_l2=True)
```

[`write_backup()`](../../python/sglang/srt/mem_cache/hiradix_cache.py)
仍调用：

```python
self.evict_host(len(node.value))
```

默认参数是 `False`，所以 D2H 写入路径保持原来的 L2-only 策略。这一调用边界有
专门单元测试，防止后续重构时误把激进回收扩散到写路径。

## 4. 为什么必须是真实叶子

`evictable_leaves` 表示的是 L1 视角的叶子：

```text
节点有 L1
节点没有 lock_ref
没有仍持有 L1 的子节点
```

它不保证：

```text
len(node.children) == 0
```

一个节点可能有 L1，同时下面挂着 L2-only 子节点。此时它在设备视角是叶子，
但在 RadixTree 拓扑上仍是内部节点。

如果直接解除这种节点：

- 子树会被整体孤立。
- parent/child 索引失去一致性。
- 后续 match、split、insert 或 Session 回收可能访问脱离树的节点。
- 容易重现 `AssertionError: non-leaf`、child key 不一致或长度断言。

因此第二阶段统一使用
`_is_reclaimable_l1_l2_leaf()`，并强制：

```text
node is not root
node.parent is not None
len(node.key) > 0
len(node.children) == 0
node.value is not None
node.host_value is not None
node.lock_ref == 0
node.host_ref_counter == 0
node.write_through_pending_id is None
parent.children[child_key] is node
len(value) == len(key) == len(host_value)
```

候选来自现有 `evictable_leaves`，但每次从 heap 取出后都重新执行完整判断。
过期、已解除或状态变化的候选只会被跳过，不使用断言推动生产控制流。

## 5. 异步操作保护

不需要知道 L3 是否永久存在，但必须知道本地页面现在是否仍被使用。

### 5.1 运行请求和 H2D/D2H

`lock_ref > 0` 表示设备前缀正在被请求或传输生命周期持有。

- 活跃请求通过 RadixTree lock 保护路径。
- 普通 write-through 在 D2H 完成前持有 lock。
- `load_back()` 在 H2D 完成前持有 lock。

因此第二阶段遇到 `lock_ref > 0` 必须跳过。

### 5.2 L3 预取锚点和 L3 写入

`host_ref_counter > 0` 表示 Host 页面正被异步存储操作使用。

- `prefetch_from_storage()` 保护 `last_host_node`。
- `write_backup_storage()` 在 L3 写完成 ACK 前保护来源 Host 页面。

因此第二阶段遇到 `host_ref_counter > 0` 必须跳过，避免 use-after-free。

### 5.3 未完成 write-through

`write_through_pending_id is not None` 表示 Host 页面已经分配，但 D2H 写入还没
完成发布。即使某种特殊流程暂时没有 lock，也不能回收该节点。

现有三个字段共同覆盖这版所需的瞬时所有权：

```text
lock_ref
host_ref_counter
write_through_pending_id
```

V1 不扫描 ongoing map 的全部内容，因为：

- 正常生命周期已经把对应引用投影到上述字段。
- 每次 Host 压力下遍历多个 operation map 会增加 Scheduler 热路径成本。
- `write_through_pending_id` 又为特殊 D2H 窗口提供了显式兜底。

## 6. 原子整叶删除

`_evict_l1_l2_leaf()` 在 Scheduler 线程中完成一个不可分割的逻辑操作：

```text
1. 重新校验节点资格
2. 保存 parent、device_indices、host_indices
3. 节点仍连接在树上时记录 GPU remove event
4. 记录 CPU remove event
5. 释放 L1 物理页面
6. 释放 L2 物理页面
7. 从 Host 候选集合删除
8. 调用 _delete_leaf()
9. 清空 value、host_value 和 parent
10. 更新父节点 Host leaf 状态
```

基础 `_delete_leaf()` 可以安全用于这里，因为节点删除前仍有 L1。它会：

- 从 `parent.children` 解除节点并验证身份。
- 调用 `_discard_session_leaf()`。
- 从 `evictable_size_` 扣减一次设备 token。
- 从 `evictable_leaves` 删除节点。
- 更新父节点的设备叶子状态。

不能对已经是 L2-only 的节点调用 `_delete_leaf()`，因为其 L1 会计早已在原设备
淘汰时扣减，再调用会造成二次扣减。第一阶段因此继续保留原来的 Host-only
删除逻辑。

### 6.1 父节点递进

删除一个 L1+L2 真实叶子后，父节点可能变成新的 L1+L2 真实叶子。

第二阶段会重新校验父节点；满足条件时，将它按同一 eviction strategy 的优先级
放回 heap。这样可以自底向上递进回收，但绝不会跨过仍有 child 的节点。

例子：

```text
root
  -> A(L1+L2)
       -> B(L1+L2, actual leaf)

删除 B
-> A 成为 actual leaf
-> 若仍有 shortage，再删除 A
```

如果父节点是 L1-only、仍有 child、被引用或正在传输，它不会进入第二阶段。

## 7. 淘汰优先级和性能

第二阶段只在以下条件全部成立时构建 heap：

```text
完整 L2 分配失败
原有 L2-only 回收已经执行
实时 L2 available_size 仍小于本次申请
部署形态通过 capability gate
```

正常分配成功时没有额外扫描。

候选 heap 继续复用当前 eviction strategy：

- LRU
- LFU
- FIFO
- MRU
- FILO
- Priority
- SLRU

因此不会引入第二套冷热策略。节点是回收粒度，单个节点可能大于 shortage，所以
允许少量 over-eviction；达到或超过缺口后立即停止。

新增热路径成本仅出现在原本已经发生 L2 分配失败的场景：

```text
O(number of current evictable L1 leaves) 建 heap
+ O(number of reclaimed nodes * log N)
```

没有逐 token 日志，没有长期新增候选集合，也没有后台统计锁。

## 8. Capability Gate

整叶删除会改变 L1 命中、GPU allocator 和 RadixTree 拓扑。多 rank 如果删除
不同节点，可能出现：

- 各 rank 匹配到不同前缀长度。
- RadixTree 拓扑分叉。
- 后续 prefill shape 或 collective 次序不一致。
- 最坏情况下错误结果或分布式死锁。

现有 Host 可用空间、写入 ACK 和候选状态在 rank 间可能短暂不一致。V1 没有
引入候选广播和全局可删除位图，因此不能在这些模式下冒险启用。

`_supports_l1_l2_read_reclaim()` 将第二阶段限制为：

```text
tp_world_size == 1
attn_cp_group world size == 1
attn_tp_group world size == 1
pp_size == 1
普通 MHATokenToKVPool，也就是 MHA/GQA
非 HybridCacheController
未启用 draft KV
```

不满足时：

- 第一阶段 L2-only 回收照常执行。
- 第二阶段跳过。
- 首次遇到真实 shortage 时打印一次 warning。
- 后续请求不重复打印，避免日志压力。

这不是性能限制，而是正确性边界。后续若要支持 TP/CP，需要设计：

```text
所有 rank 一致进入回收
-> 选择稳定候选描述
-> 各 rank 重新验证
-> 全局 MIN 得到可删除位图
-> 所有 rank 按相同顺序删除
-> 全局复核 Host 容量
```

PP、Hybrid、Unified、Mamba/SWA 和 draft 还需要各自处理辅助 pool 与组件状态，
不能仅删除基础 KV 页面后宣称支持。

## 9. 不依赖 L3 状态

整叶回收不检查“L3 已确认存在”。

原因是 L3 状态影响复用性能，不影响模型正确性：

- L3 有数据，后续重新读取。
- L3 没数据，后续重新计算。

只要没有读取已经释放的本地页面，两种情况都会得到正确 KV。

给 `TreeNode` 增加永久 `l3_present` 反而会引入复杂一致性问题：

- L3 写入可能部分成功。
- L3 后端可能独立淘汰。
- 多 rank 的写入结果可能不同。
- 状态需要随 split、merge、删除和重建传播。

所以 V1 只依赖本地瞬时所有权，不维护远端驻留真相。

## 10. V1 的已知代价

因为这一版没有 query-first，仍然是：

```text
先申请 L2并可能回收 L1+L2
-> 后台查询 L3
```

所以一个最终 L3 miss 的请求也可能已经牺牲冷 L1。这不会导致错误，但可能造成：

- L1 命中下降。
- 后续请求从 L3 重新读取。
- L3 也 miss 时增加重算。
- L2 持续满载时出现删除和重新写入抖动。

这项风险是用户明确接受的 V1 边界，也是测试时必须重点观察的内容。建议至少
对比：

```text
L3 实际读取 token
L3 miss token
L2 完整与部分预取次数
GPU cache hit
Host cache hit
端到端 cache hit
重算 token
TTFT、吞吐和 P99
```

如果 L3 有较高连续命中率，第二阶段通常能把原本进不来的 KV 拉入本地；如果
L3 miss 很高，牺牲 L1 的代价可能超过收益。

## 11. 测试覆盖

新增 CPU 单元测试：

[`test_hiradix_l1_l2_leaf_reclaim.py`](../../test/registered/unit/mem_cache/test_hiradix_l1_l2_leaf_reclaim.py)

覆盖：

1. 默认 `evict_host()` 不删除 L1+L2。
2. L2-only 始终先于 L1+L2 回收。
3. 原有空闲容量加第一阶段释放已经足够时，不额外牺牲 L1。
4. 拥有 topology child 的设备叶子不能整节点删除。
5. `host_ref_counter`、`lock_ref` 或 pending write-through 会阻止删除。
6. 删除子节点后，父节点才能递进成为候选。
7. Session leaf 索引随整叶删除清理。
8. 已从父节点脱离的 stale candidate 被安全跳过。
9. 多 rank 场景自动保持原 L2-only 策略。
10. 预取路径显式开启第二阶段。
11. 写入重试继续使用默认 L2-only 策略。
12. L1+L2 删除同时产生 GPU、CPU remove event。
13. 设备可淘汰容量只扣减一次。

真实环境还应补充：

- TP=1、PP=1 的 Qwen MHA/GQA 压测。
- L2 很小、L1 较大时持续制造 L2 pressure。
- 并发 32、64 和更高并发对比。
- 慢 L3、L3 miss 和部分命中故障注入。
- D2H、H2D、L3 backup 长时间未 ACK 时触发回收。
- Session cache 和 KV cache events 分别开启。
- allocator used/free 总量长期守恒。

## 12. 后续 query-first

query-first 是独立的下一阶段，不属于 V1。

目标顺序是：

```text
1. 后台查询 L3 连续命中长度
2. 多 rank 对实际 hit token 取 MIN
3. 只为确认命中的部分计算 L2 shortage
4. 执行两级回收
5. 一次性申请目标 L2
6. 启动实际 L3 read
```

它的主要收益是：

- L3 miss 不申请 L2。
- 短命中不按完整候选长度占用 L2。
- 只有已经确认有价值的数据才允许牺牲 L1。

但实现它需要显式拆分 PrefetchOperation 状态：

```text
QUERYING
WAITING_L2
READING
COMPLETED
CANCELED
FAILED
```

还要处理 query result queue、多 rank 同序推进、查询取消、查询超时、read
timeout 和 L3 在 query/read 之间再次淘汰等问题。它的改动量和耦合度明显高于
本分支，因此应单独开发和评审。

## 13. 最终不变量

V1 运行期间必须始终满足：

1. 只有 `len(node.children) == 0` 的 L1+L2 节点能被整节点删除。
2. root 永远不能进入整叶回收。
3. detached 或 parent-child identity 不匹配的候选只能跳过。
4. `lock_ref > 0` 的节点不能删除。
5. `host_ref_counter > 0` 的节点不能删除。
6. `write_through_pending_id` 未清除的节点不能删除。
7. L1 和 L2 必须在同一 Scheduler 操作内释放。
8. L1+L2 设备会计只扣减一次。
9. Session 索引不能指向已经解除的节点。
10. GPU 和 CPU remove event 必须在节点脱离父树前生成。
11. 第二阶段不能先于原有 L2-only 回收。
12. 实时可用容量已经满足申请时不能牺牲 L1。
13. 写路径不能通过默认调用进入第二阶段。
14. 不支持的并行或复合缓存形态必须安全保持 L2-only。
15. L3 是否存在只影响后续命中或重算，不参与本地释放正确性判断。

满足这些不变量后，这版改动不会制造 L1/L2 空洞，也不会把内部节点误删。它的
主要待验证项不是模型正确性，而是 eager 预取下“增加 L3 可读入量”和“牺牲冷
L1”之间的实际收益。

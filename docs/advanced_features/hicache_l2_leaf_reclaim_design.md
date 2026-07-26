# HiCache L2 真实叶子扩展回收方案

> 状态：设计评审完成，尚未实现
>
> 代码基线：`main` / `f1c247edf997e932ad6fa35d479e98b41a18ce6e`
>
> 设计分支：`codex/hicache-l2-leaf-reclaim-design`

## 1. 范围

本文只讨论 **放宽 HiCache L2 淘汰条件**：

- 保留“从叶子开始删除”的原则。
- 当前只能删除 `L2-only` 叶子。
- 新方案在 `L2-only` 叶子不足时，继续选择冷的 `L1+L2` 真实叶子。
- 对 `L1+L2` 真实叶子，在一次原子操作中释放 L1、释放 L2并解除整个节点。
- 不允许在树中留下 L1 或 L2 空洞。

本文不包含 L3 到 L1 的额外传输通道、临时传输 buffer 或流式恢复方案。这些是
独立方案，应在其他分支设计和实现。

## 2. 评审结论

方案可行，没有无法规避的架构级问题，但不能直接把当前
`evictable_leaves` 中的节点拿来删除。正确实现必须满足以下条件：

1. 被解除的节点必须是 **RadixTree 真实拓扑叶子**：
   `len(node.children) == 0`。
2. 节点不能被运行请求或异步 D2H、H2D、L3 读写操作持有。
3. L1、L2、树拓扑、Session 索引、候选集合和容量会计必须在同一个
   Scheduler 线程操作中一起更新。
4. 多 TP/CP rank 必须删除相同节点；第一版不应在 PP 多阶段下直接启用。
5. 第一版建议只让 **L2 写入分配压力** 触发 `L1+L2` 整叶回收。
   当前预取路径在确认 L3 hit 前就申请 L2，若预取也能牺牲 L1，会产生严重抖动。
6. 不需要增加永久的“L3 是否存在”节点状态。删除后 L3 miss 时重新计算即可。

按这些约束实现后，整叶回收不会破坏模型计算正确性。它改变的是缓存保留策略：
用一个较冷的 `L1+L2` 副本换取新的 L2 写入空间。

## 3. 当前代码行为

### 3.1 TreeNode 的 L1/L2 状态

[`TreeNode`](../../python/sglang/srt/mem_cache/radix_cache.py) 通过两个字段表示
本地缓存状态：

| `value` | `host_value` | 状态 |
| --- | --- | --- |
| 有 | 无 | L1-only |
| 有 | 有 | L1+L2 |
| 无 | 有 | L2-only |
| 无 | 无 | 不应作为有效缓存节点长期保留 |

其中：

- `node.evicted` 等价于 `node.value is None`。
- `node.backuped` 等价于 `node.host_value is not None`。
- `lock_ref` 保护正在被请求或 D2H/H2D 流程使用的设备前缀。
- `host_ref_counter` 保护正在被 L3 读写等异步操作使用的 Host 页面。
- `write_through_pending_id` 表示 D2H write-through 尚未完成。

### 3.2 当前 L2 淘汰候选

[`HiRadixCache._update_host_leaf_status`](../../python/sglang/srt/mem_cache/hiradix_cache.py)
只会将 L2-only 节点加入 `evictable_host_leaves`：

```text
node.evicted == True
node.lock_ref == 0
没有仍然 backuped 的子节点
```

[`HiRadixCache.evict_host`](../../python/sglang/srt/mem_cache/hiradix_cache.py)
从该集合构建 heap，释放 `host_value`，再把节点从父节点的 `children` 中解除。

因此，当前 L2 满时，即使大量真实叶子同时拥有 L1 和 L2，`evict_host()` 也不能
使用这些空间。L1 较大、并发写入较多时，L2 很容易被重复副本占满。

### 3.3 当前 L1 淘汰候选不是“真实叶子”

[`RadixCache._update_leaf_status`](../../python/sglang/srt/mem_cache/radix_cache.py)
对 `evictable_leaves` 的定义是：

```text
节点有 L1
节点没有 lock_ref
没有仍有 L1 的子节点
```

它表达的是 **L1 视角的叶子**。节点仍可能拥有一个或多个 L2-only 子节点。
所以：

```text
node in evictable_leaves
```

不能推出：

```text
len(node.children) == 0
```

这正是过去 `AssertionError: non-leaf, node.id=...` 一类问题的根源。新方案若
直接遍历 `evictable_leaves` 并解除节点，会把仍有子树的节点当作真实叶子删除。

## 4. 为什么可以不依赖 L3 状态

缓存层的数据不是模型输出的唯一事实来源。一个 L1+L2 真实叶子被完整删除后：

- 如果 L3 仍有该前缀，后续可以从 L3 恢复。
- 如果 L3 没有该前缀，后续重新计算。
- 两种情况都不会得到错误 KV，只会影响命中率和计算量。

因此，**“L3 已确认存在”不是整叶回收的正确性条件**，也不需要给 `TreeNode`
增加永久的 L3 状态。

但必须区分下面两个概念：

1. **L3 是否存在**
   - 不需要知道。
   - 影响的是未来复用性能。

2. **Host 页面是否正在被异步操作使用**
   - 必须知道。
   - 决定现在释放页面是否会产生 use-after-free。

第二点由现有 `lock_ref`、`host_ref_counter`、`write_through_pending_id` 和 ongoing
operation map 表达。它是瞬时资源所有权，不是 L3 驻留状态。

## 5. 最终淘汰策略

### 5.1 两级回收

每次 L2 空间不足时按固定顺序执行：

```text
第一级：回收 L2-only 真实叶子
    不牺牲任何 L1 命中

第二级：回收 L1+L2 真实叶子
    只有第一级空间不足时执行
    同时释放 L1 和 L2，并删除整个节点
```

无论哪一级，都不删除内部节点，不制造 L2 空洞。

### 5.2 按分配原因控制第二级回收

建议给 `evict_host()` 增加内部原因，而不是让所有调用点无条件使用激进策略：

```python
class HostReclaimReason(Enum):
    WRITE_STAGING = "write_staging"
    STORAGE_PREFETCH = "storage_prefetch"
```

第一版行为：

| 原因 | L2-only 真实叶子 | L1+L2 真实叶子 |
| --- | --- | --- |
| `WRITE_STAGING` | 允许 | 允许 |
| `STORAGE_PREFETCH` | 允许 | 默认禁止 |

原因在于当前
[`prefetch_from_storage`](../../python/sglang/srt/mem_cache/hiradix_cache.py)
的顺序是：

1. 先申请完整 L2。
2. 申请失败时先淘汰 L2。
3. L2 分配成功后才由后台线程查询 L3。

如果预取可以触发第二级回收，那么一个最终 L3 miss 的请求也会删除已有的
L1+L2 热数据。这会造成：

- L1 命中率下降；
- L2 写入和预取互相驱逐；
- 高并发下反复删除、重新计算、重新写入；
- 释放容量增加但有效吞吐反而下降。

只有未来把预取改成“先确认 L3 hit，再决定是否牺牲 L1”后，才适合单独评审
`STORAGE_PREFETCH` 的第二级回收。该改动不属于本文范围。

### 5.3 对现有调用点的建议

[`HiRadixCache.write_backup`](../../python/sglang/srt/mem_cache/hiradix_cache.py)：

```python
evict_host(required_tokens, reason=WRITE_STAGING)
```

[`HiRadixCache.prefetch_from_storage`](../../python/sglang/srt/mem_cache/hiradix_cache.py)：

```python
evict_host(required_tokens, reason=STORAGE_PREFETCH)
```

默认策略保持可配置，功能关闭时必须与当前 `main` 行为一致。

## 6. 候选节点定义

### 6.1 L2-only 真实叶子

必须同时满足：

```text
node is not root
node.parent is not None
len(node.children) == 0
node.value is None
node.host_value is not None
node.lock_ref == 0
node.host_ref_counter == 0
node.write_through_pending_id is None
node 不属于任何 ongoing D2H/H2D/L3 operation
父节点 children 中仍以正确 child_key 指向 node
```

当前 `_update_host_leaf_status()` 建议改为只维护这种真实叶子：

```python
def _is_l2_only_actual_leaf(node):
    return (
        node is not self.root_node
        and node.parent is not None
        and len(node.children) == 0
        and node.evicted
        and node.backuped
        and node.lock_ref == 0
    )
```

`host_ref_counter` 不建议直接放进长期集合条件，因为 `protect_host()` 和
`release_host()` 当前不会触发 leaf set 更新。它应在 heap pop 时再次校验。

### 6.2 L1+L2 真实叶子

必须同时满足：

```text
node is not root
node.parent is not None
len(node.children) == 0
node.value is not None
node.host_value is not None
node.lock_ref == 0
node.host_ref_counter == 0
node.write_through_pending_id is None
node 不属于 ongoing_write_through
node 不属于 ongoing_load_back
node 不属于 ongoing_backup
父节点 children 中仍以正确 child_key 指向 node
```

候选可以从现有 `evictable_leaves` 中筛选。真实的、未加锁的 L1+L2 拓扑叶子
一定属于该集合；但集合中的其他节点不能直接使用。

不建议再维护一个长期的 `evictable_l1_l2_leaves` 集合。节点 split、insert、
lock/unlock、D2H ACK、H2D ACK 时都需要维护它，容易再次产生过期候选。第二级
回收只在第一级不足时触发，届时从 `evictable_leaves` 构建一次过滤 heap 即可。

### 6.3 每次 pop 都重新校验

即使候选来自集合，真正删除前仍必须调用统一 predicate。不要依赖集合状态，
也不要用断言代替生产路径校验。

建议行为：

```text
候选已失效 -> 跳过并记录 reject reason
候选满足全部条件 -> 进入原子删除 helper
```

原子 helper 内部可以保留 assert，用于捕获 predicate 与实现之间的编程错误。

## 7. 原子整叶删除

### 7.1 为什么不能分两次公开淘汰

直观写法是：

1. 调用现有 `_evict_backuped(node)`，形成 L2-only 节点。
2. 再调用现有 `evict_host()` 删除它。

不建议这样实现。两个步骤会分别修改：

- `evictable_size_`；
- device leaf set；
- host leaf set；
- 父节点 leaf 状态；
- KV remove events；
- 树拓扑。

如果中间状态被其他调度逻辑观察，或者第二步因引用变化而跳过，就会留下本来
不希望出现的状态。更稳妥的方式是在 Scheduler 线程中先完成全部资格检查，
再用一个 helper 完成全部更新。

“先淘汰 L1，再淘汰 L2”仍然成立，但它是一个不可分割操作内部的释放顺序，
不是两个独立调度阶段。

### 7.2 建议 helper

```python
def _reclaim_actual_leaf(node, *, reclaim_device: bool) -> int:
    # 1. 删除前完成所有可能失败的资格验证
    assert _is_attached_actual_leaf(node)
    assert node.host_value is not None
    assert node.lock_ref == 0
    assert node.host_ref_counter == 0
    assert node.write_through_pending_id is None
    assert reclaim_device == (node.value is not None)

    parent = node.parent
    device_indices = node.value
    host_indices = node.host_value
    token_count = len(node.key)

    # 2. 节点仍挂在树上时产生事件，确保 parent/hash 上下文有效
    if reclaim_device:
        _record_remove_event(node, medium=GPU)
    _record_remove_event(node, medium=CPU)

    # 3. 释放物理页面
    if reclaim_device:
        cache_controller.evict_device(device_indices)
    cache_controller.evict_host(host_indices)

    # 4. 清理所有外部索引和候选集合
    _discard_session_leaf(node)
    evictable_leaves.discard(node)
    evictable_host_leaves.discard(node)

    # 5. 解除真实叶子
    removed = parent.children.pop(node.key.child_key(page_size))
    assert removed is node
    node.value = None
    node.host_value = None

    # 6. 只有原来拥有 L1 时才扣减设备可淘汰会计
    if reclaim_device:
        evictable_size_ -= token_count

    # 7. 父节点可能成为新的实际叶子
    _update_leaf_status(parent)
    _update_host_leaf_status(parent)
    node.parent = None

    return len(host_indices)
```

实际代码还应验证 `parent.children[...] is node`，并保证释放的 device/host
索引长度与 `node.key` 一致。

### 7.3 为什么不能直接对 L2-only 节点调用 `_delete_leaf()`

[`RadixCache._delete_leaf`](../../python/sglang/srt/mem_cache/radix_cache.py)
会无条件执行：

```python
self.evictable_size_ -= len(node.key)
```

L2-only 节点的 L1 在此前已经释放，设备会计也已经扣减。再次调用会导致
`evictable_size_` 变小甚至变成负数。

因此，新 helper 必须通过 `reclaim_device` 或 `had_device` 区分：

- L2-only 删除：不再修改设备会计。
- L1+L2 删除：扣减一次设备会计。

### 7.4 Session 与事件不能遗漏

当前基础 `_delete_leaf()` 会调用 `_discard_session_leaf(node)`，但现有
`evict_host()` 是手工解除 Host 节点。新 helper 必须显式清理 Session 索引，
否则 `release_radix_session()` 可能持有已经从树上删除的节点。

启用 KV cache events 时：

- L1+L2 整叶删除应产生一次 GPU remove 和一次 CPU remove。
- L2-only 删除只产生 CPU remove。
- 事件必须在解除父子关系前产生。

## 8. 父节点晋升

每删除一个真实叶子后，父节点可能成为新候选：

| 父节点状态 | 行为 |
| --- | --- |
| L2-only 且无 children | 加入第一级 heap |
| L1+L2 且无 children | 第一级完成后可加入第二级 heap |
| L1-only | 不释放 L2，无需加入 Host 回收 heap |
| 仍有任意 child | 不是整节点候选 |
| 有 lock/operation 引用 | 暂时跳过 |

推进过程必须是自底向上的。任何时候都不能因为“子节点没有 L1”就把仍有
children 的父节点当作真实叶子。

为避免一次 Host 分配在 Scheduler 线程运行过久，建议同时设置：

- 本次所需 Host token 数；
- 单次最多删除节点数；
- 可选的单次最多牺牲 L1 token 数。

达到任一限制就停止本轮回收。

## 9. 异步操作所有权

### 9.1 D2H write-through

`write_backup()` 分配 `host_value` 后会设置 `write_through_pending_id`，并在普通
write-through 模式下增加 `lock_ref`。D2H 完成前，节点不得进入整叶回收。

删除资格至少检查：

```text
lock_ref == 0
write_through_pending_id is None
node 不在 ongoing_write_through 中
```

### 9.2 L2 到 L1 load-back

`load_back()` 在 H2D 期间通过 `lock_ref` 保护目标链。即使 Host 引用已经释放，
`lock_ref > 0` 也必须让整叶候选跳过。

### 9.3 L3 预取

尚未插入树的预取页面不在 leaf 集合中。预取锚点通过 `host_ref_counter` 保护，
因此不能删除。

### 9.4 L3 写入

`write_backup_storage()` 会对节点调用 `protect_host()`，完整操作结束后由
`ack_backup_queue` 释放。只要 `host_ref_counter > 0`，Host 页面就不能回收。

当前 `main` 还有一个独立风险：

- `_page_backup()` 某个 batch 失败后会提前结束；
- `backup_thread_func()` 仍然发送 ACK；
- `_drain_storage_control_queues_impl()` 不判断期望长度是否全部完成，就释放
  Host 引用。

这不会导致模型计算错误，因为后续可以重算，但它不满足“写入 L3 必须最终成功”
的服务目标。扩展回收会让失败写对应页面更快成为候选，从而减少人工重试机会。

处理原则：

- 本方案不增加永久 L3 状态，也不把“L3 存在”作为删除条件。
- 如果部署要求 L3 写入保证，上线第二级回收前，应单独修正写 ACK/重试生命周期：
  部分失败继续持有 Host 页面，完整成功后才释放。
- 该修正属于 L3 写可靠性，不应与树节点增加 L3 状态混为一谈。

## 10. 多 rank 一致性

### 10.1 风险

Host 可用容量、异步 ACK 到达时间和候选集合可能在 TP/CP rank 间短暂不同。
如果某些 rank 删除 L1+L2 节点、另一些 rank 保留：

- `match_prefix()` 可能得到不同前缀长度；
- 各 rank 的 RadixTree 拓扑可能分叉；
- 后续 H2D、prefill shape 或 collective 次序可能不一致；
- 最坏情况下出现错误结果或分布式死锁。

现有 L2-only 路径已经依赖较强的执行对称性。第二级回收会同时改变 L1 和树
拓扑，不能继续只依赖这种隐含假设。

### 10.2 建议同步流程

在以下同步流程完成之前，`whole-leaf-write` 必须在 TP/CP world size 大于 1 时
启动失败或自动保持 `l2-only`，不能让每个 rank 独立删除。

第一版至少在 `WRITE_STAGING` 路径执行以下步骤：

1. 先保证所有 rank 对“当前节点是否需要 backup”做出一致的入口决策。不能在
   部分 rank 跳过 `write_backup()` 时，让其他 rank 独自进入 collective。
2. 所有 rank 在调用 `cache_controller.write()` 前检查本地 Host 可用 token。
3. 用 collective 得到全局最小可用量。
4. 任一 rank 空间不足时，所有 rank 都进入同一轮回收。
5. rank 0 从本地候选 heap 选择一批候选描述：

```text
(node_id, parent_id, key_length, stable_key_digest)
```

6. 将小批候选描述广播给其他 rank。
7. 每个 rank 根据 node id 查找并重新验证真实叶子、引用和 key digest。
8. 用一次批量 `MIN` collective 得到各候选的全局可删除位图。
9. 所有 rank 按相同顺序删除全局可删除候选。
10. 再次检查全局最小 Host 可用量；足够后所有 rank 才提交 D2H write。

不要让部分 rank 先成功分配并启动 D2H，再让其他 rank 进入整叶回收。当前
`cache_controller.write()` 同时完成 Host 分配和 D2H 入队，因此应先使用
`mem_pool_host.available_size()` 做全局预检。

候选选择和验证按批次 collective，不能每删一个 token 做一次 collective。

### 10.3 PP 与复杂缓存

第一版建议增加 capability gate：

- 支持普通 MHA/GQA `HiRadixCache`。
- 支持经过上述一致性设计的 TP/CP。
- `pp_size > 1` 默认关闭。
- `HybridCacheController`、DSA/MSA、UnifiedRadixCache、Mamba/SWA 默认关闭。
- 启用 draft KV 时默认关闭。

这些模式存在额外 pool、组件级引用或跨 PP 同步，直接只释放基础 KV pool 可能
泄漏辅助页面或让不同组件状态分叉。应分别设计，不能复用第一版 helper 后直接
宣称支持。

## 11. 性能保护

### 11.1 第二级只在必要时执行

执行顺序必须是：

```text
尝试分配
-> L2-only 真实叶子回收
-> 再次检查容量
-> 仍不足才扫描 L1+L2 真实叶子
-> 只回收到本次所需容量
```

正常 Host 分配成功时，不构建第二级 heap，不扫描 `evictable_leaves`。

### 11.2 避免过度牺牲 L1

整叶是节点粒度，单个节点可能远大于本次申请量。建议记录并限制：

- `l1_tokens_sacrificed`；
- `l2_tokens_freed`；
- `over_evicted_tokens`；
- 每次最大整叶节点数；
- 每轮最大 L1 token 预算；
- 可选的最小冷却时间。

候选优先级继续使用当前 eviction strategy，如 LRU、LFU、SLRU 或 priority，
并使用 `node.id` 作为稳定 tie-breaker。

### 11.3 可能出现的抖动

在 L2 持续满载时可能形成：

```text
删除冷 L1+L2
-> 为新节点写入 L2
-> 旧前缀再次到达
-> 从 L3 恢复或重算
-> 再写入 L2
```

如果以下指标同时上升，说明第二级回收过于激进：

- L1 sacrifice token；
- 重算 token；
- L3 read token；
- D2H write token；
- 同一时间窗口内的整叶回收次数。

此时应降低单轮预算、提高冷却时间，或退回 `l2-only` 策略。

## 12. 建议代码改动边界

### 12.1 `hiradix_cache.py`

建议新增或修改：

```text
HostReclaimReason
_is_attached_actual_leaf()
_is_l2_only_actual_leaf()
_is_l1_l2_actual_leaf()
_reclaim_actual_leaf()
_build_host_only_heap()
_build_l1_l2_heap()
evict_host(num_tokens, reason) -> int
```

同时：

- `_update_host_leaf_status()` 改为真实 L2-only 叶子语义。
- `write_backup()` 使用 `WRITE_STAGING`。
- `prefetch_from_storage()` 使用 `STORAGE_PREFETCH`。
- reset 时清理新增的统计状态。

### 12.2 `server_args.py`

建议增加一个默认关闭的新策略参数：

```text
--hicache-host-eviction-policy
    l2-only             # 当前行为
    whole-leaf-write    # 推荐：只有写入压力启用第二级
```

暂不暴露 `whole-leaf-all`，避免用户在当前 query-after-allocation 预取路径误用。

### 12.3 不建议修改

第一版不需要：

- 给 `TreeNode` 增加 L3 状态；
- 修改 Scheduler 请求状态机；
- 修改普通 `RadixCache._update_leaf_status()` 的 L1 逻辑叶子定义；
- 修改 L3 key 或存储格式；
- 引入新的传输 buffer；
- 同时修改 Unified/Mamba/Hybrid cache。

这样可以把维护面限制在 HiRadix Host 回收路径。

## 13. 监控指标

指标应使用低开销计数器，按周期汇总，不在每个节点删除时打印日志。

### 13.1 请求与结果

- `host_reclaim_attempts`
- `host_reclaim_requested_tokens`
- `host_reclaim_freed_tokens`
- `host_reclaim_shortfall_tokens`
- `host_reclaim_by_reason=write_staging/storage_prefetch`

### 13.2 第一级 L2-only

- `l2_only_candidates`
- `l2_only_nodes_evicted`
- `l2_only_tokens_evicted`
- `l2_only_over_evicted_tokens`

### 13.3 第二级 L1+L2

- `l1_l2_candidates`
- `l1_l2_nodes_evicted`
- `l1_l2_l1_tokens_sacrificed`
- `l1_l2_l2_tokens_freed`
- `l1_l2_over_evicted_tokens`
- `l1_l2_reclaim_ms`

### 13.4 拒绝原因

- `reclaim_reject_not_actual_leaf`
- `reclaim_reject_detached`
- `reclaim_reject_device_locked`
- `reclaim_reject_host_owned`
- `reclaim_reject_pending_d2h`
- `reclaim_reject_pending_h2d`
- `reclaim_reject_pending_l3`
- `reclaim_reject_rank_mismatch`
- `reclaim_reject_budget`

分析收益时重点比较：

```text
写入成功 token 增量
L2 allocation failure 降幅
L1 sacrifice token
后续 L3 read token
后续 recompute token
cache hit 与 end-to-end hit
TTFT、吞吐和 P99
```

如果只看到 `L2 allocation failure` 下降，但 L1 sacrifice 和 recompute 大幅
上升，不能认为优化成功。

## 14. 测试门槛

### 14.1 必须补齐的单元测试

1. `evictable_leaves` 中的节点拥有 L2-only child 时，不能整节点删除。
2. L2-only 真实叶子删除后不重复扣减 `evictable_size_`。
3. L1+L2 真实叶子删除后只扣减一次设备会计。
4. `lock_ref > 0` 时两个层级都不能删除。
5. `host_ref_counter > 0` 时两个层级都不能删除。
6. `write_through_pending_id` 或 ongoing operation 存在时不能删除。
7. 删除后父节点正确晋升为 L2-only 或 L1+L2 实际叶子。
8. 删除后 Session leaf 索引不再引用节点。
9. L1+L2 删除产生 GPU 和 CPU remove events；L2-only 只产生 CPU event。
10. Heap 中的过期、已解除、状态变化候选只被跳过，不触发断言。
11. 一个节点大于申请量时会正确停止，并统计 over-eviction。
12. `STORAGE_PREFETCH` 默认绝不触发第二级回收。
13. `WRITE_STAGING` 第一级不足时才触发第二级。
14. 功能关闭时行为与 `main` 一致。
15. TP 某个 rank 候选缺失或被锁时，所有 rank 都不删除该候选。
16. TP 某个 rank Host 容量不足时，所有 rank 一起进入回收并一起重试。

### 14.2 随机状态测试

建议生成随机 RadixTree，反复执行：

```text
insert
split
lock/unlock
D2H submit/ack
H2D submit/ack
L3 operation protect/release
L1 eviction
L2 eviction
session tag/release
```

每步检查：

- child key 与节点 key 一致；
- 非 root 节点有且只有一个 parent；
- 无内部节点被整节点删除；
- allocator free/used 总量守恒；
- `evictable_size_` 与实际未锁 L1 token 一致；
- Session 索引只指向仍附着的节点；
- candidate set 中不存在已解除节点；
- 所有 rank 删除序列一致。

### 14.3 故障注入与压测

- D2H 长时间未完成时触发 Host 压力。
- H2D load-back 进行中触发 Host 压力。
- L3 写部分失败后触发 Host 压力。
- 并发 32、64 及更高时持续触发 Host 回收。
- L2 很小、L1 较大时验证第二级回收收益。
- 热前缀反复访问时验证不会被错误选为冷候选。
- Session cache 和 KV events 分别启用时运行。

## 15. 分阶段实施

### 阶段 1：真实叶子与原子 helper

- 收紧 `_update_host_leaf_status()` 为真实叶子语义。
- 统一 L2-only 删除 helper，修复 Session 和会计边界。
- 实现 L1+L2 真实叶子的原子删除。
- 先以 TP=1、普通 MHA/GQA 单元测试验证。

### 阶段 2：写入路径启用

- 增加 `whole-leaf-write` 策略。
- 只在 `write_backup()` 的 L2 分配失败路径启用第二级。
- 加入回收预算、指标和周期汇总。

### 阶段 3：TP/CP 一致性

- 增加 Host 容量全局预检。
- rank 0 批量选择候选，其他 rank 验证。
- 所有 rank 同步删除和重试 D2H。
- 故障注入验证 rank mismatch 时安全降级。

### 阶段 4：生产灰度

- 默认继续使用 `l2-only`。
- 小流量开启 `whole-leaf-write`。
- 同时观察写入成功、L1 sacrifice、L3 read、重算、命中率和延迟。
- 验证收益后再扩大流量。

不建议在本方案中顺带启用预取整叶回收或实现其他传输路径。

## 16. 最终不变量

实现完成后必须始终成立：

1. 只有 `len(node.children) == 0` 的节点能从树上整体解除。
2. 内部节点无论 L1/L2 状态如何都不能整节点删除。
3. 有 `lock_ref` 或 Host/operation 引用的节点不能回收。
4. L2-only 删除不改变设备容量会计。
5. L1+L2 删除只改变一次设备容量会计。
6. 删除节点前清理 Session，且 GPU/CPU event 类型正确。
7. 父节点只在最后一个 child 被删除后成为真实叶子候选。
8. 任何 rank 不满足资格时，所有 rank 都不删除该候选。
9. 预取在确认 L3 hit 前不能牺牲 L1+L2 节点。
10. 功能关闭时不增加扫描、collective 或行为变化。
11. 删除决策不依赖永久 L3 状态。
12. 删除后 L3 miss 必须走正常重算，不能读取无效 KV。

## 17. Review 决策表

| 问题 | 决策 |
| --- | --- |
| 是否继续只从叶子删除 | 是，而且必须是真实拓扑叶子 |
| L1+L2 是否允许回收 | 允许 |
| 如何回收 L1+L2 | 预检后，在一个原子 helper 内释放 L1、L2并解除节点 |
| 是否先暴露为 L2-only 再删除 | 否，不向其他调度步骤暴露中间状态 |
| 是否检查 L3 一定存在 | 不检查 |
| L3 miss 后怎么办 | 正常重算 |
| 是否检查异步操作引用 | 必须检查 |
| 是否直接使用 `evictable_leaves` | 只作为候选来源，必须再过滤真实叶子 |
| 是否新增长期候选集合 | 第一版不新增 |
| 是否允许预取触发整叶回收 | 第一版不允许 |
| 是否允许写入触发整叶回收 | 允许，第一级不足时触发 |
| 是否直接支持 Hybrid/Unified/Mamba/PP | 第一版不支持 |
| 是否需要修改 Scheduler | 不需要 |
| 是否需要引入传输 buffer | 不需要，且不属于本方案 |

最终结论：在上述边界内，该方案设计合理、改动集中，能够让 L2 在写入压力下
回收原本不可用的 `L1+L2` 重复容量。主要风险不是 L3 状态，而是真实叶子判断、
异步所有权、容量会计和多 rank 一致性；这些必须通过原子 helper、同步协议和
故障测试解决后才能上线。

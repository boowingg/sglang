# 07. 写入、写穿与 L3 持久化

## 7.1 写路径的目标

新计算 KV 首先位于 L1。若希望未来请求跨 L1 淘汰、跨进程或跨实例复用，必须依次完成：

```text
L1 new KV
  -> Radix metadata insert
  -> L2 slots allocated
  -> D2H complete and ACK committed
  -> L3 page write complete
  -> host reference released
```

L2 写失败意味着当前节点还只有 L1；L3 写失败意味着本机 L2 淘汰后该页无法从 L3 恢复。

## 7.2 三种写策略

### `write_through`

节点首次满足写条件就从 L1 备份到 L2。源码的 `write_through_threshold=1`。

优点：

- 热度尚不明确时也尽快获得 L2/L3 副本；
- L1 淘汰更容易直接降级为 host-only；
- 跨请求/实例复用建立得早。

代价：

- D2H 与 L3 写流量最大；
- 短命/一次性前缀也会写；
- L2 staging 和 write queue 压力较高。

### `write_through_selective`

通常阈值为 2：节点在 insert 路径第二次被复用/覆盖时才写 L2。`hit_count` 由 `_inc_hit_count()` 在 insert 中增加；纯 `match_prefix()` 不增加，chunked insert 也跳过。它用这一近似热度粗略过滤一次性数据。

优点是减少写放大；缺点是第一次计算后若很快被 L1 淘汰，尚未达到阈值的 KV 不会进入 L3，未来请求还会重算。

### `write_back`

平时不主动备份；L1 需要淘汰未备份节点时，先 D2H，等待写回完成，再释放 L1。

优点是只写真正受到容量压力的数据；缺点是淘汰关键路径更复杂，瞬时 L2 和 D2H 压力集中，写回失败会直接限制 L1 可回收容量。

## 7.3 连续 L2 前缀不变量

经典 write-through 的重要约束：从 root 到已备份节点不能有空洞。

```text
root -> A(backed) -> B(not backed) -> C(backed)   X 不允许
root -> A(backed) -> B(backed) -> C(backed)       ✓
```

如果 C 有 host 副本但 B 没有，L1 淘汰后无法从 root 连续走到 C，L3 的 hash anchor 和 host match 都会断裂。

因此 `write_backup(node, write_back=False)` 在 parent 非 root 且 parent 未备份时返回 0。它不会递归跳过父节点直接备份孩子。

## 7.4 `write_backup()` 分步解释

### 1. 检查父前缀

保证上述连续性。write-back 淘汰场景使用不同策略，可批量处理路径。

### 2. Controller 分配 host slots

```python
host_indices = controller.write(device_indices=node.value, node_id=node.id)
```

`controller.write()`：

- `mem_pool_host.alloc(len(device_indices))`；
- 创建 `CacheOperation(host_indices, device_indices, node_id)`；
- 追加到 `write_queue`；
- 当前实现立即调用 `start_writing()`，提交异步 D2H 并登记 ACK；
- 返回 host indices。

若失败，Tree 先 `evict_host()` 再重试一次。仍失败则本次写入放弃，节点保持只有 L1。

### 3. 控制面暂存 `host_value`

成功返回后 Tree 设置 `node.host_value` 并登记 pending ACK。注意 D2H 在 `controller.write()` 内已经提交但通常还没完成，因此：

- `backuped` 从字段上看为真；
- 但 `write_through_pending_id` 表明数据未发布完成；
- 节点 L1 路径被 `inc_lock_ref()` 保护，不能在 D2H 期间释放源数据。

这再次说明不能只看 `host_value is not None` 判断字节已经安全。

提交 D2H 先于 Tree 登记 `host_value/pending id`，但完成 ACK 只能由后续 Scheduler polling 收割，所以当前串行调用内不会出现“ACK 已提交、Tree 还没登记就被消费”的重入。若以后改成 CUDA callback 或后台线程直接提交 Tree，必须先调整这段注册顺序或增加显式 operation 状态机。

## 7.5 D2H 提交与“合并”边界

`start_writing()` 能把 `write_queue` 中多个 operation 合并。下面是它支持的语义示例：

```text
op1: host [0..255] <- device [1000..1255], node 7
op2: host [512..767] <- device [2000..2255], node 9

merge:
host_indices   = cat(op1.host, op2.host)
device_indices = cat(op1.device, op2.device)
node_ids       = [7, 9]
```

但经典路径的 `write()` 每追加一项就立即调用 `start_writing()` 并清空 queue，所以常态是 `merge_ops([op1])`，并没有跨多个 tree node 的实际批量。`load_queue` 到 batch 结束才启动，H2D 才是真正的跨请求合并。评估小 D2H launch/concat 开销时必须按这一真实调用时机分析。

根据 layout/backend，`move_indices()` 可能：

- 把 host indices 异步搬到 GPU，供 kernel 索引；
- 把 device indices 搬到 CPU，供 direct copy；
- 对 layer-first direct 路径按 host index 排序，提高拷贝地址规律性。

随后在 `write_stream` 上执行 all-layer D2H，记录 `finish_event`，并把 indices tensor `record_stream()`，保证它们在 stream 使用完前不会被 Python/allocator 回收。

完成记录为：

```text
ack_write_queue += (start_event, finish_event, [node ids])
```

## 7.6 ACK 为什么由 Scheduler 收割

后台 CUDA stream 只负责字节搬运。树结构、lock_ref、ongoing dict 由 Scheduler 线程串行修改，减少数据竞争。

`writing_check()`：

1. PP0 对 ACK queue 从头调用 `event.query()`，只统计连续完成项；
2. TP/PP 同步共同 `finish_count`；
3. 对这些项 `event.synchronize()`；
4. 调 `_finish_write_through_ack()`；
5. 从 queue 头部移除。

即使某 rank 的 `ongoing_write_through` 为空，也必须进入 collective。条件性跳过会让不同 rank 的 collective 序列错位并死锁。

## 7.7 ACK 与节点分裂

D2H 排队后节点可能被新请求匹配并分裂。原操作复制的是 enqueue 时整段数据，例如 12 token；分裂后可能成为 8+4 两个节点。

pending 记录包含：

```text
(lock_node, backup_len, publish_nodes)
```

分裂时 `_replace_pending_write_through_node(old, [prefix, suffix])` 更新发布列表。ACK 到来时：

- 清理每个碎片的 pending id；
- 为每个碎片发布 CPU store event；
- 保留 enqueue 时的 `backup_len`，供 L3 写重建原操作范围；
- 释放原 lock node 的 L1 锁。

不做这一步会出现“树结构已经分裂，但 ACK 仍只认识旧节点”的悬空状态。

## 7.8 从 L2 写 L3

`_finish_write_through_ack()` 在启用 L3 时调用 `write_backup_storage()`。

写 L3 所需三项：

```text
token key  -> 生成或核对每页 hash
hash_value -> L3 entry keys
host_value -> 每页数据的 L2 源地址
```

如果节点在 D2H 期间分裂，`_concat_split_chain(node, backup_len)` 沿祖先链恢复 enqueue 时的完整 key/hash/host indices。Bigram 模式还要去掉段之间共享的边界 token。

Controller 创建 `StorageOperation` 放入 `backup_queue`，Tree 保存：

```text
ongoing_backup[operation_id] = node
node.protect_host()
```

host ref 必须一直持有到 L3 写 ACK，否则 L2 槽可能被重分配，后端写出的就是其他节点数据。

## 7.9 L3 backup thread

`backup_thread_func()` 循环取 operation，调用 `_page_backup()`：

1. 每 128 页切批；
2. `page_set_func(batch_hashes, batch_host_indices)`；
3. 只有整批被 `page_set_func` 聚合判定成功才增加 `completed_tokens`；
4. 失败则记录 `Write page to storage: N pages failed` 并停止后续批；
5. 无论成功多少，operation 最终进入 `ack_backup_queue`。

Scheduler drain backup ACK 时：

- 从 `ongoing_backup` 删除 operation；
- `release_host()`；
- 记录实际 completed token 指标。

### 关键语义

ACK 表示“后台写操作已结束”，不一定表示“全部页成功”。必须比较：

```text
operation.completed_tokens vs operation planned tokens
```

只统计 operation ACK 数会把失败操作误判为成功。

当前 `_page_backup()` 还有 `todo: allow partial success`：generic 接口返回单个 bool，V1 的 per-page list 也会先被 `all(...)` 折叠。只要批内一页失败，Controller 对该批 128 页（或尾批）整体不增加 `completed_tokens`，即使 backend 内部已有部分页实际落地。因此这里的 failed token 是“Controller 未确认的整批 token”，不一定等于后端物理上一页都没写成。

## 7.10 写失败会造成什么

### L2 alloc 失败

- 不产生 D2H operation；
- 节点仍只有 L1；
- L1 后续淘汰时可能丢失可复用副本；
- write-through 的连续前缀可能停在 parent。

### D2H 内核/事件异常

- pending 节点锁可能无法释放；
- 可淘汰 L1 下降，最终调度 OOM；
- host_value 不应被当作有效数据；
- 通常属于严重运行错误，不是普通 cache miss。

### L3 set 返回失败

- L2 数据仍可能有效；
- 当前机器在 L2 未淘汰前仍可复用；
- L2 淘汰或进程重启后该页不可恢复；
- 跨实例 L3 命中下降；
- 若失败集中在高并发，未来命中会滞后下降。

因此分析“写失败是否影响 cache-hit”必须考虑时间延迟：当前轮可能仍命中 L2，后续工作集轮转后才暴露 L3 缺页。

## 7.11 写压力怎样反过来影响读

L2 同时承担：

- 可复用 host-only 数据；
- L3 prefetch 目标缓冲；
- L1 -> L2 写穿目标；
- L2 -> L3 在途源缓冲；
- draft/aux pool 数据。

高并发下可能形成：

```text
更多新计算
 -> 更多 D2H 写穿
 -> 更多 L3 backup
 -> host ref 持有时间变长
 -> 可回收 L2 减少
 -> L3 prefetch allocation 降低
 -> 有 L3 命中也无法读入
 -> 更多重算
 -> 产生更多新写
```

这是正反馈压力环。只看 backend read/write 带宽，可能看不到真正瓶颈是 host slots 长时间被在途写保护。

## 7.12 写路径应监控的 token 指标

| 指标 | 含义 |
|---|---|
| write attempts/tokens | 有多少 L1 数据尝试进入 L2。 |
| L2 alloc success tokens | 有多少真正取得 host slots。 |
| D2H enqueued/completed tokens | 排队与 CUDA 完成差距。 |
| write ACK lag | 已排队但控制面尚未收割的 token。 |
| L3 planned/completed tokens | L3 写完整率。 |
| failed operations/batches/tokens | 失败规模，token 最重要。 |
| host refs held by backup | L3 在途写占住多少 L2。 |
| backup queue depth/age | 后端写服务率是否低于产生率。 |
| D2H GB/s / latency | PCIe/内核是否成为瓶颈。 |
| L3 set GB/s / latency | 分布式存储写能力。 |

由于经典 `write_queue` 会被立即清空，它的瞬时 depth 通常无法表示 D2H 压力；更有意义的是 `ack_write_queue` 在途 token、最老 ACK age、write stream 时长和被 pending lock 保护的 L1 token。

## 7.13 `Write page to storage` 日志来源

该警告来自 Controller 的 `_page_backup()`，不是普通文件后端自动打印：

```text
page_set_func(...) 返回 False
  -> logger.warning("Write page to storage: ... pages failed.")
```

它说明一整个当前 batch 未被 Controller 认可为成功。要进一步区分：

- backend 是全部失败还是部分失败却聚合为 False；
- 网络 timeout、容量拒绝、key 冲突还是 buffer 地址问题；
- operation 计划页数与 `completed_tokens`；
- 失败前是否已有若干 batch 成功。

下一章：[淘汰、容量与资源所有权](08-eviction-and-ownership.md)。

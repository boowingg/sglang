# 06. L1、L2、L3 读路径精讲

## 6.1 读路径不是一次 lookup

完整读路径有三个不同动作：

1. **Local match**：只查 Radix 元数据，找到 L1/L2 已知前缀；
2. **Storage prefetch**：为剩余 token 生成 hash，查询并读取 L3 到 L2；
3. **Load back**：把 L2 命中搬到 L1，供模型计算。

```text
token -> tree match -> L3 query -> L3 read -> host tree publish
      -> rematch -> L1 allocation -> H2D -> layer event -> reuse
```

## 6.2 本地匹配：连续 L1 + 连续 L2

HiRadix 的匹配结果需要区分：

- `device_indices`：从 root 开始连续存在于 L1 的前缀；
- `last_device_node`：最后一个完整 L1 节点；
- `last_host_node/best_match_node`：继续沿 host-only 节点走到的最深位置；
- `host_hit_length`：L1 之后可从 L2 回载的长度。

为什么只允许连续前缀？Transformer 不能跳过中间上下文。即使 L2 有第 10 页而第 9 页缺失，第 10 页 KV 的计算上下文也无法被当前请求安全接上。

## 6.3 L3 prefetch 候选范围

Scheduler 从 `matched_len` 后取 token，Tree 再做：

1. 构造带同一 `extra_key` 的 `RadixKey`，用于保持本地树语义；当前 native L3 hash 仍只读取 token/bigram 和 prior digest；
2. `page_aligned(page_size)`；
3. 检查 `prefetch_length >= prefetch_threshold`；
4. 检查 `prefetch_rate_limited()`。

短于 threshold 时直接重算，原因是远端 query/read、host 分配和 H2D 的固定开销可能大于节省的计算。

## 6.4 为什么先预留整段 L2

`prefetch_from_storage()` 在知道 L3 命中长度前，为候选后缀预留 host slots。这样后台 I/O 获得稳定目标地址，不会在 query 后再跨线程争抢 allocator。

流程：

```text
尝试 alloc(prefetch_length)
  失败 -> evict_host(prefetch_length) 后重试
  仍失败 -> 按当前 available_size 缩短到 page boundary
  缩短后仍低于 threshold -> 放弃
```

代价是大量并发 prefetch 会暂时占用 L2。Controller 额外设置：

```text
prefetch_capacity_limit = 0.5 × host_pool.size
```

`prefetch_tokens_occupied` 达到这个上限时新预取被 rate limit，意图为写回路径保留大约一半 L2。它不是物理容量分区：单个新 operation 可从阈值以下一次跨过阈值，resident/写回也仍能使用同一 free list；query 后已释放尾部时，该计数仍可能在请求收割前按原候选长度保留。因此它更接近 admission watermark，不是精确的“prefetch 当前实占 slots”。

## 6.5 Anchor host lock

后续 hash 从 `last_host_node.get_last_hash_value()` 接续。如果预取期间 anchor 被 host eviction 删除：

- hash 链上下文可能失效；
- `ongoing_prefetch` 保存的节点引用变成脱树对象；
- 收割时无法在正确位置插入。

因此 prefetch 对 anchor 增加 host 引用，结束或 abort 后释放。

## 6.6 Query thread

`_storage_hit_query()`：

1. 根据 token、`last_hash` 和 `page_size` 生成全部候选页 hash；
2. 每 `STORAGE_BATCH_SIZE=128` 页调用一次 backend exists；
3. backend 返回这一批从头连续存在的页数；
4. 如果小于本批长度，立即停止；
5. 必要时把已确认页追加到 `prefix_keys`。

例如：

```text
batch 0: 128/128 hit
batch 1:  37/128 hit
=> 总命中 165 pages，后续不再查询
```

### 跨 rank 同步

每个 attention rank 都查自己对应的 L3 shard，然后 all-reduce MIN。只要一个 rank 少一页，所有 rank 都缩短到该页之前。

### 低收益 revoke

共同 `storage_hit_count < prefetch_threshold` 时：

- request id 放入 `prefetch_revoke_queue`；
- 全部预留 host indices 放入 release queue；
- 不进行真实读取。

Tree 在 scheduler step 中 drain revoke，释放 anchor lock 和 occupied 计数。

## 6.7 Read thread

命中达到 threshold 后：

- `operation.hash_value` 截到共同命中页；
- `operation.host_indices` 截到对应 token；
- 未命中尾部进入 release queue；
- operation 交给 prefetch I/O thread。

`_page_transfer()` 同样按 128 页批处理。每批必须完整成功才继续下一批：

```text
已成功 128 pages
下一批计划 80 pages，但第 20 页 read 失败
=> operation 只发布实现明确确认的连续完成 token
=> mark_terminate，停止后续读取
```

`completed_tokens` 必须在数据真正写入 host buffer**之后**增加，否则 Scheduler 可能先看到完成计数，再读取尚未填好的页。

## 6.8 三种停止策略

### `best_effort`

Scheduler 一检查就允许终止，甚至可能发生在 query thread 尚未填入 `hash_value` 时。它能降低等待，但后台尚未读到的后缀会被放弃，适合不愿让 L3 增加排队延迟的场景。

### `wait_complete`

只有 `completed_tokens == queried_hit_tokens` 才终止。命中利用率高，但后端卡住会让请求长时间停在 waiting queue。

### `timeout`

完成或超过线性 timeout：

```text
timeout = min(max_timeout,
              base_timeout + per_ki_token × token_count / 1024)
```

短请求给固定开销，长请求按 token 增加等待，上限避免无限阻塞。

对于多 pool operation，即使主 KV 完成，必要 auxiliary pool 尚未完成也不能终止为成功。

### 这不是 backend RPC 的强制取消

`timeout` 是 Scheduler 决定“请求不再继续等多少结果”的 cutoff。L3 的 `batch_exists/batch_get` 是后台线程中的同步调用；若自研 backend 卡在一次 RPC 内，`mark_terminate()` 不能强行中断这次 Python/底层调用，线程仍要等 backend 自己的 timeout/cancel 返回。生产后端必须设置有上限的连接、RPC 和重试超时，否则一个卡住的调用会长期占住 query/read worker，即使请求侧已经开始重算。

### 跨 rank 终止语义

状态打包后 all-reduce：

- 所有 rank 都满足完成/超时，才正常终止；
- 任一 rank 已被显式 terminate，其他 rank 也终止；
- 避免有的 rank 进入 batch、有的仍在等 L3。

## 6.9 收割与 host tree 插入

`check_prefetch_progress()` 在 Scheduler 线程执行：

1. 若 operation 尚未真正发出，视为本次无可等待结果；
2. 若停止条件未满足，返回 `False`，请求本轮跳过；
3. terminate 并取完成长度；
4. 再跨 rank 取 MIN；
5. `_insert_helper_host(last_host_node, fetched_key, written_indices, hashes)`；
6. 处理与已有树前缀重复的 `prefix_len`；
7. 释放重复分配和 rank 尾差；
8. 保存真正新增到树的 token 数。

### 为什么 `loaded_from_storage` 不是 `completed_tokens`

假设后台读了 1024 token，但在收割前另一个请求已把其中 512 token 插入 L2：

```text
completed_tokens = 1024
insert_result.prefix_len = 512
loaded_from_storage = 512
```

前 512 token 的预留 slots 是重复副本，应释放；只有后 512 扩展了可复用目录。

## 6.10 L2 回载的 all-or-none 原则

经典 `load_back()` 收集从最近 L1 ancestor 到 deepest host hit 的连续节点，并一次性申请对应 L1 slots。它不会只载入任意中间一部分，原因是：

- 请求的 `prefix_indices` 需要是连续上下文；
- node.value 的发布与树边长度必须一致；
- 分批部分成功会显著增加 rollback、锁和跨 rank 一致性复杂度。

两个跳过条件：

```text
host token 数 < load_back_threshold
host token 数 > mem_quota + lock delta
```

太短时重算可能更划算；超 quota 时加载会破坏 Scheduler 的 L1 预算。

## 6.11 L1 分配失败

Controller `load()` 先向 device allocator 申请。失败时 Tree：

1. 尝试淘汰足够 L1 token；
2. 再次 `load()`；
3. 仍失败则释放 host 保护，返回 `None`；
4. `init_load_back()` 向上退到仍有 L1 的 ancestor；
5. 未恢复后缀留在 extend input 中重算。

所以“GPU 等 L3 是否超时”只描述 L3 阶段。即使 L3 已完整读入 L2，最终还可能因 L1 容量不足而重算。

## 6.12 H2D 与模型计算重叠

`start_loading()` 对每层调用 `load_to_device_per_layer()`，每层完成后记录 event。`LayerLoadingEvent.wait(layer_id)` 让模型在需要某层前等待。

这要求：

- device indices 在整个 stream 执行期间存活；
- host indices 在整个 H2D 期间不能被覆盖；
- node.value 虽已发布，但节点 lock_ref 必须保持到 finish ACK；
- batch consumer index 必须对应正确 producer ring slot。

`loading_check()` 在后续 scheduler step 中确认共同完成 ACK，释放节点的 L1 路径保护。经典路径的 host ref 在 `load_back()` 提交控制面状态后已经释放；L2 槽仍因节点已有 L1 value 而不满足 host-only eviction 条件。

## 6.13 读路径损失漏斗

建议把一次运行统计成：

```text
attempt_tokens
  - local L1 hit
  - local L2 candidate
  - L3 query candidate
      × query enqueue ratio
      × backend continuous hit ratio
      × L2 allocation ratio
      × storage read completion ratio
      × L1 allocation/load-back ratio
  = end-to-end reused tokens
```

每一层都应使用 token 数而不是只用 operation 数。一个 4096-token 失败操作和一个 4-token 失败操作权重完全不同。

## 6.14 读路径失败定位表

| 现象 | 主要检查 |
|---|---|
| L3 query 很少 | threshold、rate limit、没有可靠 host anchor、请求提前匹配到末尾。 |
| Backend hit 高但最终命中低 | L2 allocation、read completion、L1 load-back。 |
| 并发提高后 allocation 比例下降 | prefetch 预留占 L2、host-only 工作集、写回 staging 竞争。 |
| `read_failed=0` 但命中下降 | 失败可能发生在 query miss 或 allocation，根本没有进入 read。 |
| request 等待时间长 | wait_complete、L3 tail latency、跨 rank 最慢 shard。 |
| H2D 平均时间升高 | PCIe/NUMA、搬运布局、批量大小、计算争用。 |
| L3 有页但不复用 | 链式 hash/namespace 不同、某 auxiliary pool 缺页、非连续缺口。 |

下一章：[写入、写穿与 L3 持久化](07-write-path.md)。

# HiCache Query-First L3 预取

> 状态：已实现，待真实 GPU/L3 负载验证
>
> 开发分支：`hicache-query-first`
>
> 前置改动：
> [`hicache_l2_leaf_reclaim_design.md`](./hicache_l2_leaf_reclaim_design.md)

## 1. 目标

原来的 L3 预取顺序是：

```text
按候选前缀长度申请 L2
-> 必要时回收 L2-only 或 L1+L2 叶子
-> 查询 L3 实际连续命中长度
-> 释放没有命中的预分配页面
-> 读取命中 KV
```

候选长度为 4096 token、L3 只命中 1024 token 时，原路径仍会先尝试为 4096
token 腾出 L2。高并发下，这会放大 L2 压力，并可能为最终 miss 的数据牺牲冷
L1 节点。

Query-first 将顺序调整为：

```text
后台查询 L3
-> 得到实际连续命中长度
-> 只为实际命中部分申请或回收 L2
-> 启动原有 L3 到 L2 读取
```

它解决的是“查询结果未知时过度预留 L2”，不是新的传输协议，也不是新的缓存
层。

## 2. 最小改动边界

本实现刻意复用现有结构：

- 复用原 `prefetch_thread` 执行 L3 `batch_exists()`。
- 复用原 `prefetch_buffer` 和 `prefetch_io_aux_thread` 读取页面。
- 复用原 `PrefetchOperation`，只增加查询与读取阶段状态。
- 复用原 `evict_host(..., allow_l1_l2=True)` 两级回收。
- Scheduler 只新增一个本地查询结果队列的消费步骤。

没有修改：

- L3 backend 接口和 key/hash 格式。
- RadixTree 节点字段、匹配、切分或插入算法。
- D2H、L2 到 L3 写入和写入 ACK 流程。
- L3 到 L2 的页面读取实现。
- L2 到 L1 的 load-back 流程。
- CLI 参数和服务端请求结构。
- Unified、Mamba、Hybrid 与 draft KV 路径。

因此后续同步 SGLang 上游时，主要冲突面仍局限于
`cache_controller.py` 和 `hiradix_cache.py` 的 prefetch 小段。

## 3. 请求时序

```text
Scheduler
  |
  | protect_host(anchor)
  | query_prefetch(candidate_tokens)
  v
prefetch_queue
  |
  v
prefetch_thread
  | batch_exists()
  | 发布实际连续命中 token
  v
prefetch_query_result_queue
  |
  v
Scheduler
  | 命中不足：结束，不申请 L2
  | 命中足够：按实际命中申请 L2
  | 分配失败：L2-only -> L1+L2 真实叶子回收
  | 仍不足：尝试页对齐的部分读取
  | resume_prefetch()
  v
prefetch_buffer
  |
  v
prefetch_io_aux_thread
  | 原有 page transfer
  v
Scheduler
  | 将完成前缀插入 Host RadixTree
  | release_host(anchor)
```

结果队列传递 `PrefetchOperation` 对象本身，而不是只传 request id。Scheduler
会同时校验 map 中保存的 operation 身份，过期结果不能误续传一个复用相同 id
的新请求。

## 4. Operation 状态

没有引入枚举或新的状态机类。`PrefetchOperation` 只增加：

| 字段 | 含义 |
| --- | --- |
| `query_first` | 是否走新路径，创建后不变 |
| `_query_done` | L3 hit query 是否已经发布 |
| `_read_started` | 是否已经绑定 L2并进入原读取队列 |
| `storage_hit_tokens` | L3 查询确认的连续命中 token |

状态转换只有一条主路径：

```text
QUERYING
-> QUERY_DONE
-> READING
-> 原有完成或终止处理
```

取消可以发生在任意阶段：

```text
QUERYING --cancel/timeout--> TERMINATED
QUERY_DONE --cancel--------> TERMINATED
READING ----cancel---------> 原有 terminate + 页面释放
```

`finish_query()` 和 `start_read()` 在 operation 原有锁内执行，并且只能成功一次。
这样可以防止：

- 查询完成与请求取消同时发生时发布无主结果。
- 结果队列与请求进度检查同时尝试启动两次读取。
- 已取消 operation 重新绑定刚申请的 L2 页面。

`host_indices is None` 只表示查询阶段尚未绑定 L2；业务判断仍使用显式
`query_first` 和 `is_read_started()`，不靠空指针推断完整状态。

## 5. 容量申请

Scheduler 收到查询结果后执行：

```text
target = min(L3 连续命中, 候选长度)
target = page aligned
```

如果 `target < prefetch_threshold`：

- 不申请 L2。
- 不触发任何 L1/L2 回收。
- 释放 anchor 的 Host 引用。
- 归还该请求占用的 prefetch budget。

如果命中足够：

```text
1. alloc(target)
2. 失败则 evict_host(target, allow_l1_l2=True)
3. 再次 alloc(target)
4. 仍失败则按实时 available_size 尝试部分分配
5. 部分长度仍必须页对齐且不低于 prefetch_threshold
```

例如：

```text
候选 4096
L3 实际连续命中 1024
```

新路径只申请、回收和读取 1024 token。L3 零命中时，L2 allocator 完全不会被
调用。

## 6. 取消、超时和异常

### 6.1 查询期间取消

Scheduler：

1. 标记 operation terminated。
2. 从 `ongoing_prefetch` 删除同一个 operation。
3. 释放 anchor。
4. 归还完整候选长度对应的 prefetch budget。

后台查询稍后返回时，`finish_query()` 会拒绝发布结果。

### 6.2 查询超时

查询阶段继续使用现有线性 timeout 配置，但 token 数取完整查询候选长度。读取
真正开始后会重置 `start_time`，读取阶段再按实际读取长度使用现有 timeout。

`best_effort` 不等待尚未完成的查询；`wait_complete` 等待查询；`timeout` 等到
查询完成或查询阶段超时。这保持了原 prefetch stop policy 的含义。

底层 `batch_exists()` 没有通用取消接口。因此 timeout 能让请求停止等待并释放
本地 bookkeeping，但不能强行中断已经进入 backend 的函数调用。

### 6.3 查询异常

Query-first 的 L3 查询异常会被转换为零命中结果，Scheduler 随后正常清理请求，
避免 prefetch 线程退出后请求永久等待。

### 6.4 Query 与 Read 之间 L3 数据消失

L3 可能在查询后、读取前淘汰页面。现有 `_page_transfer()` 会在首个读取失败处
停止，只发布完整读取的连续前缀，并释放未完成页面。因此不会把未初始化 KV
插入 RadixTree；代价只是该请求对缺失部分重新计算。

## 7. 并行与模型边界

第一版只在以下条件全部满足时启用：

```text
TP = 1
CP = 1
PP = 1
普通 MHATokenToKVPool（MHA/GQA）
非 HybridCacheController
未启用 draft KV
```

不满足时自动使用原 eager allocation 路径，并只打印一次 warning。

限制的原因不是 L3 查询本身，而是查询结果返回 Scheduler 后会执行 L2 allocator
和可能改变 RadixTree 拓扑的 L1+L2 整叶回收。多 rank 若没有一致的候选选择和
同序状态推进，可能导致树结构分叉或 collective 次序错误。

## 8. 性能考虑

新增热路径成本：

- 每个 query-first operation 的少量布尔状态和一个锁内状态切换。
- 查询完成时向无界本地 `Queue` 放入一个对象引用。
- Scheduler 每轮按队列开始时的 `qsize()` 快照消费，避免持续到达的结果无限
  拉长单轮调度。

没有：

- 新线程。
- token 级扫描。
- 周期统计锁。
- L2 数据复制。
- 写路径额外判断。

候选 token 在查询阶段仍计入现有 `prefetch_tokens_occupied`。它不代表已经占用
L2，而是作为查询积压的保守限流，避免慢 L3 下无限堆积查询请求。

## 9. 测试

CPU 单元测试：

[`test_hiradix_query_first.py`](../../test/registered/unit/mem_cache/test_hiradix_query_first.py)

覆盖：

1. 查询和读取状态只能发布或启动一次。
2. 已取消查询不能发布结果或启动读取。
3. 查询线程在不持有 L2 时发布命中结果。
4. 查询异常转换为零命中。
5. 提交阶段不调用 L2 allocator 或回收。
6. 按实际命中而不是候选长度申请 L2。
7. 低命中不申请 L2并完整释放 bookkeeping。
8. 压力回收使用实际命中长度。
9. 完整分配失败后只读取页对齐的可用部分。
10. 查询期间 abort 与 timeout 不泄漏 anchor 或 budget。
11. resume 失败时释放刚申请的 L2。
12. stale result 不能续传复用 request id 的新 operation。
13. `best_effort` 不因 query-first 额外等待。
14. 读取完成后正确接回原 Host RadixTree 插入与会计路径。

本地没有 GPU、模型或真实 L3，因此不会启动 SGLang。真实环境还需验证：

- 并发 32、64、128 下 L2 allocation failure 和端到端 hit。
- L3 零命中、短命中、慢查询和 query/read 间淘汰。
- 长时间运行后的 allocator used/free 守恒。
- 请求 abort、waiting timeout 与 runtime detach。
- 写入吞吐和 L3 write failure 与基线一致。

## 10. 正确性不变量

1. L3 查询完成前不能申请 L2。
2. L3 零命中或低于 threshold 时不能触发 L1/L2 回收。
3. L2 申请长度不能超过 L3 连续命中和候选长度。
4. 读取长度、host index 数量和 hash 页数必须一致。
5. 同一个 operation 最多启动一次读取。
6. canceled operation 不能发布查询结果或绑定新 L2。
7. stale result 必须通过 operation 身份校验被丢弃。
8. anchor 和 prefetch budget 必须且只能释放一次。
9. 部分读取必须保持页对齐。
10. 写路径不能调用 query-first 接口或激进 L1+L2 回收。
11. 不支持的部署形态必须保留原 eager 路径。

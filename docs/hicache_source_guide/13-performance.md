# 13. 性能模型、高并发与命中率

## 13.1 命中率由内容和系统共同决定

理论内容命中表示“过去是否计算过相同前缀”。实际有效命中还要经过资源和时序漏斗：

```text
effective reuse
  = content overlap
  × metadata discoverability
  × L3 retention/visibility
  × L2 admission
  × L3 read completion before cutoff
  × L1 admission
  × H2D completion
  × cross-rank common fraction
```

任一项随并发下降，最终 cache-hit 就低于理论内容复用上限。

本章的 token 漏斗字段和 `HiCache cumulative summary` 案例用于说明推荐的观测模型，也可对应曾经的插桩分支日志；本书固定的 `main@f1c247e` 并不内置这整行十秒累积汇总。它已有一小组 Prometheus 指标，见 13.12，但覆盖不了完整漏斗。不要因为在正文看到字段名，就假设当前主分支已经采集了它。

## 13.2 用 token 流量而不是请求数规划

请求数不能直接表示缓存压力。两个并发 64 的工作负载可能完全不同：

```text
A: 64 × 512-token prompt
B: 64 × 8192-token prompt
```

应至少统计：

- 输入 token/s；
- page/s；
- 每请求候选复用 token 分布；
- 新计算 token/s；
- L1/L2/L3 read/write token/s；
- KV bytes/token。

换算：

```text
KV byte rate = token rate × size_per_token
page rate = token rate / page_size
```

对于 128 KiB/token，10000 token/s 意味着约 1.22 GiB/s 的单方向 KV 字节流，还未考虑读写同时发生和辅助 pool。

## 13.3 四个基本容量/服务率

定义：

```text
C1 = L1 token capacity
C2 = L2 token capacity
R_h2d = L2->L1 token/s
R_d2h = L1->L2 token/s
R_l3r = L3->L2 token/s
R_l3w = L2->L3 token/s
Q_exist = L3 exists pages/s
```

稳定运行至少需要长期平均：

```text
需要回载率 < R_h2d
需要写穿率 < R_d2h
L3 miss后读取率 < R_l3r
新持久化率 < R_l3w
查询页率 < Q_exist
```

即使平均小于服务率，突发流量仍需要 queue/buffer 吸收；p99 延迟决定瞬时占用。

## 13.4 Little's Law 看在途 L2

近似：

```text
inflight_tokens ≈ arrival_token_rate × average_operation_latency
```

例如 L3 prefetch 进入率 8000 token/s，平均 query+read 0.5s：

```text
约 4000 token 的 L2 只是在途预留
```

若 p99 达 3s，突发时可能需要 24000 token。高并发引发后端排队，延迟增加；延迟又增加在途 L2，占用进一步压缩 resident 和写回空间。

## 13.5 L1 工作集与 reuse distance

缓存能否命中不仅取决于“曾经出现”，还取决于再次访问前插入了多少其他数据。Reuse distance 大于 L1/L2 可保留工作集时，副本会被淘汰。

并发从 32 到 64：

- 同一会话相邻轮次在时间上被更多请求隔开；
- 中间产生更多新 KV；
- L1/L2 eviction churn 增加；
- 下一轮到来时数据更可能只剩 L3，甚至 L3 尚未写完。

因此即使请求集合不变，提高并发也会改变访问顺序和 reuse distance。

## 13.6 L2 的四方竞争

传统 HiCache 的 L2 同时服务：

```text
1. resident host cache
2. L3 prefetch destination
3. D2H write-through destination
4. L3 write source protected by host ref
```

再加 Unified auxiliary/draft pool。容量分配不是显式静态分区，但生命周期约束形成事实上的竞争。

Controller 把 prefetch occupied 限制为 L2 的 50%，只是阻止 speculative read 吃满全部 L2；它不保证另外 50% 对写路径一定够，也不表示 resident host data 有固定保留量。

## 13.7 并发升高后命中下降的主要原因

### 1. L2 admission 下降

预取候选存在于 L3，但无法获得 host slots。表现：backend hit 尚可，allocation ratio 明显下降，read failed 可能仍为 0，因为根本没进入 read。

### 2. L3 写入落后

更多新计算产生更多 backup；`R_l3w` 跟不上时 queue/host refs 增长。未来请求查询时页尚未写成功，backend miss 增加。

经典 D2H 的 `write_queue` 往往不是合适的积压指标：`write()` 入队后会立即调用 `start_writing()` 并清空它。应观察 `ack_write_queue` 的未完成数量/最老年龄、D2H planned-to-completed 延迟、L3 `backup_queue` 等待年龄，以及被 backup host ref 保护的 token。

### 3. L3 读取赶不上调度 cutoff

Best-effort/timeout 会在排队延迟变大后更早放弃后缀。后端最终可能能读到，但请求已经进入重算。

### 4. L1 load-back admission 下降

L2/L3 成功，GPU slots 却不足。Scheduler 为 decode、新输入和回载共同争 L1，最终 host hit 不能转成有效复用。

### 5. 跨 rank tail amplification

更多并发让某个 shard/NUMA/网卡成为热点。最终取 MIN，单个慢 rank 拉低所有 rank 的可用长度。

### 6. Reuse distance 变大

前缀再次使用前已被更多其他请求挤出 L1/L2。

### 7. 请求组成改变

Poisson/限并发 benchmark 中，提高 `max_parallel` 会改变排队和轮次交错。若后续轮次延迟，公共前缀复用窗口也会变化。

### 8. Scheduler/CPU 开销

更多 ongoing dict、queue drain、tensor concat/all-reduce 和日志使 scheduler step 变慢。ACK 收割延迟会表现为资源长期被锁。

## 13.8 一个诊断案例

以下数值来自一组外部测试日志，不是当前 `main` 自动打印的默认字段。

假设两次运行有如下漏斗：

```text
并发 32:
backend_hit_pct = 74.95%
allocation_pct  = 99.41%
load_pct        = 100%
end_to_end_pct  = 74.27%

并发 64:
backend_hit_pct = 69.26%
allocation_pct  = 79.62%
load_pct        = 100%
end_to_end_pct  = 54.96%
```

解释顺序：

1. Backend hit 下降约 5.7 个百分点，说明 L3 内容可用性/写入时序/reuse mix 有一些恶化；
2. 最大损失来自 allocation：约 20% 的 backend-hit token 没取得 L2 空间；
3. 已分配部分 load 仍 100%，因此不是 L3 read 失败；
4. `write_failed=0` 不能排除写路径压力，因为成功但较慢的 L3 写仍会延长 host ref 和减少 L2 available；
5. 应优先看 L2 占用分解、prefetch reservations、backup host refs 和 allocation retry，而不是先调 L3 read timeout。

这类分析必须按 token 漏斗，不应仅凭 queue 末尾为 0 断言没有压力。运行结束时 queue 清空，不代表运行中没有高峰。

## 13.9 Queue 当前值、峰值和等待年龄

三种指标回答不同问题：

| 指标 | 回答 |
|---|---|
| 当前 depth | 此刻积压多少。 |
| sampled peak | 采样点看过的最大积压。 |
| oldest age / queue wait | 工作在队列里等了多久。 |

Queue 可以一直很浅但服务仍不足：若生产和消费都慢，depth 不高，token throughput 已下降。也可以 peak 很高但最终 0，说明短时突发已造成 timeout/admission 损失。

## 13.10 每个节点的压力项

| 节点 | 容量/服务资源 | 压力指标 |
|---|---|---|
| Tokenization/入队 | CPU、waiting queue | input token/s、queue age。 |
| Radix match | Scheduler CPU | match latency、nodes/pages traversed、split count。 |
| L3 query admission | threshold、prefetch cap | attempts、skips by reason。 |
| L2 allocator | free slots、evictable host leaves | alloc requested/granted tokens、retry、shrink。 |
| L3 exists | metadata QPS | pages/s、latency、continuous hit boundary。 |
| L3 read | network/memory bandwidth | bytes/s、page latency、partial failures。 |
| Host publish | Scheduler CPU/tree lock | insert latency、duplicate prefix tokens。 |
| L1 allocator | GPU slots/evictable L1 | requested/granted、eviction retries。 |
| H2D | PCIe/NVLink/load stream | bytes/s、per-layer wait、p99。 |
| Model prefill | GPU compute | recomputed token/s、batch utilization。 |
| D2H | write stream | queued/completed token、ACK lag。 |
| L3 write | backend write service | planned/completed bytes、oldest backup age。 |
| Control drain | collective/Scheduler | revoke/ack/release queue peaks and age。 |

## 13.11 低开销监控原则

高并发主路径不要为每 token：

- 获取全局锁；
- 格式化长日志字符串；
- `.item()` 强制 GPU 同步；
- 遍历整棵树；
- 每事件打印日志；
- 在多个线程更新复杂 Python histogram。

推荐：

1. 操作完成边界用整数 token counter；
2. queue depth 只做轻量采样；
3. 每 10s 在 Scheduler 线程汇总一次；
4. window counter 用 swap/reset，cumulative counter 只单调加；
5. latency 采用采样或固定 bucket；
6. 不在统计路径读取 KV tensor；
7. 跨 rank 已有 collective 中顺便打包必要 counter，避免新增高频同步。

## 13.12 当前主分支已有的指标

### 请求级有效命中来源

当前 `main` 已经在请求第一次统计 cached token 时计算：

```text
device = len(prefix_indices) - host_hit_length
storage = min(host_hit_length, storage_hit_length)
host = host_hit_length - storage
```

结果保存在 `Req.cached_tokens_device/host/storage`，由 `SchedulerOutputStreamer.get_cached_tokens_details()` 放入请求输出。OpenAI 兼容请求可设置扩展字段 `return_cached_tokens_details=true`；仓库 benchmark 的 `--cache-report` 会为对应 SGLang backend 打开这一返回值。示例结果近似为：

```json
{
  "device": 1024,
  "host": 2048,
  "storage": 4096,
  "storage_backend": "MyHiCacheStorage"
}
```

这组数是**最终进入请求缓存前缀的来源归因**，比 backend query hit 更接近有效复用，但仍不是完整 I/O 漏斗。`storage_hit_length` 只记录本请求 prefetch 新增发布到 host tree 的 token；若并发请求抢先插入了同一页，本请求释放重复预留后，那段复用可能归到 `host`，即使数据最初来自 L3。它也不记录 query miss、L2 allocation 失败、read cutoff 或写失败原因。

启用服务参数 `enable_metrics`（CLI 通常为 `--enable-metrics`）后，`TokenizerMetricsCollector` 会把完成请求的这组明细累计到：

```text
sglang:cached_tokens_total{cache_source="device"}
sglang:cached_tokens_total{cache_source="host"}
sglang:cached_tokens_total{cache_source="storage_<backend class>"}
```

### Storage 与 Radix 指标

源码中的 `StorageMetricsCollector` 暴露：

```text
sglang:prefetched_tokens_total
sglang:backuped_tokens_total
sglang:prefetch_pgs
sglang:backup_pgs
sglang:prefetch_bandwidth
sglang:backup_bandwidth
```

`RadixCacheMetricsCollector` 还暴露：

```text
sglang:evicted_tokens_total
sglang:load_back_tokens_total
sglang:eviction_duration_seconds
sglang:load_back_duration_seconds
```

`cached_tokens_total`、prefetch/backup token 以及 evict/load-back token 是 Prometheus Counter；页数、带宽和两项 duration 是 Histogram。它们是 exporter 指标，不会按十秒自动打印成一行日志。Storage 的页数/带宽还依赖具体 backend 的 `get_stats()` 是否返回 `StorageMetrics`，默认接口可返回 `None`。

`check_hicache_events()` 在 metrics 开启时会于每个 scheduler 维护周期调用 backend `get_stats()`，再逐项 observe。HF3FS/Mooncake 的实现返回本轮累积列表后立即 clear。自研后端的 `get_stats()` 也应只做便宜的本地快照/交换，不能在这里发网络 RPC、遍历全量历史，或反复返回不清空的累计 sample list，否则指标本身会拖慢 Scheduler。

口径也要看清：

| 指标 | 当前更准确的含义 | 不代表 |
|---|---|---|
| `cached_tokens_total{cache_source=...}` | 完成请求最终有效 cached token 的来源归因 | L3 尝试、miss、allocation/read/write 损失的完整漏斗 |
| `prefetched_tokens_total` | L3 read 后被 Scheduler 收割并发布到 L2 的 token | 已完成 H2D、最终免算 token |
| `backuped_tokens_total` | L3 backup operation 报告完成的 token | 所有计划页都成功、后端永久不淘汰 |
| `load_back_tokens_total` | `load_back()` 成功分配并提交 H2D 的 token | final CUDA event 已完成 |
| `evicted_tokens_total` | Tree 本次 eviction 处理的 token | 低层副本一定已持久化 |
| `*_duration_seconds` | 对应 Tree 方法的 host 侧调用时长 | 完整 GPU DMA 或 backend RPC 端到端延迟 |

这些现有指标能回答“最终从各层复用了多少”“成功预取/备份了多少”“L1/L2 搬运与淘汰大致怎样”，但不能直接回答：发起多少、backend miss 多少、L2 allocation 丢多少、读写失败多少、queue 等多久。因此分析 cache-hit 下降仍需下一节的漏斗。源码见 [A19c](appendix/A19c-hicache-metrics.md) 和 [A19d](appendix/A19d-cache-hit-reporting.md)。

## 13.13 建议的窗口与累积指标（需要插桩）

### 请求/查询漏斗

```text
attempt_ops/tokens
enqueued_ops/tokens
query_ops/tokens
backend_hit/miss_tokens
```

### L2 admission

```text
allocation_requested/granted/failed_tokens
eviction_retry_tokens
prefetch_shrunk_tokens
prefetch_skip_threshold/rate/host_gap
```

### Read

```text
read_ops/pages/tokens/bytes
read_completed/incomplete/failed_tokens
read_latency / queue_wait
```

### Load-back

```text
L1_requested/granted/failed_tokens
H2D_enqueued/completed_tokens
per-layer wait / total H2D latency
```

### Write

```text
D2H_attempted/enqueued/completed_tokens
L3_write_planned/completed/failed_tokens
backup_queue_wait
host_ref_tokens_held_by_backup
ACK_lag_tokens
```

### 容量快照/峰值

```text
L1 free/evictable/protected
L2 free/resident/prefetch_reserved/backup_protected
ongoing operation counts/tokens
queue current/peak/oldest_age
```

窗口值看瞬时阶段，累积值看整次运行。二者都要保留，且明确单位是 op、page、token 还是 byte。

这些名称是一份建议的数据字典，不代表当前源码每项都已有 counter。实现时优先复用 operation 完成边界上的整数累加，避免为了补齐整张表而在每 token、每页或每 scheduler step 遍历树和 tensor。

## 13.14 关键比率

```text
enqueue_pct = enqueued_tokens / attempt_tokens
backend_hit_pct = backend_hit_tokens / queried_tokens
L2_allocation_pct = granted_tokens / backend_hit_tokens
read_completion_pct = read_completed_tokens / granted_tokens
L1_load_pct = H2D_granted_tokens / host_hit_tokens
end_to_end_pct = effective_reused_tokens / attempt_tokens
write_completion_pct = L3_completed_tokens / L3_planned_tokens
```

分母为 0 时输出 `N/A` 比输出 0% 更准确，否则容易把“未发生”误判为“全部失败”。

## 13.15 L2 流式传输能解决什么

当前经典 L3 prefetch 倾向先为候选范围预留 L2，再查询/裁剪/读取。流式方案可以使用小 ring/chunk：

```text
L3 chunk 0 -> L2 buffer slot A -> H2D -> slot A reusable
L3 chunk 1 -> L2 buffer slot B -> H2D -> slot B reusable
...
```

收益：

- 在途 L2 从“整个命中前缀”降到固定窗口；
- L2 容量不再直接限制单请求可读取长度；
- 高并发下可按 chunk 做公平调度；
- 更接近 L1 直接经 buffer 获取 L3。

但它不会消除：

- L1 必须容纳最终上下文 KV；
- L3/H2D 带宽上限；
- hash query 和跨 rank tail；
- write staging 的 L2 需求；
- Radix 节点只在完整 page/chunk 可见的提交协议；
- abort、重试、顺序和双缓冲状态机复杂度。

要做到正确流式，需要每 chunk 独立的 allocate/read/H2D/event/commit/release 状态，并保证请求在模型读取某层/某页前数据已完成。它提高容量利用率，不保证在任意高并发下达到理论命中极限。

## 13.16 容量规划近似

L2 最低需求可粗分：

```text
C2 >= resident_reuse_working_set
    + peak_prefetch_inflight
    + peak_D2H_inflight
    + peak_L3_write_protected
    + auxiliary_pool_reserve
    + fragmentation/headroom
```

若目标把 L2 主要当传输 buffer，则 resident 项下降，但 L3 read QPS/bytes 会明显增加，后端必须承受更高重复读取率。

## 13.17 调优顺序

1. 先验证正确性和完整写入，不能用丢数据换命中表象。
2. 建立 token 漏斗，定位最大损失阶段。
3. 若 backend hit 低，查写完成、key namespace、reuse mix、L3 retention；尤其确认请求级 `extra_key` 是否只隔离了本地树，却没有进入 L3 namespace。
4. 若 L2 allocation 低，查占用分解和 host ref age，再考虑容量/流式化。
5. 若 read completion 低，查 L3 tail、timeout、batch 和错误。
6. 若 L1 load 低，查 GPU 容量、调度预算和 eviction locks。
7. 若各阶段成功但吞吐下降，查 D2H/H2D 与模型争 GPU/PCIe、Scheduler CPU。
8. 最后调 threshold、timeout、block quota 等策略参数。

## 13.18 基准测试方法

每组至少固定：

- 模型、dtype、TP/PP/CP；
- page size、L1/L2 size、layout/backend；
- 数据集和 seed；
- request rate、max parallel、轮数；
- L3 初始是否 warm；
- 预热/测量时长；
- 写策略和 prefetch policy。

报告：

- 内容层理论可复用 token；
- 各层 token 漏斗；
- TTFT p50/p95/p99；
- output throughput；
- L3 read/write bandwidth 与 tail；
- L1/L2 peak occupancy；
- 输出正确性。

只报告最终 cache-hit 会掩盖“少算了但更慢”或“命中高但写入未保障”的问题。

下一章：[故障定位与断言分析](14-debugging.md)。

# 11. L3 存储接口与自研后端契约

本章尤其适合对接自研分布式内存缓存。完整接口和文件后端代码见 [A10](appendix/A10-storage-contract.md)，工厂见 [A11](appendix/A11-storage-factory.md)。

## 11.1 Controller 眼中的 L3

Controller 把 L3 当成按 page hash 寻址的同步接口，但在专用 Python 后台线程中调用，所以不会直接阻塞 Scheduler：

```text
batch_exists: 哪些连续页存在
batch_get:    把这些页填入指定 L2 地址
batch_set:    从指定 L2 地址写出这些页
```

“同步接口”指一次方法返回时结果已可判定；后端内部可以异步实现，但必须在返回前满足接口的完成/可见性语义，除非它另有被 Controller 正确等待的完成机制。

## 11.2 `HiCacheStorageConfig`

后端初始化收到：

| 字段 | 用途 |
|---|---|
| `tp_rank/tp_size` | 区分 TP shard，或决定哪一 rank 写。 |
| `pp_rank/pp_size` | 区分 pipeline stage 层范围。 |
| `attn_cp_rank/size` | 区分 context-parallel KV slice。 |
| `is_mla_model` | 选择数据结构和 rank 写策略。 |
| `is_page_first_layout` | 判断地址/页组织。 |
| `model_name` | namespace，避免不同模型键冲突。 |
| `tp_lcm_size/should_split_heads` | 异构 TP/head 分片。 |
| `extra_config` | 后端地址、容量、超时和私有参数。 |

自研后端不能只用 page hash 作为全局 key。至少还应编码模型/版本、rank shard、PP stage、CP slice、KV layout/shape 等会改变字节语义的信息，或由独立 namespace 隔离。

### 一个必须单独处理的 namespace 边界

当前原生链式 hash 的直接输入只有：本页 token ids、上一页 digest，以及 bigram/page 分组方式。`RadixKey.extra_key` 会参与本地 `children` 查找和 key 兼容性，但 `RadixKey.hash_page()` 调用 `get_hash_str()` 时没有把它传入；Controller 的 storage operation 也不会把每个请求的 `extra_key` 交给 backend。

因此 L3 的实际键应理解为：

```text
backend_namespace(model/rank/layout/version/...)
  + native_chained_page_hash(token_ids, prior_digest)
```

不能理解为 `hash(extra_key, token_ids, ...)`。如果 LoRA、`cache_salt`、多模态输入或其他请求级语义会改变 KV，仅靠本地 `extra_key` 隔离不足以保证 L3 数据正确。后端也无法凭现有参数自行还原每请求 `extra_key`。可选方案是按服务实例/adapter 建独立 backend namespace、禁止这些请求跨 L3 复用，或扩展 hash/storage 协议并做版本迁移；不能假设 `model_name` 自动覆盖所有动态语义。

## 11.3 V1、V2 与旧接口

### 旧 generic 接口

```python
batch_get(keys, target_locations=None, target_sizes=None) -> list[tensor | None] | int
batch_set(keys, values=None, target_locations=None, target_sizes=None) -> bool
batch_exists(keys, extra_info=None) -> int
```

Controller 可能为每页创建 dummy tensor，再由 host pool `set_from_flat_data_page()` 写入。

### V1 zero-copy 接口

```python
batch_get_v1(keys, host_indices, extra_info=None) -> list[bool]
batch_set_v1(keys, host_indices, extra_info=None) -> list[bool]
```

后端已注册 host pool，可根据 indices 获取页指针，省去中间 tensor。

### V2 multi-pool 接口

```python
batch_exists_v2(keys, pool_transfers=None, extra_info=None) -> PoolTransferResult
batch_get_v2(transfers, extra_info=None) -> dict[pool_name, list[bool]]
batch_set_v2(transfers, extra_info=None) -> dict[pool_name, list[bool]]
```

它让一次 operation 同时携带 KV、Mamba、SWA、indexer、draft 等 pool。

Controller 根据 backend 类型、layout 和功能选择实际函数。自研动态后端必须确认运行时会走哪个版本，不能只实现一个从未被选中的方法。

## 11.4 `PoolTransfer` 字段

```text
name:             pool 名称
host_indices:     L2 slots
device_indices:   L1 slots（L1/L2 复合搬运时）
keys:             此 pool 的 L3 keys
hit_policy:       all_pages / trailing_pages
nodes_to_load:    本次覆盖的树节点
indices_from_pool: sidecar 复用哪个真实 pool 的 indices
```

Sidecar pool 可能不独立分配 indices，而是与 KV/SWA/Mamba pool 使用同样页位置。后端仍必须把组件数据放到注册的正确 host pool。

## 11.5 最长连续前缀语义

`batch_exists(keys)` 返回的是从 `keys[0]` 开始连续存在的页数：

```text
exists: [true, true, false, true, true]
return: 2
```

不能返回 4，也不能返回离散 bitmap 给未适配的调用者。第 3 页缺失后，后面页对本请求不再构成连续可复用上下文。

V2 的 `PoolTransferResult.kv_hit_pages` 同样是最终可用前缀页数。

## 11.6 多 pool 命中策略

### `ALL_PAGES`

辅助 pool 在 KV 前缀的每页都必须存在。任一缺失位置截断最终前缀。适合每 token 都依赖的 DSA/indexer 数据。

### `TRAILING_PAGES`

只要求候选 KV 前缀末尾若干页存在，适合只保存尾部状态的 Mamba/SWA 类组件。后端需要从最长 KV 前缀向前检查 trailing 范围，找到满足条件的最大 prefix length。

最终长度：

```text
final_pages = min(KV prefix, every required pool's usable boundary)
```

主 KV 命中 100 页而 Mamba 状态只支持 80 页，必须返回 80。

## 11.7 Read 返回契约

`batch_get_v2` 返回每个 pool 每个 entry 的 bool：

```python
{
    "kv": [True, True, False],
    "mamba": [True],
}
```

要求：

1. `True` 返回前，对应 host page 字节已经写完且对后续 H2D 可见；
2. 返回列表长度与 transfer keys 一致；
3. 不存在、校验失败、短读、timeout 都返回 False；
4. 发生部分成功时准确标出，不能总返回全 True；
5. 不写成功的目标页不能被计入 completed token；
6. 异常要么转成失败结果，要么让 Controller 明确处理，不能吞掉后返回成功。

## 11.8 Write 返回契约

`batch_set_v2` 同样返回 per-entry success。一个页被判定成功，应至少意味着：

- 后端已经接收完整字节；
- 之后的 `exists` 不会长期返回假阴性；
- 后续 `get` 能读到同一完整版本；
- 如果后端承诺跨实例复用，数据已达到相应可见/复制级别；
- host source buffer 在方法返回后可由 HiCache 释放或重用。

如果方法只是把指针交给后端异步队列就返回 True，而后端稍后才读 host buffer，HiCache 可能已复用该 slot，造成静默数据损坏。此时后端必须在返回前完成读取，或扩展协议把 completion 反馈给 Controller 并延长 host ref。

### 当前经典 Controller 对部分成功的限制

后端接口可以返回逐页结果，但当前经典 `_page_backup()` 仍按 `STORAGE_BATCH_SIZE` 处理一个 batch，并把每批结果折叠成单个 bool：

- generic `batch_set()` 直接返回一个 bool；
- zero-copy V1 的 `list[bool]` 由 `_page_set_zero_copy()` 用 `all(...)` 折叠；
- 任一页失败，当前 batch 的 `completed_tokens` 一页也不增加，并停止后续 batch；
- 源码留有 `todo: allow partial success`。

所以“日志显示这一批全部失败”不一定等于后端物理上一页都没写进去；它表示 HiCache 控制面没有确认这批形成可安全依赖的连续已完成前缀。自研后端仍应准确返回逐页状态，便于未来细化和后端自身观测，但现阶段容量/命中分析要同时查看 backend 的 physical success pages 与 Controller 的 conservative completed tokens。

## 11.9 `exists`、`set`、`get` 的一致性

自研分布式内存 L3 至少明确：

### Read-after-write

`set` 成功返回后，同一 namespace 的 `exists/get` 何时可见？HiCache 默认倾向把成功当成后续可读取。

### 幂等写

相同 content hash 重复 `set` 应安全。不同请求可能并发写同一页。推荐：

- 内容相同直接视为成功；
- 原子发布完整值；
- 不让读者看到半写状态。

### 删除/TTL/后端淘汰

如果 L3 有 TTL 或容量淘汰，metadata cache 的正命中不能无限期相信旧状态。`HiCacheFile.MetadataCache` 支持 TTL，就是为避免永久缓存过期存在性。

`CacheInitParams.cache_ttl_seconds` 虽然存在于公共初始化参数中，但在本书固定的源码快照里没有运行时读取者，不能把它当成已启用的 Radix/L3 TTL。后端 TTL、文件后端 metadata TTL 和本地树淘汰是三套不同机制。

### 校验

content hash 是 key，但仍建议存长度、dtype/layout/version checksum，读取时验证短读和版本不符。

## 11.10 页字节大小与地址

Generic 后端可从 flat page tensor 得到大小。Zero-copy 后端通过注册的 host pool 和 `get_page_buffer_meta()` 获得：

```text
[(address_0, bytes_0), (address_1, bytes_1), ...]
```

注意：

- layer-first 一页可能对应多段地址；
- page-first 通常 K/V 两段；
- K-only 只有一段；
- 非对称 K/V 两段大小不同；
- page-head 可能按 head split；
- 页起始 indices 必须按 `page_size` 对齐；
- 外部 RDMA 注册区必须覆盖完整 host buffer 生命周期。

## 11.11 `HiCacheFile` 参考实现

文件后端虽然不是高性能方案，但清楚展示契约：

### Key namespace

文件名 suffix 包含模型和 rank 配置；CP 多 rank 时加入 CP suffix，避免不同 slice 互相覆盖。

### 原子写

`set()`：

1. 已存在则 touch LRU 并跳过重写；
2. evictor 预留容量；
3. 写唯一临时文件；
4. `os.replace()` 原子发布；
5. commit LRU；
6. 异常时 abort 预留、删除 temp、清 metadata。

### 防短读

`get()` 计算 expected bytes，`readinto()` 返回不足时抛错，不把半页当成功。

### V2 多组件

组件 key 使用 `{page_hash}.{pool_name}`，`batch_exists_v2` 同时检查主 KV 与 auxiliary component，并计算共同前缀。

## 11.12 Factory 与动态后端

内置后端在 `StorageBackendFactory` 注册 module/class，创建时延迟 import，避免无关依赖影响启动。

动态后端 extra config 需要：

```json
{
  "backend_name": "my_distributed_memory",
  "module_path": "my_package.hicache_backend",
  "class_name": "MyHiCacheStorage"
}
```

类必须继承 `HiCacheStorage`。动态构造参数路径与内置后端并不完全相同，接入前应写最小启动测试验证收到的 `storage_config/kwargs`。

## 11.13 自研分布式内存后端检查清单

### 功能正确性

1. 同一个 page hash 在所有客户端使用完全相同的 namespace 规则。
2. TP/PP/CP shard 不会 key 冲突。
3. `batch_exists` 返回最长连续前缀，不是命中总数。
4. `batch_get/set` 的结果长度和顺序稳定。
5. 部分失败准确返回。
6. `set=True` 后 source buffer 可立即安全重用。
7. `get=True` 前 target buffer 已完整可见。
8. 并发同 key 写幂等且原子。
9. 短读、断连、server restart 不会返回成功。
10. clear/TTL/eviction 与 metadata cache 一致。

### 性能

1. 单页 bytes 和 128-page batch bytes 是否匹配网络最佳消息大小。
2. exists 是否能真正 batch，而不是内部逐 key RPC。
3. read/write 是否分独立连接池或限流，避免写压饿死读。
4. 是否支持 scatter/gather，避免 page-first host 数据再复制。
5. RDMA registration 是否启动时完成，而不是每 I/O 注册。
6. timeout、重试和副本切换是否有上限。
7. 后端 queue depth、active ops、bytes/s、p50/p99/p999 是否可观测。
8. 最慢 shard 是否单独暴露，不能只报集群平均。

### 生命周期

1. attach/detach 可重复执行，无线程和连接泄漏。
2. detach 会等待或取消所有仍使用 host pointer 的操作。
3. backend callback 不在 HiCache 已释放的 buffer 上继续 DMA。
4. fork/multiprocess/rank 初始化顺序正确。
5. shutdown 异常路径不会让 server 永久 hang。

## 11.14 建议的后端单元测试矩阵

| 场景 | 期望 |
|---|---|
| 空 keys | 返回空成功结果，不发 RPC。 |
| 第一页 miss | continuous hit=0。 |
| 中间页 miss | 截到 miss 前。 |
| 全 hit | 返回全部页。 |
| read 第 N 页失败 | 前 N-1 页成功，后续不计完成。 |
| write 部分失败 | 后端精确返回 per-page false；集成测试同时确认经典 Controller 当前按失败 batch 保守截断。 |
| 同 key 并发写 | 最终完整且一致。 |
| set 后立即 exists/get | 满足承诺的一致性。 |
| target/source slot 随后重用 | 后端不再访问旧 pointer。 |
| TP rank namespace | 各 shard 不覆盖。 |
| auxiliary pool 缺页 | V2 final prefix 正确收缩。 |
| timeout/retry | 不无限等待、不重复回调。 |
| detach in-flight | 明确 drain/cancel，资源无泄漏。 |

## 11.15 后端指标应怎样映射到 HiCache

后端至少提供：

```text
exists_ops/pages/hit_pages/latency
get_ops/pages/success_pages/bytes/latency
set_ops/pages/success_pages/bytes/latency
queue_depth / active_ops
timeouts / retries / capacity_rejects / connection_errors
server/shard label
```

HiCache 再把它们与：

```text
prefetch attempts -> L2 allocation -> backend hit -> read complete -> H2D complete
write attempts -> D2H complete -> set complete
```

关联。只有后端指标而没有 Tree/Controller 的 token 漏斗，无法判断 cache-hit 下降是不是 L3 本身造成。

下一章：[UnifiedRadixCache 与混合模型扩展](12-unified-and-hybrid.md)。

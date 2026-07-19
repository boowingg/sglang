# 17. 术语表

## 模型与推理

| 术语 | 通俗解释 |
|---|---|
| Token | 模型处理文本的离散编号单位，不等同于汉字或单词。 |
| Prefill / Extend | 批量计算输入中尚无 KV 的 token。 |
| Decode | 使用历史 KV，每一步生成新 token。 |
| Attention | 当前 Query 根据历史 Key/Value 聚合上下文的计算。 |
| MHA | Multi-Head Attention，多头注意力。 |
| GQA/MQA | 减少 KV heads 的注意力变体，降低 KV Cache 大小。 |
| MLA | Multi-head Latent Attention，用压缩 latent 表示降低 KV 占用。 |
| SWA | Sliding Window Attention，只关注有限窗口。 |
| Mamba/SSM | 使用 recurrent/state-space 状态的模型组件，不等同于普通 KV。 |
| Speculative decoding | Draft 模型先猜 token，Target 模型验证；可有独立 draft KV。 |

## 缓存层级

| 术语 | 通俗解释 |
|---|---|
| L1 | GPU KV pool，可直接被模型读取。 |
| L2 | CPU host KV pool，是 GPU 与 L3 之间的缓存/传输层。 |
| L3 | 文件、远端内存或分布式存储。 |
| Hit | 找到可复用内容；必须说明是 query、read、load-back 还是最终有效命中。 |
| Miss | 某一层无法提供连续可复用页，不一定表示其他层也没有。 |
| Admission | 为命中数据成功取得目标层 slots。 |
| Promotion / load-back | 从低层恢复到高层，HiCache 中常指 L2 -> L1。 |
| Demotion | 释放高层副本，保留低层副本，例如 L1 -> host-only。 |
| Write-through | 较早把 L1 数据备份到 L2。 |
| Write-back | 淘汰 L1 时才备份。 |
| Resident | 长时间保留在某层、不是仅用于在途传输的数据。 |

## Radix Tree

| 术语 | 通俗解释 |
|---|---|
| Prefix | 从 token 序列开头起连续相同的部分。 |
| Radix Tree | 一条边可存一段 key 的压缩前缀树。 |
| Root | 空 key 的树根，不对应真实 KV。 |
| Node | 保存一段 token 与各层索引/状态的元数据对象。 |
| Edge key | parent 到 node 这段 token，即 `node.key`。 |
| Split | 匹配在压缩边中间结束时，把节点拆成公共前缀和后缀。 |
| Structural leaf | `children` 为空的真正树叶。 |
| Device leaf | 在 L1 视角下没有仍驻留 L1 的后代，可能仍有 host-only children。 |
| Host-tier leaf | 在 L2 视角下没有仍带 host 副本的后代，结构上仍可能有无 host 数据的 children。 |
| Tombstone | 某组件数据已移除但仍为其他组件/后代保留结构的节点。 |
| `value` | 节点对应的 L1 slot indices。 |
| `host_value` | 节点对应的 L2 slot indices。 |
| `hash_value` | 节点每页的链式 L3 hash。 |
| `backuped` | 当前实现等价于 `host_value is not None`；pending D2H 时也为真，不单独证明 L2 字节已完成。 |

## 内存与布局

| 术语 | 通俗解释 |
|---|---|
| Slot | Pool 中可存一个逻辑 token 状态的位置。 |
| Index tensor | 一串 slot 编号，不是 KV 字节。 |
| `req_pool_idx` | 请求在 `ReqToTokenPool` 中占用的行号。 |
| `req_to_token` | 从“请求行 + 序列位置”映射到 L1 KV slot id 的二维索引表。 |
| `out_cache_loc` | 当前 batch 为新 token 分配的 L1 slot ids。 |
| `kv_allocated_len` | 请求当前已占有 L1 slots 覆盖到的序列长度，可能大于已完成计算长度。 |
| `kv_committed_len` | Forward 结果已确认有效、可以提交给 Tree 或按结束语义释放的 KV 长度；不表示 Tree 已经接管。 |
| `cache_protected_len` | 已由 Radix Tree 接管、不能再按请求私有 KV 释放的页对齐前缀边界。 |
| Page | 多个 token 组成的分配、hash 和 L3 I/O 单位。 |
| Page alignment | 长度向下取整到完整 page。 |
| Pinned memory | 不可被 OS 换出的 host 内存，可供 GPU DMA 高效访问。 |
| `layer_first` | L2 中按 layer 聚合 token 的布局。 |
| `page_first` | L2 中按 token/page 聚合全部 layer 的布局。 |
| Stride | 相邻逻辑元素地址相差的字节/元素数。 |
| View | 共享同一底层数据、只改变 shape/stride 的 tensor。 |
| Contiguous | 数据按当前 shape 的标准顺序连续排布。 |
| Staging buffer | 在最终目的地前整理 layout 的临时缓冲。 |
| Zero-copy | 后端直接使用已注册 host buffer 地址，避免中间 tensor copy。 |
| O_DIRECT | 绕过页缓存的文件 I/O，通常要求地址/大小对齐。 |

## 异步与并发

| 术语 | 通俗解释 |
|---|---|
| Operation | 描述一次待执行传输的状态对象。 |
| Queue | 在线程间移交 operation/ACK/待释放资源。 |
| CUDA stream | GPU 命令的有序队列，不是 Python 线程。 |
| Event | stream 中的完成标记，用于 query/wait/synchronize。 |
| ACK | 完成通知；需看具体 operation 判断是数据完成还是仅操作结束。 |
| Polling | Scheduler 周期性非阻塞检查事件。 |
| Happens-before | 一个动作完成并对另一个动作可见的顺序保证。 |
| Backpressure | 下游变慢后限制上游继续产生工作。 |
| In-flight | 已分配/排队但尚未最终提交和释放的资源。 |
| Ring slot | 循环复用的多组 event/counter 槽，用于重叠 batch。 |
| `record_stream` | 告诉 PyTorch tensor 仍被某 stream 使用，延迟回收。 |

## 引用与所有权

| 术语 | 通俗解释 |
|---|---|
| Ownership | 当前谁负责保持和最终释放一段 slots。 |
| `lock_ref` | 保护 L1 Radix 路径不被淘汰的计数。 |
| `host_ref_counter` | 保护 L2 slots 不被释放的计数。 |
| `RLock` | 同一线程可重复获取的互斥锁，保护 allocator 等共享状态。 |
| Pending | operation 已开始但控制面尚未提交完成状态。 |
| Publish | 把已完成数据正式挂到树/请求，使后续逻辑可见。 |
| Commit | 确认操作结果并转移所有权，具体语义由调用链决定。 |
| Rollback | 某步骤失败后撤销已分配资源和部分状态。 |
| Double free | 同一 slot 被释放两次。 |
| Leak | slot/ref/operation 永远没有被释放。 |

## 分布式

| 术语 | 通俗解释 |
|---|---|
| TP | Tensor Parallel，同一层参数/KV 切到多个 rank。 |
| PP | Pipeline Parallel，不同层位于不同 stage。 |
| CP | Context Parallel，上下文/KV 按序列维切分。 |
| Rank | 一个分布式进程及其组内编号。 |
| Process group | 参与某组 collective 的 rank 集合。 |
| Collective | all-reduce/barrier 等所有成员按同一顺序参与的操作。 |
| All-reduce MIN | 所有 rank 采用共同最小成功长度。 |
| Barrier | 所有参与 rank 都到达后才继续。 |
| Tail latency | 最慢少量请求/rank 的延迟，常决定共同完成时间。 |

## L3 与内容寻址

| 术语 | 通俗解释 |
|---|---|
| Content-addressed key | 根据 token 内容与上下文 hash 得到的存储键。 |
| Chained hash | 当前页 hash 包含上一页 digest，绑定完整前缀上下文。 |
| Local `extra_key` | 参与 `RadixKey` 兼容性和本地 child key，隔离本地树；当前不会自动进入原生 L3 页 hash。 |
| Backend namespace | 后端在 page hash 之外编码的模型、rank、layout、版本等语义；需要与本地隔离规则共同保证正确性。 |
| Longest contiguous prefix | 从第一页开始连续存在的最大页数。 |
| Idempotent set | 相同 key/value 重复写仍安全。 |
| Read-after-write | 写成功后何时能被 exists/get 看见。 |
| Partial success | 一批页只有部分成功，必须精确反馈。 |
| Durability | 后端成功后数据在何种故障范围内仍可读取。 |
| Metadata cache | 缓存 exists 结果，需考虑 TTL 和后端淘汰。 |
| Sidecar pool | 使用主 pool 索引但保存额外组件数据的 pool。 |

## 性能与指标

| 术语 | 通俗解释 |
|---|---|
| Throughput | 单位时间完成的请求/token/byte。 |
| TTFT | Time To First Token，首 token 延迟。 |
| Bandwidth | 单位时间传输字节数。 |
| Queue depth | 当前等待 operation 数。 |
| Queue age | 最老 operation 已等待多久。 |
| Working set | 某时间窗口内需要保留以获得命中的数据集合。 |
| Reuse distance | 同一前缀两次访问之间插入/访问了多少其他数据。 |
| Churn | 数据频繁插入和淘汰。 |
| Write amplification | 为少量有效复用写出大量一次性数据。 |
| Token funnel | 从候选到最终复用逐阶段统计 token 损失。 |
| Window metric | 固定时间窗口内的增量。 |
| Cumulative metric | 进程启动以来的单调累计值。 |
| p50/p99 | 延迟分位数，p99 反映最慢 1% 附近。 |

返回[全书目录](README.md)。

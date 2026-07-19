# 概念图素材

这两张图由 GPT 图片生成工具创建，只承担章节导览。网页中的 Mermaid 图、状态说明和源码链接才是精确技术表达。

## `hicache-data-journey.jpg`

用途：全书导览中的 L1/L2/L3 空间关系。

生成约束：以工程教科书式等距剖面表现 GPU、host memory 和分布式内存节点；用蓝、绿、珊瑚色 KV pages 表示数据流；无文字、数字、Logo 和人物。

## `radix-prefix-map.jpg`

用途：Radix Tree 章节中的“共享前缀 + 分层索引”概念。

生成约束：用透明分叉通道和 token tiles 表现公共前缀与分支，并让路径指向三种缓存层级；无文字、数字、Logo 和人物。

原始生成结果保留在 Codex 的 `generated_images` 目录；网页使用质量 90 的 JPEG 副本，将两张图从约 4 MB 压缩到约 0.9 MB。

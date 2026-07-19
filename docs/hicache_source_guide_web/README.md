# HiCache 源码剖析交互阅读版

这个目录把 [`../hicache_source_guide`](../hicache_source_guide/README.md) 中的 Markdown 书籍转换为本地交互式网页。Markdown 是唯一正文来源；网页只负责索引、渲染、导航和交互增强，不复制维护第二套书稿。

## 启动

```bash
cd docs/hicache_source_guide_web
npm run dev
```

服务默认监听 `http://127.0.0.1:4173`。端口已被占用时会自动尝试后续端口。项目没有安装步骤，运行时库已经固定在 `vendor/`。

## 页面能力

- 正文、逐函数附录和术语的统一搜索，附录按需载入；
- 章节导航、页内目录、阅读进度、书签和阅读位置恢复；
- Mermaid 图的本地渲染、平移、缩放、全屏查看和主题重绘；
- 函数代码折叠、行号、复制，以及 `源文件:行号` 的源码预览和范围高亮；
- 请求路径、Radix 共享前缀和 L2 并发占页三个交互实验台；
- 桌面三栏、手机抽屉、浅色/深色和专注阅读布局；
- 两张 GPT 生成的概念导览图。精确状态和调用关系仍由源码、Mermaid 与交互组件表达。

## 构建关系

```text
../hicache_source_guide/**/*.md
        │
        ▼
scripts/build-content.mjs
        │
        ├── .generated/content/**/*.md
        └── .generated/book-manifest.json
                    │
                    ▼
         app.js + reader.js + search.js
```

`.generated/` 是可删除的构建产物并已加入 `.gitignore`。`npm run dev` 会在启动服务前重新生成它，因此修改书稿后只需刷新服务。

## 检查

```bash
npm run check
```

检查项包括：

- 所有 Markdown 文件全部进入目录且不重复；
- 每篇恰好一个一级标题，代码围栏完整闭合；
- Markdown 章节内链及目标锚点存在；
- 行号型源码引用的文件存在，且起止行没有超出当前源码快照；
- README、附录覆盖清单的源码快照一致；
- 网页入口、脚本、概念图和本地运行时库齐全；
- 附录覆盖统计能写入网页 manifest。

书稿的逐函数源码一致性仍由原目录中的命令检查：

```bash
python3 docs/hicache_source_guide/tools/generate_function_appendix.py --check
```

## 文件职责

| 文件 | 职责 |
|---|---|
| `app.js` | 路由、章节状态、书签、主题、源码预览和页面装配。 |
| `reader.js` | Markdown、代码块、链接重写和 Mermaid 渲染。 |
| `search.js` | 正文/附录搜索、评分、结果定位和键盘选择。 |
| `interactives.js` | 三个教学实验台和概念图注入。 |
| `styles.css` | 桌面、移动、深色、打印和各交互状态。 |
| `scripts/build-content.mjs` | 从书稿生成内容副本和搜索 manifest。 |
| `scripts/dev-server.mjs` | 零依赖静态服务及仓库内只读源码访问。 |
| `scripts/check-site.mjs` | 内容映射、内链和资源检查。 |
| `assets/` | 网页概念视觉素材及其说明。 |
| `vendor/` | 固定版本的浏览器运行库及许可证。 |

## 设计边界

- GPT 概念图不作为字段、时序或容量结论的证据。
- L2 压力实验台展示并发占页上界，不预测真实运行值；真实值还受命中页数、I/O 完成速度、保护页和 ACK 释放影响。
- `/__repo/` 只由本地开发服务提供，并限制在当前 SGLang 仓库内；普通静态托管不会暴露源码预览接口。
- 网页索引固定读取书稿记录的源码快照，切换源码提交后应先更新并检查书稿。

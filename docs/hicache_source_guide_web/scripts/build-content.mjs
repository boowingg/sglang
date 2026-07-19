import { cp, mkdir, readFile, readdir, rm, stat, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
export const webRoot = path.resolve(scriptDir, "..");
export const repoRoot = path.resolve(webRoot, "../..");
export const sourceRoot = path.resolve(webRoot, "../hicache_source_guide");
export const generatedRoot = path.resolve(webRoot, ".generated");
const generatedContentRoot = path.join(generatedRoot, "content");

const CORE_GROUPS = [
  {
    id: "start",
    title: "从零开始",
    description: "先建立 token、page、index 与三层缓存的共同语言。",
    files: ["README.md", "00-reading-guide.md", "01-foundations.md"],
  },
  {
    id: "structure",
    title: "结构与装配",
    description: "看懂对象如何创建，以及树、池和控制器如何连接。",
    files: [
      "02-architecture.md",
      "03-startup-and-configuration.md",
      "03a-l1-memory-and-request-mapping.md",
      "03b-state-and-data-contracts.md",
      "04-radix-tree.md",
    ],
  },
  {
    id: "lifecycle",
    title: "请求与状态机",
    description: "沿完整请求跟踪读、写、回收和异步所有权。",
    files: [
      "05-request-lifecycle.md",
      "06-read-path.md",
      "07-write-path.md",
      "08-eviction-and-ownership.md",
      "09-concurrency.md",
    ],
  },
  {
    id: "data-plane",
    title: "数据面与后端",
    description: "深入 host layout、搬运内核、L3 契约与混合模型。",
    files: [
      "10-host-layout-and-kernels.md",
      "11-storage-backends.md",
      "12-unified-and-hybrid.md",
    ],
  },
  {
    id: "practice",
    title: "性能与实战",
    description: "把心智模型用于指标分析、排障与源码练习。",
    files: [
      "13-performance.md",
      "14-debugging.md",
      "15-function-map.md",
      "16-reading-labs.md",
      "17-glossary.md",
    ],
  },
];

const collator = new Intl.Collator("zh-CN", {
  numeric: true,
  sensitivity: "base",
});

function normalizePath(filePath) {
  return filePath.split(path.sep).join("/");
}

async function walkMarkdown(directory, prefix = "") {
  const entries = await readdir(directory, { withFileTypes: true });
  const result = [];
  for (const entry of entries.sort((a, b) => collator.compare(a.name, b.name))) {
    const relative = normalizePath(path.join(prefix, entry.name));
    if (entry.isDirectory()) {
      result.push(...(await walkMarkdown(path.join(directory, entry.name), relative)));
    } else if (entry.isFile() && entry.name.endsWith(".md")) {
      result.push(relative);
    }
  }
  return result;
}

function plainInline(text) {
  return text
    .replace(/!\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(/\[([^\]]+)\]\([^)]*\)/g, "$1")
    .replace(/<[^>]+>/g, "")
    .replace(/[`*~]/g, "")
    .replace(/\\([\\`*{}\[\]()#+.!_>-])/g, "$1")
    .trim();
}

export function slugifyHeading(value) {
  return plainInline(value)
    .normalize("NFKC")
    .toLowerCase()
    .replace(/&[a-z]+;/g, "")
    .replace(/[^\p{L}\p{N}\p{M}\s_-]+/gu, "")
    .replace(/\s+/g, "-")
    .replace(/-+/g, "-")
    .replace(/^-+|-+$/g, "") || "section";
}

function extractHeadings(markdown) {
  const headings = [];
  const slugCounts = new Map();
  let fence = null;

  for (const line of markdown.split("\n")) {
    const fenceMatch = line.match(/^\s*(```+|~~~+)/);
    if (fenceMatch) {
      const marker = fenceMatch[1][0];
      fence = fence === marker ? null : fence || marker;
      continue;
    }
    if (fence) continue;

    const match = line.match(/^(#{1,4})\s+(.+?)\s*#*\s*$/);
    if (!match) continue;
    const title = plainInline(match[2]);
    const base = slugifyHeading(title);
    const count = slugCounts.get(base) || 0;
    slugCounts.set(base, count + 1);
    headings.push({
      level: match[1].length,
      title,
      slug: count === 0 ? base : `${base}-${count + 1}`,
    });
  }
  return headings;
}

function extractSummary(markdown) {
  const lines = markdown.split("\n");
  let fence = null;
  let paragraph = [];

  for (const line of lines.slice(1)) {
    const fenceMatch = line.match(/^\s*(```+|~~~+)/);
    if (fenceMatch) {
      const marker = fenceMatch[1][0];
      fence = fence === marker ? null : fence || marker;
      continue;
    }
    if (fence) continue;
    const trimmed = line.trim();
    if (!trimmed) {
      if (paragraph.length) break;
      continue;
    }
    if (/^(#|>|\||[-*+]\s|\d+\.\s)/.test(trimmed)) {
      if (paragraph.length) break;
      continue;
    }
    paragraph.push(trimmed);
  }

  const summary = plainInline(paragraph.join(" "));
  return summary.length > 180 ? `${summary.slice(0, 177)}...` : summary;
}

function extractSearchText(markdown, isAppendix) {
  const headings = extractHeadings(markdown).map((heading) => heading.title);
  const codeTerms = [...markdown.matchAll(/`([^`\n]{2,100})`/g)].map((match) => match[1]);
  const uniqueTerms = [...new Set(codeTerms)].slice(0, isAppendix ? 1200 : 500);
  if (isAppendix) {
    return [...headings, ...uniqueTerms].join(" ").normalize("NFKC").toLowerCase();
  }

  const prose = markdown
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/~~~[\s\S]*?~~~/g, " ")
    .replace(/<[^>]+>/g, " ")
    .replace(/[|#>*_~\[\]()`]/g, " ")
    .replace(/\s+/g, " ");
  return `${prose} ${uniqueTerms.join(" ")}`.normalize("NFKC").toLowerCase();
}

function countCodeLines(markdown) {
  let fence = null;
  let count = 0;
  for (const line of markdown.split("\n")) {
    const fenceMatch = line.match(/^\s*(```+|~~~+)/);
    if (fenceMatch) {
      const marker = fenceMatch[1][0];
      fence = fence === marker ? null : fence || marker;
      continue;
    }
    if (fence) count += 1;
  }
  return count;
}

function cleanDisplayTitle(title, filePath) {
  if (filePath === "README.md") return "全书导览";
  return title
    .replace(/^HiCache\s+源码剖析[:：]\s*/i, "")
    .replace(/^附录\s*/i, "")
    .trim();
}

function buildGroups(allFiles) {
  const knownCore = new Set(CORE_GROUPS.flatMap((group) => group.files));
  const unknownCore = allFiles.filter(
    (file) => !file.startsWith("appendix/") && !knownCore.has(file),
  );
  if (unknownCore.length) {
    throw new Error(`核心章节未分组: ${unknownCore.join(", ")}`);
  }

  const appendixFiles = allFiles
    .filter((file) => file.startsWith("appendix/") && file !== "appendix/README.md")
    .sort(collator.compare);

  return [
    ...CORE_GROUPS,
    {
      id: "appendix",
      title: "逐函数附录",
      description: "按源文件查阅完整代码、状态读写和直接调用。",
      files: ["appendix/README.md", ...appendixFiles],
    },
  ];
}

export async function buildBook() {
  const allFiles = await walkMarkdown(sourceRoot);
  const groups = buildGroups(allFiles);
  const orderedFiles = groups.flatMap((group) => group.files);

  if (new Set(orderedFiles).size !== allFiles.length) {
    throw new Error("章节分组存在重复或遗漏");
  }

  await rm(generatedContentRoot, { recursive: true, force: true });
  await mkdir(generatedContentRoot, { recursive: true });

  const documents = [];
  for (const [index, filePath] of orderedFiles.entries()) {
    const absoluteSource = path.join(sourceRoot, filePath);
    const markdown = await readFile(absoluteSource, "utf8");
    const sourceStats = await stat(absoluteSource);
    const headings = extractHeadings(markdown);
    const title = headings.find((heading) => heading.level === 1)?.title || filePath;
    const isAppendix = filePath.startsWith("appendix/");
    const group = groups.find((candidate) => candidate.files.includes(filePath));
    const codeLines = countCodeLines(markdown);
    const wordLikeCount = markdown
      .replace(/```[\s\S]*?```/g, " ")
      .replace(/\s+/g, "")
      .length;
    const destination = path.join(generatedContentRoot, filePath);
    await mkdir(path.dirname(destination), { recursive: true });
    await cp(absoluteSource, destination);

    documents.push({
      id: filePath.replace(/\.md$/, ""),
      path: filePath,
      contentUrl: `./.generated/content/${filePath}`,
      sourcePath: normalizePath(path.relative(repoRoot, absoluteSource)),
      title,
      displayTitle: cleanDisplayTitle(title, filePath),
      summary: extractSummary(markdown),
      headings,
      groupId: group.id,
      isAppendix,
      order: index,
      stats: {
        proseCharacters: wordLikeCount,
        codeLines,
        diagrams: (markdown.match(/```mermaid\s/g) || []).length,
        estimatedMinutes: Math.max(1, Math.ceil(wordLikeCount / 500)),
      },
      searchText: extractSearchText(markdown, isAppendix),
      modifiedAt: sourceStats.mtime.toISOString(),
    });
  }

  const intro = await readFile(path.join(sourceRoot, "README.md"), "utf8");
  const coverage = JSON.parse(
    await readFile(path.join(sourceRoot, "appendix/coverage.json"), "utf8"),
  );
  const snapshot =
    coverage.revision || intro.match(/main@([0-9a-f]{7,40})/)?.[1] || "unknown";
  const manifest = {
    schemaVersion: 1,
    generatedAt: new Date().toISOString(),
    sourceSnapshot: snapshot,
    totals: {
      documents: documents.length,
      coreDocuments: documents.filter((doc) => !doc.isAppendix).length,
      appendixDocuments: documents.filter((doc) => doc.isAppendix).length,
      diagrams: documents.reduce((sum, doc) => sum + doc.stats.diagrams, 0),
      codeLines: documents.reduce((sum, doc) => sum + doc.stats.codeLines, 0),
      functions: coverage.python_function_total,
      classes: coverage.python_class_total,
      coveredSourceEntries: coverage.entries.length,
    },
    conceptAssets: {
      dataJourney: "./assets/hicache-data-journey.jpg",
      radixMap: "./assets/radix-prefix-map.jpg",
    },
    groups: groups.map(({ files, ...group }) => ({
      ...group,
      documentIds: files.map((file) => file.replace(/\.md$/, "")),
    })),
    documents,
  };

  await writeFile(
    path.join(generatedRoot, "book-manifest.json"),
    `${JSON.stringify(manifest)}\n`,
    "utf8",
  );
  return manifest;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const manifest = await buildBook();
  console.log(
    `Built ${manifest.totals.documents} documents, ${manifest.totals.diagrams} diagrams, ` +
      `${manifest.totals.codeLines} code lines.`,
  );
}

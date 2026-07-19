import { access, readFile } from "node:fs/promises";
import path from "node:path";
import {
  buildBook,
  repoRoot,
  sourceRoot,
  webRoot,
} from "./build-content.mjs";

const manifest = await buildBook();
const errors = [];
const documentPaths = new Set(manifest.documents.map((document) => document.path));
const documentsByPath = new Map(
  manifest.documents.map((document) => [document.path, document]),
);
const sourceLineCounts = new Map();
const sourceReferencePattern = /`((?:python|sgl-kernel|test|benchmark|examples)\/[^`\n:]+\.(?:py|c|cc|cpp|cu|cuh|h|hpp|hip|rs|go|java|ts|js)):(\d+)(?:-(\d+))?`/g;

function scanMarkdown(markdown) {
  const visibleLines = [];
  let fence = null;

  for (const [index, line] of markdown.split("\n").entries()) {
    const match = line.match(/^\s*(`{3,}|~{3,})/);
    if (!match) {
      if (!fence) visibleLines.push(line);
      continue;
    }

    const marker = match[1];
    if (!fence) {
      fence = { character: marker[0], length: marker.length, line: index + 1 };
    } else if (marker[0] === fence.character && marker.length >= fence.length) {
      fence = null;
    }
  }

  return { visibleMarkdown: visibleLines.join("\n"), unclosedFence: fence };
}

function decodeAnchor(anchor, context) {
  try {
    return decodeURIComponent(anchor).toLowerCase();
  } catch {
    errors.push(`${context}: 无法解码锚点 #${anchor}`);
    return null;
  }
}

async function getSourceLineCount(sourcePath) {
  if (sourceLineCounts.has(sourcePath)) return sourceLineCounts.get(sourcePath);
  try {
    const source = await readFile(path.join(repoRoot, sourcePath), "utf8");
    const count = source.split(/\r?\n/).length;
    sourceLineCounts.set(sourcePath, count);
    return count;
  } catch {
    sourceLineCounts.set(sourcePath, null);
    return null;
  }
}

for (const document of manifest.documents) {
  const markdown = await readFile(path.join(sourceRoot, document.path), "utf8");
  const { visibleMarkdown, unclosedFence } = scanMarkdown(markdown);
  if (unclosedFence) {
    errors.push(
      `${document.path}:${unclosedFence.line}: ${unclosedFence.character.repeat(unclosedFence.length)} 围栏未闭合`,
    );
  }

  const h1Count = document.headings.filter((heading) => heading.level === 1).length;
  if (h1Count !== 1) {
    errors.push(`${document.path}: 应恰好包含一个一级标题，实际为 ${h1Count}`);
  }

  const links = [
    ...visibleMarkdown.matchAll(/\[[^\]]*\]\(([^)\s]+)(?:\s+"[^"]*")?\)/g),
  ];
  for (const [, href] of links) {
    if (/^(?:https?:|mailto:)/.test(href)) continue;
    const [target, rawAnchor] = href.split("#", 2);
    let resolved = document.path;
    if (target) {
      if (!target.endsWith(".md")) continue;
      try {
        resolved = path
          .normalize(path.join(path.dirname(document.path), decodeURIComponent(target)))
          .split(path.sep)
          .join("/");
      } catch {
        errors.push(`${document.path}: 无法解码链接 ${href}`);
        continue;
      }
    }

    if (!documentPaths.has(resolved)) {
      errors.push(`${document.path}: 找不到 ${href}`);
      continue;
    }

    if (rawAnchor) {
      const anchor = decodeAnchor(rawAnchor, document.path);
      if (anchor === null) continue;
      const targetDocument = documentsByPath.get(resolved);
      if (!targetDocument.headings.some((heading) => heading.slug === anchor)) {
        errors.push(`${document.path}: 找不到 ${resolved}#${rawAnchor} 对应标题`);
      }
    }
  }

  for (const match of visibleMarkdown.matchAll(sourceReferencePattern)) {
    const [, sourcePath, rawStart, rawEnd] = match;
    const start = Number(rawStart);
    const end = Number(rawEnd || rawStart);
    const lineCount = await getSourceLineCount(sourcePath);
    if (lineCount === null) {
      errors.push(`${document.path}: 源码引用 ${sourcePath}:${start}-${end} 的文件不存在`);
    } else if (start > end || end > lineCount) {
      errors.push(
        `${document.path}: 源码引用 ${sourcePath}:${start}-${end} 超出文件范围 1-${lineCount}`,
      );
    }
  }
}

for (const required of [
  "index.html",
  "styles.css",
  "app.js",
  "assets/hicache-data-journey.jpg",
  "assets/radix-prefix-map.jpg",
  "vendor/marked.esm.js",
  "vendor/lucide.min.js",
  "vendor/mermaid.min.js",
]) {
  try {
    await access(path.join(webRoot, required));
  } catch {
    errors.push(`网页缺少 ${required}`);
  }
}

const sourcePaths = new Set(manifest.documents.map((document) => document.sourcePath));
if (sourcePaths.size !== manifest.documents.length) {
  errors.push("manifest 中存在重复源码路径");
}

const radixAppendix = manifest.documents.find(
  (document) => document.id === "appendix/A03-radix-cache",
);
if (!radixAppendix?.headings.some((heading) => heading.title === "RadixCache.cache_finished_req")) {
  errors.push("函数标题索引丢失了源码标识符或下划线");
}

const intro = await readFile(path.join(sourceRoot, "README.md"), "utf8");
const introSnapshot = intro.match(/main@([0-9a-f]{40})/)?.[1];
if (introSnapshot !== manifest.sourceSnapshot) {
  errors.push(
    `README 源码快照 ${introSnapshot || "missing"} 与 coverage ${manifest.sourceSnapshot} 不一致`,
  );
}

if (errors.length) {
  console.error(errors.join("\n"));
  process.exitCode = 1;
} else {
  console.log(
    `Verified ${manifest.totals.documents} documents from ${path.relative(repoRoot, sourceRoot)}; ` +
      `headings, anchors, fences, snapshot, links, source ranges, and required assets are valid.`,
  );
}

import { marked, Renderer } from "./vendor/marked.esm.js";
import { escapeHtml, icon, refreshIcons, routeFor, slugifyHeading } from "./utils.js";

let diagramSequence = 0;
let currentDiagramContainer = null;

function normalizeRepositoryPath(value) {
  const parts = [];
  for (const segment of value.split("/")) {
    if (!segment || segment === ".") continue;
    if (segment === "..") parts.pop();
    else parts.push(segment);
  }
  return parts.join("/");
}

function resolveLink(href, currentDocument, documentsBySourcePath) {
  if (!href) return { href: "#" };
  if (/^(?:https?:|mailto:|tel:)/i.test(href)) {
    return { href, external: true };
  }
  if (href.startsWith("#")) {
    return { href: routeFor(currentDocument.id, decodeURIComponent(href.slice(1))) };
  }

  const [target, section = ""] = href.split("#");
  const sourceDirectory = currentDocument.sourcePath.split("/").slice(0, -1).join("/");
  const resolvedPath = normalizeRepositoryPath(`${sourceDirectory}/${decodeURIComponent(target)}`);
  const linkedDocument = documentsBySourcePath.get(resolvedPath);
  if (linkedDocument) {
    return { href: routeFor(linkedDocument.id, section ? decodeURIComponent(section) : "") };
  }

  const repositoryHref = `/__repo/${resolvedPath
    .split("/")
    .map((part) => encodeURIComponent(part))
    .join("/")}`;
  const isSource = /\.(?:py|c|cc|cpp|cu|cuh|h|hpp|hip|rs|go|java|ts|js)$/i.test(resolvedPath);
  return { href: repositoryHref, sourcePath: isSource ? resolvedPath : "" };
}

function renderCodeLines(text) {
  return text
    .split("\n")
    .map(
      (line, index) =>
        `<span class="code-line" data-line="${index + 1}"><span>${escapeHtml(line) || " "}</span></span>`,
    )
    .join("");
}

function createRenderer(currentDocument, documentsBySourcePath) {
  const renderer = new Renderer();
  const headingCounts = new Map();
  const defaultTable = renderer.table;

  renderer.heading = function heading({ tokens, depth }) {
    const html = this.parser.parseInline(tokens);
    const text = tokens.map((token) => token.text || token.raw || "").join("");
    const base = slugifyHeading(text);
    const count = headingCounts.get(base) || 0;
    headingCounts.set(base, count + 1);
    const slug = count === 0 ? base : `${base}-${count + 1}`;
    const anchor = depth > 1
      ? `<a class="heading-anchor" href="${routeFor(currentDocument.id, slug)}" aria-label="链接到本节">#</a>`
      : "";
    return `<h${depth} id="${escapeHtml(slug)}">${anchor}${html}</h${depth}>\n`;
  };

  renderer.code = function code({ text, lang = "" }) {
    const language = lang.trim().split(/\s+/)[0] || "text";
    if (language.toLowerCase() === "mermaid") {
      return `
        <div class="diagram-shell" data-diagram-index="${diagramSequence++}">
          <div class="diagram-toolbar">
            <span>可交互流程图</span>
            <div class="diagram-actions">
              <button type="button" data-diagram-action="zoom-out" title="缩小" aria-label="缩小">${icon("zoom-out")}</button>
              <button type="button" data-diagram-action="reset" title="重置" aria-label="重置">${icon("scan")}</button>
              <button type="button" data-diagram-action="zoom-in" title="放大" aria-label="放大">${icon("zoom-in")}</button>
              <button type="button" data-diagram-action="fullscreen" title="全屏查看" aria-label="全屏查看">${icon("maximize-2")}</button>
            </div>
          </div>
          <div class="diagram-viewport">
            <div class="diagram-canvas"><div class="diagram-loading">正在绘制...</div></div>
          </div>
          <pre class="diagram-definition" hidden>${escapeHtml(text)}</pre>
        </div>`;
    }

    const lines = text.split("\n").length;
    const collapsible = lines > 22;
    return `
      <div class="code-shell${collapsible ? " collapsible" : ""}" data-language="${escapeHtml(language)}">
        <div class="code-toolbar">
          <div class="code-meta">
            <span class="code-language">${escapeHtml(language)}</span>
            <span>${lines} 行</span>
          </div>
          <div class="code-actions">
            <button type="button" data-code-action="copy" title="复制代码" aria-label="复制代码">${icon("copy")}</button>
          </div>
        </div>
        <pre><code class="language-${escapeHtml(language)}">${renderCodeLines(text)}</code></pre>
        <textarea class="raw-code" hidden>${escapeHtml(text)}</textarea>
        ${
          collapsible
            ? `<button class="code-expand" type="button" data-code-action="expand">${icon("chevrons-down")}<span>展开完整代码</span></button>`
            : ""
        }
      </div>`;
  };

  renderer.table = function table(token) {
    return `<div class="table-scroll">${defaultTable.call(this, token)}</div>`;
  };

  renderer.link = function link({ href, title, tokens }) {
    const text = this.parser.parseInline(tokens);
    const resolved = resolveLink(href, currentDocument, documentsBySourcePath);
    const attributes = [
      `href="${escapeHtml(resolved.href)}"`,
      title ? `title="${escapeHtml(title)}"` : "",
      resolved.external ? 'target="_blank" rel="noreferrer"' : "",
      resolved.sourcePath
        ? `class="source-link" data-source-path="${escapeHtml(resolved.sourcePath)}"`
        : "",
    ]
      .filter(Boolean)
      .join(" ");
    return `<a ${attributes}>${text}${resolved.external ? icon("external-link") : ""}</a>`;
  };

  renderer.image = function image({ href, title, text }) {
    const resolved = resolveLink(href, currentDocument, documentsBySourcePath);
    return `<img src="${escapeHtml(resolved.href)}" alt="${escapeHtml(text)}"${
      title ? ` title="${escapeHtml(title)}"` : ""
    } loading="lazy" />`;
  };

  return renderer;
}

function setupCodeBlocks(container, showToast) {
  container.querySelectorAll(".code-shell").forEach((shell) => {
    shell.querySelector('[data-code-action="copy"]')?.addEventListener("click", async (event) => {
      const code = shell.querySelector(".raw-code")?.value || "";
      try {
        await navigator.clipboard.writeText(code);
        showToast("代码已复制");
        const button = event.currentTarget;
        button.innerHTML = icon("check");
        refreshIcons(button);
        window.setTimeout(() => {
          button.innerHTML = icon("copy");
          refreshIcons(button);
        }, 1200);
      } catch {
        showToast("浏览器未允许写入剪贴板");
      }
    });

    shell.querySelector('[data-code-action="expand"]')?.addEventListener("click", (event) => {
      const expanded = shell.classList.toggle("expanded");
      event.currentTarget.innerHTML = `${icon(expanded ? "chevrons-up" : "chevrons-down")}<span>${
        expanded ? "收起代码" : "展开完整代码"
      }</span>`;
      refreshIcons(event.currentTarget);
    });
  });
}

function setupDetails(container) {
  container.querySelectorAll("details > summary").forEach((summary) => {
    const label = summary.textContent.trim();
    summary.innerHTML = `${icon("code-2")}<span>${escapeHtml(label)}</span>${icon("chevron-right")}`;
  });
}

function setupInlineSourceReferences(container, openSource) {
  const sourcePattern = /^((?:python|sgl-kernel|test|benchmark|examples)\/[^:]+\.(?:py|c|cc|cpp|cu|cuh|h|hpp|hip|rs|go|java|ts|js)):(\d+)(?:-(\d+))?$/;
  container.querySelectorAll(":not(pre) > code").forEach((code) => {
    const match = code.textContent.trim().match(sourcePattern);
    if (!match) return;
    const button = document.createElement("button");
    button.type = "button";
    button.className = "source-reference";
    button.title = "查看对应源码";
    button.innerHTML = `${icon("file-code-2")}<span>${escapeHtml(code.textContent.trim())}</span>`;
    button.addEventListener("click", () => {
      openSource(match[1], { start: Number(match[2]), end: Number(match[3] || match[2]) });
    });
    code.replaceWith(button);
  });
}

function configureMermaid() {
  if (!window.mermaid) return false;
  const dark = document.documentElement.dataset.theme === "dark";
  window.mermaid.initialize({
    startOnLoad: false,
    securityLevel: "strict",
    theme: "base",
    fontFamily:
      '-apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif',
    themeVariables: dark
      ? {
          background: "#171d1c",
          primaryColor: "#173a40",
          primaryTextColor: "#f4f8f6",
          primaryBorderColor: "#52b9ca",
          secondaryColor: "#1b382f",
          tertiaryColor: "#412725",
          lineColor: "#9ca8a4",
          textColor: "#dce4e1",
          noteBkgColor: "#3c331f",
          noteTextColor: "#f4f8f6",
        }
      : {
          background: "#ffffff",
          primaryColor: "#d9f0f3",
          primaryTextColor: "#101716",
          primaryBorderColor: "#087f95",
          secondaryColor: "#dff0e9",
          tertiaryColor: "#f7e4e1",
          lineColor: "#66716f",
          textColor: "#202827",
          noteBkgColor: "#f5ebcd",
          noteTextColor: "#202827",
        },
    flowchart: { htmlLabels: true, curve: "basis", padding: 12 },
    sequence: { diagramMarginX: 24, diagramMarginY: 18, useMaxWidth: true },
  });
  return true;
}

function applyDiagramTransform(shell) {
  const canvas = shell.querySelector(".diagram-canvas");
  const state = shell._diagramState;
  if (!canvas || !state) return;
  canvas.style.transform = `translate(${state.x}px, ${state.y}px) scale(${state.scale})`;
}

function bindDiagramInteractions(shell) {
  if (shell.dataset.interactionsBound === "true") return;
  shell.dataset.interactionsBound = "true";
  shell._diagramState = { scale: 1, x: 0, y: 0, dragging: false };
  const viewport = shell.querySelector(".diagram-viewport");

  shell.querySelectorAll("[data-diagram-action]").forEach((button) => {
    button.addEventListener("click", () => {
      const action = button.dataset.diagramAction;
      const state = shell._diagramState;
      if (action === "zoom-in") state.scale = Math.min(2.4, state.scale + 0.2);
      if (action === "zoom-out") state.scale = Math.max(0.55, state.scale - 0.2);
      if (action === "reset") Object.assign(state, { scale: 1, x: 0, y: 0 });
      if (action === "fullscreen") {
        const dialog = document.querySelector("#diagram-dialog");
        const content = document.querySelector("#diagram-dialog-content");
        content.innerHTML = shell.querySelector(".diagram-canvas")?.innerHTML || "";
        dialog.showModal();
      }
      applyDiagramTransform(shell);
    });
  });

  viewport.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    const state = shell._diagramState;
    state.dragging = true;
    state.pointerX = event.clientX;
    state.pointerY = event.clientY;
    viewport.setPointerCapture(event.pointerId);
  });
  viewport.addEventListener("pointermove", (event) => {
    const state = shell._diagramState;
    if (!state.dragging) return;
    state.x += event.clientX - state.pointerX;
    state.y += event.clientY - state.pointerY;
    state.pointerX = event.clientX;
    state.pointerY = event.clientY;
    applyDiagramTransform(shell);
  });
  const stopDragging = () => {
    shell._diagramState.dragging = false;
  };
  viewport.addEventListener("pointerup", stopDragging);
  viewport.addEventListener("pointercancel", stopDragging);
}

async function renderDiagram(shell, index) {
  const canvas = shell.querySelector(".diagram-canvas");
  const source = shell.querySelector(".diagram-definition")?.textContent || "";
  bindDiagramInteractions(shell);
  canvas.innerHTML = '<div class="diagram-loading">正在绘制...</div>';
  try {
    const renderId = `hicache-diagram-${Date.now()}-${index}-${Math.random().toString(16).slice(2)}`;
    const { svg, bindFunctions } = await window.mermaid.render(renderId, source);
    canvas.innerHTML = svg;
    const renderedSvg = canvas.querySelector("svg");
    renderedSvg?.removeAttribute("height");
    renderedSvg?.setAttribute("role", "img");
    renderedSvg?.setAttribute("aria-label", "HiCache 源码流程图");
    bindFunctions?.(canvas);
    applyDiagramTransform(shell);
  } catch (error) {
    canvas.innerHTML = `<div class="diagram-error">图示解析失败：${escapeHtml(
      error instanceof Error ? error.message : String(error),
    )}</div>`;
  }
}

export async function renderMermaidDiagrams(container) {
  currentDiagramContainer = container;
  const shells = [...container.querySelectorAll(".diagram-shell")];
  if (!shells.length) return;
  if (!configureMermaid()) {
    shells.forEach((shell) => {
      shell.querySelector(".diagram-canvas").innerHTML =
        '<div class="diagram-error">Mermaid 渲染器未加载</div>';
    });
    return;
  }
  for (const [index, shell] of shells.entries()) {
    await renderDiagram(shell, index);
  }
}

export async function rerenderCurrentDiagrams() {
  if (!currentDiagramContainer?.isConnected) return;
  await renderMermaidDiagrams(currentDiagramContainer);
}

export async function renderMarkdown({
  markdown,
  currentDocument,
  documentsBySourcePath,
  container,
  showToast,
  openSource,
}) {
  const renderer = createRenderer(currentDocument, documentsBySourcePath);
  container.innerHTML = marked.parse(markdown, {
    renderer,
    gfm: true,
    breaks: false,
  });
  setupCodeBlocks(container, showToast);
  setupDetails(container);
  setupInlineSourceReferences(container, openSource);
  container.querySelectorAll("a.source-link").forEach((link) => {
    link.addEventListener("click", (event) => {
      event.preventDefault();
      openSource(link.dataset.sourcePath);
    });
  });
  refreshIcons(container);
  await renderMermaidDiagrams(container);
}

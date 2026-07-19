import { injectChapterEnhancements } from "./interactives.js";
import { renderMarkdown, rerenderCurrentDiagrams } from "./reader.js";
import { SearchController } from "./search.js";
import {
  debounce,
  escapeHtml,
  formatNumber,
  icon,
  parseRoute,
  readJsonStorage,
  refreshIcons,
  routeFor,
  writeJsonStorage,
} from "./utils.js";

const STORAGE_KEYS = {
  read: "hicache-guide-read",
  bookmarks: "hicache-guide-bookmarks",
  scroll: "hicache-guide-scroll",
  collapsed: "hicache-guide-collapsed-groups",
  focus: "hicache-guide-focus-mode",
};

const elements = {
  article: document.querySelector("#article"),
  chapterFacts: document.querySelector("#chapter-facts"),
  chapterNavigation: document.querySelector("#chapter-navigation"),
  chapterPager: document.querySelector("#chapter-pager"),
  chapterSidebar: document.querySelector("#chapter-sidebar"),
  chapterStatus: document.querySelector("#chapter-status"),
  pageToc: document.querySelector("#page-toc"),
  progressBar: document.querySelector("#reading-progress-bar"),
  bookProgressBar: document.querySelector("#book-progress-bar"),
  bookProgressLabel: document.querySelector("#book-progress-label"),
  bookMeta: document.querySelector("#book-meta"),
  bookmarkButton: document.querySelector("#bookmark-button"),
  focusButton: document.querySelector("#focus-button"),
  navToggle: document.querySelector("#nav-toggle"),
  pageAside: document.querySelector("#page-aside"),
  readerMain: document.querySelector("#reader-main"),
  sidebarBackdrop: document.querySelector("#sidebar-backdrop"),
  sourceSnapshot: document.querySelector("#source-snapshot"),
  sourceDialog: document.querySelector("#source-dialog"),
  sourceDialogTitle: document.querySelector("#source-dialog-title"),
  sourceDialogPath: document.querySelector("#source-dialog-path"),
  sourceDialogCode: document.querySelector("#source-dialog-code"),
  toast: document.querySelector("#toast"),
};

const mobileNavigationMedia = window.matchMedia("(max-width: 820px)");
const pageAsideMedia = window.matchMedia("(max-width: 1180px)");

function readStoredArray(key, fallback = []) {
  const value = readJsonStorage(key, fallback);
  return Array.isArray(value) ? value : fallback;
}

function readStoredRecord(key) {
  const value = readJsonStorage(key, {});
  return value && typeof value === "object" && !Array.isArray(value) ? value : {};
}

const state = {
  manifest: null,
  documentsById: new Map(),
  documentsBySourcePath: new Map(),
  currentDocument: null,
  currentSection: "",
  contentCache: new Map(),
  readDocuments: new Set(readStoredArray(STORAGE_KEYS.read)),
  bookmarks: new Set(readStoredArray(STORAGE_KEYS.bookmarks)),
  scrollPositions: readStoredRecord(STORAGE_KEYS.scroll),
  collapsedGroups: new Set(readStoredArray(STORAGE_KEYS.collapsed, ["appendix"])),
  routeToken: 0,
  sourceRequestToken: 0,
  tocObserver: null,
  readTimer: null,
  toastTimer: null,
  search: null,
};

function showToast(message) {
  elements.toast.textContent = message;
  elements.toast.classList.add("visible");
  window.clearTimeout(state.toastTimer);
  state.toastTimer = window.setTimeout(() => elements.toast.classList.remove("visible"), 1800);
}

function setMobileNavigationOpen(open, restoreFocus = false) {
  const mobile = mobileNavigationMedia.matches;
  const effectiveOpen = mobile && open;
  const focusMode = document.body.classList.contains("focus-mode") && !mobile;
  const sidebarHidden = (mobile && !effectiveOpen) || focusMode;
  const pageAsideHidden = pageAsideMedia.matches || focusMode;
  if (sidebarHidden && elements.chapterSidebar.contains(document.activeElement)) {
    (restoreFocus ? elements.navToggle : elements.readerMain).focus({ preventScroll: true });
  }
  if (pageAsideHidden && elements.pageAside.contains(document.activeElement)) {
    elements.readerMain.focus({ preventScroll: true });
  }
  document.body.classList.toggle("nav-open", effectiveOpen);
  elements.navToggle.setAttribute("aria-expanded", String(effectiveOpen));
  elements.navToggle.setAttribute(
    "aria-label",
    effectiveOpen ? "关闭章节目录" : "打开章节目录",
  );
  elements.navToggle.title = effectiveOpen ? "关闭章节目录" : "章节目录";

  elements.chapterSidebar.toggleAttribute("inert", sidebarHidden);
  if (sidebarHidden) elements.chapterSidebar.setAttribute("aria-hidden", "true");
  else elements.chapterSidebar.removeAttribute("aria-hidden");

  elements.pageAside.toggleAttribute("inert", pageAsideHidden);
  if (pageAsideHidden) elements.pageAside.setAttribute("aria-hidden", "true");
  else elements.pageAside.removeAttribute("aria-hidden");

  elements.sidebarBackdrop.toggleAttribute("inert", !effectiveOpen);
  elements.sidebarBackdrop.setAttribute("aria-hidden", String(!effectiveOpen));
  if (!effectiveOpen && restoreFocus) elements.navToggle.focus({ preventScroll: true });
}

function currentGroup() {
  return state.manifest.groups.find(
    (group) => group.id === state.currentDocument?.groupId,
  );
}

function renderNavigation() {
  elements.chapterNavigation.innerHTML = state.manifest.groups
    .map((group) => {
      const collapsed = state.collapsedGroups.has(group.id);
      const documents = group.documentIds.map((id) => state.documentsById.get(id));
      return `
        <section class="nav-group${collapsed ? " collapsed" : ""}" data-group="${group.id}">
          <button class="nav-group-header" type="button" aria-expanded="${!collapsed}">
            <span>${escapeHtml(group.title)}</span>${icon("chevron-down")}
          </button>
          <div class="nav-group-items">
            ${documents
              .map((document) => {
                const code = document.isAppendix
                  ? document.id.split("/").pop().match(/^([A-Z]\d+[a-z]?)/i)?.[1] || "REF"
                  : "";
                return `
                  <a class="chapter-link"
                     data-document-id="${escapeHtml(document.id)}"
                     href="${routeFor(document.id)}"
                     title="${escapeHtml(document.title)}">
                    <span class="status-dot" aria-hidden="true"></span>
                    <span>${escapeHtml(document.displayTitle)}</span>
                    ${code ? `<span class="appendix-code">${escapeHtml(code)}</span>` : ""}
                  </a>`;
              })
              .join("")}
          </div>
        </section>`;
    })
    .join("");

  elements.chapterNavigation.querySelectorAll(".nav-group-header").forEach((button) => {
    button.addEventListener("click", () => {
      const groupElement = button.closest(".nav-group");
      const collapsed = groupElement.classList.toggle("collapsed");
      button.setAttribute("aria-expanded", String(!collapsed));
      if (collapsed) state.collapsedGroups.add(groupElement.dataset.group);
      else state.collapsedGroups.delete(groupElement.dataset.group);
      writeJsonStorage(STORAGE_KEYS.collapsed, [...state.collapsedGroups]);
    });
  });
  elements.chapterNavigation.addEventListener("click", (event) => {
    if (event.target.closest(".chapter-link")) setMobileNavigationOpen(false);
  });
  refreshIcons(elements.chapterNavigation);
  updateNavigationState();
}

function updateNavigationState() {
  elements.chapterNavigation.querySelectorAll(".chapter-link").forEach((link) => {
    const id = link.dataset.documentId;
    link.classList.toggle("active", id === state.currentDocument?.id);
    link.classList.toggle("read", state.readDocuments.has(id));
    link.classList.toggle("bookmarked", state.bookmarks.has(id));
  });

  const activeLink = elements.chapterNavigation.querySelector(".chapter-link.active");
  const activeGroup = activeLink?.closest(".nav-group");
  if (activeGroup?.classList.contains("collapsed")) {
    activeGroup.classList.remove("collapsed");
    activeGroup.querySelector(".nav-group-header")?.setAttribute("aria-expanded", "true");
  }
  activeLink?.scrollIntoView({ block: "nearest" });

  const progress = state.manifest
    ? Math.round((state.readDocuments.size / state.manifest.totals.documents) * 100)
    : 0;
  elements.bookProgressLabel.textContent = `${Math.min(100, progress)}%`;
  elements.bookProgressBar.style.width = `${Math.min(100, progress)}%`;

  const bookmarked = state.bookmarks.has(state.currentDocument?.id);
  elements.bookmarkButton.classList.toggle("active", bookmarked);
  elements.bookmarkButton.setAttribute("aria-pressed", String(bookmarked));
  elements.bookmarkButton.title = bookmarked ? "取消收藏" : "收藏当前章节";
}

function markCurrentAsRead() {
  const id = state.currentDocument?.id;
  if (!id || state.readDocuments.has(id)) return;
  state.readDocuments.add(id);
  writeJsonStorage(STORAGE_KEYS.read, [...state.readDocuments]);
  updateNavigationState();
}

function toggleBookmark() {
  const id = state.currentDocument?.id;
  if (!id) return;
  if (state.bookmarks.has(id)) {
    state.bookmarks.delete(id);
    showToast("已取消收藏");
  } else {
    state.bookmarks.add(id);
    showToast("已收藏当前章节");
  }
  writeJsonStorage(STORAGE_KEYS.bookmarks, [...state.bookmarks]);
  updateNavigationState();
}

function renderChapterMetadata(document) {
  const group = currentGroup();
  elements.chapterStatus.innerHTML = `
    <span class="status-label">${escapeHtml(group?.title || "章节")}</span>
    <span>${document.isAppendix ? "逐函数对照" : `第 ${document.order + 1} 篇`}</span>
    <span class="status-separator">/</span>
    <span>${document.stats.estimatedMinutes} 分钟</span>
    ${document.stats.diagrams ? `<span class="status-separator">/</span><span>${document.stats.diagrams} 张图</span>` : ""}`;

  elements.chapterFacts.innerHTML = `
    <div class="aside-heading">章节数据</div>
    <dl>
      <div><dt>正文字符</dt><dd>${formatNumber(document.stats.proseCharacters)}</dd></div>
      <div><dt>代码行</dt><dd>${formatNumber(document.stats.codeLines)}</dd></div>
      <div><dt>流程图</dt><dd>${formatNumber(document.stats.diagrams)}</dd></div>
      <div><dt>阅读估时</dt><dd>${document.stats.estimatedMinutes} 分钟</dd></div>
    </dl>`;
}

function renderPager(document) {
  const documents = state.manifest.documents;
  const previous = documents[document.order - 1];
  const next = documents[document.order + 1];
  const link = (target, direction) => {
    if (!target) return "<span></span>";
    const isNext = direction === "next";
    return `
      <a class="pager-link ${direction}" href="${routeFor(target.id)}">
        ${isNext ? "" : icon("arrow-left")}
        <span><span>${isNext ? "下一章" : "上一章"}</span><strong>${escapeHtml(target.displayTitle)}</strong></span>
        ${isNext ? icon("arrow-right") : ""}
      </a>`;
  };
  elements.chapterPager.innerHTML = `${link(previous, "previous")}${link(next, "next")}`;
  refreshIcons(elements.chapterPager);
}

function renderTableOfContents() {
  state.tocObserver?.disconnect();
  const headings = [...elements.article.querySelectorAll("h2[id], h3[id]")];
  elements.pageToc.innerHTML = headings
    .map(
      (heading) =>
        `<a href="${routeFor(state.currentDocument.id, heading.id)}" data-level="${heading.tagName.slice(1)}" data-heading-id="${escapeHtml(heading.id)}">${escapeHtml(heading.textContent.replace(/^#/, "").trim())}</a>`,
    )
    .join("");
  if (!headings.length) return;

  state.tocObserver = new IntersectionObserver(
    (entries) => {
      const visible = entries
        .filter((entry) => entry.isIntersecting)
        .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)[0];
      if (!visible) return;
      elements.pageToc.querySelectorAll("a").forEach((link) => {
        link.classList.toggle("active", link.dataset.headingId === visible.target.id);
      });
    },
    { rootMargin: "-72px 0px -72% 0px", threshold: [0, 1] },
  );
  headings.forEach((heading) => state.tocObserver.observe(heading));
}

function saveScrollPosition() {
  if (!state.currentDocument) return;
  state.scrollPositions[state.currentDocument.id] = window.scrollY;
  writeJsonStorage(STORAGE_KEYS.scroll, state.scrollPositions);
}

const saveScrollPositionDebounced = debounce(saveScrollPosition, 250);

function updateReadingProgress() {
  const articleRect = elements.article.getBoundingClientRect();
  const articleTop = window.scrollY + articleRect.top;
  const readableDistance = Math.max(1, articleRect.height - window.innerHeight + 80);
  const progress = Math.max(0, Math.min(1, (window.scrollY - articleTop + 90) / readableDistance));
  elements.progressBar.style.width = `${Math.round(progress * 100)}%`;
  if (progress > 0.3) markCurrentAsRead();
  saveScrollPositionDebounced();
}

function scrollToRouteSection(section, restorePosition = false) {
  requestAnimationFrame(() => {
    if (section) {
      const target = document.getElementById(section);
      if (target) target.scrollIntoView({ block: "start" });
      else window.scrollTo({ top: 0 });
    } else if (restorePosition && state.scrollPositions[state.currentDocument.id] > 0) {
      window.scrollTo({ top: state.scrollPositions[state.currentDocument.id] });
    } else {
      window.scrollTo({ top: 0 });
    }
    updateReadingProgress();
  });
}

async function getMarkdown(document) {
  if (state.contentCache.has(document.id)) return state.contentCache.get(document.id);
  const response = await fetch(document.contentUrl, { cache: "no-cache" });
  if (!response.ok) throw new Error(`无法读取 ${document.path}: HTTP ${response.status}`);
  const markdown = await response.text();
  state.contentCache.set(document.id, markdown);
  return markdown;
}

async function openSource(sourcePath, range = {}) {
  if (!sourcePath) return;
  const requestToken = ++state.sourceRequestToken;
  elements.sourceDialogTitle.textContent = sourcePath.split("/").pop();
  elements.sourceDialogPath.textContent = sourcePath;
  elements.sourceDialogCode.textContent = "正在读取源码...";
  if (!elements.sourceDialog.open) elements.sourceDialog.showModal();
  try {
    const encoded = sourcePath
      .split("/")
      .map((part) => encodeURIComponent(part))
      .join("/");
    const response = await fetch(`/__repo/${encoded}`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const source = await response.text();
    if (requestToken !== state.sourceRequestToken) return;

    const lines = source.replace(/\r\n?/g, "\n").split("\n");
    const requestedStart = Number(range.start);
    const hasRange = Number.isFinite(requestedStart) && requestedStart > 0;
    if (!hasRange) {
      elements.sourceDialogPath.textContent = `${sourcePath} · ${formatNumber(lines.length)} 行`;
      elements.sourceDialogCode.dataset.view = "full";
      elements.sourceDialogCode.textContent = source;
      return;
    }

    const start = Math.min(lines.length, Math.max(1, Math.trunc(requestedStart)));
    const requestedEnd = Number(range.end);
    const end = Math.min(
      lines.length,
      Math.max(start, Number.isFinite(requestedEnd) ? Math.trunc(requestedEnd) : start),
    );
    const contextLines = 40;
    const renderStart = Math.max(1, start - contextLines);
    const renderEnd = Math.min(lines.length, end + contextLines);
    elements.sourceDialogPath.textContent = `${sourcePath} · 第 ${start}-${end} 行（显示 ${renderStart}-${renderEnd} / 共 ${formatNumber(lines.length)} 行）`;
    elements.sourceDialogCode.dataset.view = "window";
    elements.sourceDialogCode.innerHTML = lines
      .slice(renderStart - 1, renderEnd)
      .map((line, index) => {
        const lineNumber = renderStart + index;
        const highlighted = lineNumber >= start && lineNumber <= end;
        return `<span class="source-line${highlighted ? " highlighted" : ""}" data-line="${lineNumber}"><span>${lineNumber}</span><span>${escapeHtml(line) || " "}</span></span>`;
      })
      .join("");
    requestAnimationFrame(() => {
      const targetLine = elements.sourceDialogCode.querySelector(`[data-line="${start}"]`);
      const preview = targetLine?.closest(".source-preview");
      if (targetLine && preview) preview.scrollTop = Math.max(0, targetLine.offsetTop - 80);
    });
  } catch (error) {
    if (requestToken !== state.sourceRequestToken) return;
    elements.sourceDialogCode.textContent = `源码读取失败：${
      error instanceof Error ? error.message : String(error)
    }`;
  }
}

async function loadDocument(targetDocument, section = "") {
  const token = ++state.routeToken;
  state.tocObserver?.disconnect();
  window.clearTimeout(state.readTimer);
  elements.article.innerHTML = `
    <div class="loading-state"><span class="loading-mark" aria-hidden="true"></span><p>正在读取 ${escapeHtml(targetDocument.displayTitle)}...</p></div>`;
  elements.pageToc.innerHTML = "";
  state.currentDocument = targetDocument;
  state.currentSection = section;
  updateNavigationState();
  renderChapterMetadata(targetDocument);
  renderPager(targetDocument);
  globalThis.document.title = `${targetDocument.displayTitle} · HiCache 源码剖析`;

  try {
    const markdown = await getMarkdown(targetDocument);
    if (token !== state.routeToken) return;
    await renderMarkdown({
      markdown,
      currentDocument: targetDocument,
      documentsBySourcePath: state.documentsBySourcePath,
      container: elements.article,
      showToast,
      openSource,
    });
    if (token !== state.routeToken) return;
    injectChapterEnhancements(elements.article, targetDocument, state.manifest);
    refreshIcons(elements.article);
    renderTableOfContents();
    scrollToRouteSection(section, true);
    state.readTimer = window.setTimeout(markCurrentAsRead, 15000);
  } catch (error) {
    if (token !== state.routeToken) return;
    elements.article.innerHTML = `
      <div class="error-state"><h1>章节加载失败</h1><p>${escapeHtml(
        error instanceof Error ? error.message : String(error),
      )}</p></div>`;
  }
}

async function handleRoute() {
  if (!state.manifest) return;
  const route = parseRoute(location.hash, new Set(state.documentsById.keys()));
  const targetDocument = state.documentsById.get(route.id);
  setMobileNavigationOpen(false);
  if (state.currentDocument?.id === targetDocument.id) {
    state.currentSection = route.section;
    scrollToRouteSection(route.section, false);
    return;
  }
  saveScrollPosition();
  await loadDocument(targetDocument, route.section);
}

function navigate(documentId, section = "") {
  const target = routeFor(documentId, section);
  if (location.hash === target) {
    scrollToRouteSection(section, false);
  } else {
    location.hash = target;
  }
}

function resolveTheme(mode) {
  if (mode === "system") {
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }
  return mode;
}

async function applyTheme(mode) {
  const resolved = resolveTheme(mode);
  document.documentElement.dataset.themeMode = mode;
  document.documentElement.dataset.theme = resolved;
  try {
    localStorage.setItem("hicache-guide-theme", mode);
  } catch {
    // Theme switching still works for the current session without persistence.
  }
  document.querySelector("#theme-button").title = `主题：${
    mode === "system" ? "跟随系统" : mode === "dark" ? "深色" : "浅色"
  }`;
  await rerenderCurrentDiagrams();
}

function cycleTheme() {
  const modes = ["system", "light", "dark"];
  const current = document.documentElement.dataset.themeMode || "system";
  applyTheme(modes[(modes.indexOf(current) + 1) % modes.length]);
}

function initializeEvents() {
  document.querySelector("#search-trigger").addEventListener("click", () => state.search.open());
  document.querySelector("#mobile-search-button").addEventListener("click", () => state.search.open());
  elements.navToggle.addEventListener("click", () => {
    setMobileNavigationOpen(!document.body.classList.contains("nav-open"));
  });
  elements.sidebarBackdrop.addEventListener("click", () => {
    setMobileNavigationOpen(false, true);
  });
  elements.bookmarkButton.addEventListener("click", toggleBookmark);
  document.querySelector("#theme-button").addEventListener("click", cycleTheme);
  elements.focusButton.addEventListener("click", () => {
    const focused = document.body.classList.toggle("focus-mode");
    elements.focusButton.classList.toggle("active", focused);
    elements.focusButton.setAttribute("aria-pressed", String(focused));
    writeJsonStorage(STORAGE_KEYS.focus, focused);
    setMobileNavigationOpen(false);
  });
  document.querySelector("#glossary-command").addEventListener("click", () => {
    state.search.open("术语", "core");
  });

  document.addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
      event.preventDefault();
      state.search.open();
    }
    if (
      event.key === "Escape" &&
      document.body.classList.contains("nav-open") &&
      !document.querySelector("dialog[open]")
    ) {
      event.preventDefault();
      setMobileNavigationOpen(false, true);
    }
  });
  window.addEventListener("hashchange", handleRoute);
  window.addEventListener("scroll", updateReadingProgress, { passive: true });
  window.addEventListener("resize", debounce(updateReadingProgress, 120));
  mobileNavigationMedia.addEventListener("change", () => setMobileNavigationOpen(false));
  pageAsideMedia.addEventListener("change", () => setMobileNavigationOpen(false));
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    if (document.documentElement.dataset.themeMode === "system") applyTheme("system");
  });

  document.querySelector("#diagram-dialog [data-dialog-close]").addEventListener("click", () => {
    document.querySelector("#diagram-dialog").close();
  });
  document.querySelector("#diagram-dialog").addEventListener("click", (event) => {
    if (event.target === event.currentTarget) event.currentTarget.close();
  });
  const closeSourceDialog = () => {
    state.sourceRequestToken += 1;
    if (elements.sourceDialog.open) elements.sourceDialog.close();
  };
  document.querySelector("[data-source-close]").addEventListener("click", closeSourceDialog);
  elements.sourceDialog.addEventListener("click", (event) => {
    if (event.target === elements.sourceDialog) closeSourceDialog();
  });
  elements.sourceDialog.addEventListener("cancel", () => {
    state.sourceRequestToken += 1;
  });
}

async function initialize() {
  try {
    const response = await fetch("./.generated/book-manifest.json", { cache: "no-cache" });
    if (!response.ok) throw new Error(`目录索引读取失败: HTTP ${response.status}`);
    state.manifest = await response.json();
    state.documentsById = new Map(state.manifest.documents.map((document) => [document.id, document]));
    state.documentsBySourcePath = new Map(
      state.manifest.documents.map((document) => [document.sourcePath, document]),
    );
    const knownDocumentIds = new Set(state.documentsById.keys());
    const knownGroupIds = new Set(state.manifest.groups.map((group) => group.id));
    state.readDocuments = new Set(
      [...state.readDocuments].filter((id) => knownDocumentIds.has(id)),
    );
    state.bookmarks = new Set([...state.bookmarks].filter((id) => knownDocumentIds.has(id)));
    state.collapsedGroups = new Set(
      [...state.collapsedGroups].filter((id) => knownGroupIds.has(id)),
    );
    state.scrollPositions = Object.fromEntries(
      Object.entries(state.scrollPositions).filter(
        ([id, position]) =>
          knownDocumentIds.has(id) && Number.isFinite(position) && position >= 0,
      ),
    );
    writeJsonStorage(STORAGE_KEYS.read, [...state.readDocuments]);
    writeJsonStorage(STORAGE_KEYS.bookmarks, [...state.bookmarks]);
    writeJsonStorage(STORAGE_KEYS.collapsed, [...state.collapsedGroups]);
    writeJsonStorage(STORAGE_KEYS.scroll, state.scrollPositions);
    elements.bookMeta.textContent = `${state.manifest.totals.coreDocuments} 篇正文 · ${formatNumber(
      state.manifest.totals.functions,
    )} 个函数 · ${state.manifest.sourceSnapshot.slice(0, 8)}`;
    elements.sourceSnapshot.textContent = `源码快照 ${state.manifest.sourceSnapshot.slice(0, 12)}`;
    state.search = new SearchController(state.manifest, navigate);
    renderNavigation();
    initializeEvents();

    const focused = readJsonStorage(STORAGE_KEYS.focus, false) === true;
    document.body.classList.toggle("focus-mode", focused);
    elements.focusButton.classList.toggle("active", focused);
    elements.focusButton.setAttribute("aria-pressed", String(focused));
    setMobileNavigationOpen(false);
    refreshIcons();

    if (!location.hash.startsWith("#/")) {
      history.replaceState(null, "", routeFor("README"));
    }
    await handleRoute();
  } catch (error) {
    elements.article.innerHTML = `
      <div class="error-state"><h1>网页版初始化失败</h1><p>${escapeHtml(
        error instanceof Error ? error.message : String(error),
      )}</p></div>`;
  }
}

initialize();

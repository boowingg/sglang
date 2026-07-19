import {
  escapeHtml,
  icon,
  normalizeSearch,
  refreshIcons,
  routeFor,
} from "./utils.js";

const EMPTY_SUGGESTIONS = ["cache_finished_req", "host_value", "write_storage", "ACK"];

function highlight(text, query) {
  const escaped = escapeHtml(text);
  const term = query.trim();
  if (!term) return escaped;
  const safeTerm = term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  return escaped.replace(new RegExp(`(${safeTerm})`, "ig"), "<mark>$1</mark>");
}

function scoreDocument(document, query) {
  const normalized = normalizeSearch(query);
  const terms = normalized.split(/\s+/).filter(Boolean);
  if (!terms.length) return null;

  const title = normalizeSearch(`${document.title} ${document.displayTitle}`);
  const heading = document.headings.find((item) =>
    terms.every((term) => normalizeSearch(item.title).includes(term)),
  );
  const haystack = document.searchText;
  if (!terms.every((term) => title.includes(term) || haystack.includes(term))) return null;

  let score = document.isAppendix ? 0 : 8;
  for (const term of terms) {
    if (title === term) score += 160;
    else if (title.includes(term)) score += 80;
    if (heading && normalizeSearch(heading.title).includes(term)) score += 46;
    const index = haystack.indexOf(term);
    if (index >= 0) score += Math.max(4, 20 - Math.floor(index / 2500));
  }

  return {
    document,
    heading,
    score,
    summary: document.summary || heading?.title || document.title,
  };
}

export class SearchController {
  constructor(manifest, onNavigate) {
    this.manifest = manifest;
    this.onNavigate = onNavigate;
    this.dialog = document.querySelector("#search-dialog");
    this.input = document.querySelector("#search-input");
    this.resultsElement = document.querySelector("#search-results");
    this.countElement = document.querySelector("#search-count");
    this.scope = "core";
    this.results = [];
    this.selectedIndex = 0;
    this.bindEvents();
    this.syncScopeButtons();
    this.renderEmpty();
  }

  bindEvents() {
    document.querySelector("#search-close").addEventListener("click", () => this.close());
    this.dialog.addEventListener("click", (event) => {
      if (event.target === this.dialog) this.close();
    });
    this.input.addEventListener("input", () => this.search(this.input.value));
    this.input.addEventListener("keydown", (event) => this.onKeydown(event));
    this.dialog.querySelectorAll("[data-scope]").forEach((button) => {
      button.addEventListener("click", () => {
        this.scope = button.dataset.scope;
        this.syncScopeButtons();
        this.search(this.input.value);
      });
    });
    this.resultsElement.addEventListener("mousemove", (event) => {
      const result = event.target.closest(".search-result");
      if (!result) return;
      this.selectedIndex = Number(result.dataset.index);
      this.syncSelection();
    });
    this.resultsElement.addEventListener("click", (event) => {
      const suggestion = event.target.closest("[data-suggestion]");
      if (suggestion) {
        this.input.value = suggestion.dataset.suggestion;
        this.search(this.input.value);
        this.input.focus();
        return;
      }
      const result = event.target.closest(".search-result");
      if (!result) return;
      event.preventDefault();
      this.navigateResult(Number(result.dataset.index));
    });
  }

  syncScopeButtons() {
    this.dialog.querySelectorAll("[data-scope]").forEach((button) => {
      const selected = button.dataset.scope === this.scope;
      button.classList.toggle("active", selected);
      button.setAttribute("aria-pressed", String(selected));
    });
  }

  open(initialQuery = "", scope = this.scope) {
    this.scope = scope === "all" ? "all" : "core";
    this.syncScopeButtons();
    if (!this.dialog.open) this.dialog.showModal();
    this.input.value = initialQuery;
    this.search(initialQuery);
    requestAnimationFrame(() => this.input.focus());
  }

  close() {
    if (this.dialog.open) this.dialog.close();
  }

  search(query) {
    const normalized = normalizeSearch(query);
    if (!normalized) {
      this.results = [];
      this.selectedIndex = 0;
      this.countElement.textContent = "";
      this.renderEmpty();
      return;
    }

    this.results = this.manifest.documents
      .filter((document) => this.scope === "all" || !document.isAppendix)
      .map((document) => scoreDocument(document, query))
      .filter(Boolean)
      .sort((a, b) => b.score - a.score || a.document.order - b.document.order)
      .slice(0, 40);
    this.selectedIndex = 0;
    this.countElement.textContent = `${this.results.length} 项`;
    this.renderResults(query);
  }

  renderEmpty() {
    this.resultsElement.innerHTML = `
      <div class="search-empty">
        ${icon("scan-search")}
        <strong>定位一个概念或源码符号</strong>
        <span>${EMPTY_SUGGESTIONS.map(
          (term) => `<button type="button" data-suggestion="${escapeHtml(term)}">${escapeHtml(term)}</button>`,
        ).join(" · ")}</span>
      </div>`;
    refreshIcons(this.resultsElement);
  }

  renderResults(query) {
    if (!this.results.length) {
      this.resultsElement.innerHTML = `
        <div class="search-empty">
          ${icon("file-question")}
          <strong>没有找到对应内容</strong>
          <span>当前检索范围内没有匹配条目</span>
        </div>`;
      refreshIcons(this.resultsElement);
      return;
    }

    this.resultsElement.innerHTML = this.results
      .map(({ document, heading, summary }, index) => {
        const section = heading?.slug || "";
        const type = document.isAppendix ? "函数附录" : "正文";
        return `
          <a class="search-result${index === 0 ? " selected" : ""}"
             data-index="${index}"
             href="${routeFor(document.id, section)}">
            <span class="search-result-icon">${icon(document.isAppendix ? "braces" : "book-open")}</span>
            <span>
              <strong>${highlight(document.displayTitle, query)}</strong>
              ${heading ? `<small>${highlight(heading.title, query)}</small>` : ""}
              <p>${highlight(summary, query)}</p>
            </span>
            <span class="search-result-type">${type}</span>
          </a>`;
      })
      .join("");
    refreshIcons(this.resultsElement);
  }

  syncSelection() {
    this.resultsElement.querySelectorAll(".search-result").forEach((element, index) => {
      element.classList.toggle("selected", index === this.selectedIndex);
    });
    this.resultsElement
      .querySelector(`.search-result[data-index="${this.selectedIndex}"]`)
      ?.scrollIntoView({ block: "nearest" });
  }

  onKeydown(event) {
    if (!this.results.length) return;
    if (event.key === "ArrowDown") {
      event.preventDefault();
      this.selectedIndex = (this.selectedIndex + 1) % this.results.length;
      this.syncSelection();
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      this.selectedIndex = (this.selectedIndex - 1 + this.results.length) % this.results.length;
      this.syncSelection();
    } else if (event.key === "Enter") {
      event.preventDefault();
      this.navigateResult(this.selectedIndex);
    }
  }

  navigateResult(index) {
    const result = this.results[index];
    if (!result) return;
    this.close();
    this.onNavigate(result.document.id, result.heading?.slug || "");
  }
}

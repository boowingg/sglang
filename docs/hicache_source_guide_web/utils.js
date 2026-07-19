export function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

export function stripInlineMarkdown(value) {
  return String(value)
    .replace(/!\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(/\[([^\]]+)\]\([^)]*\)/g, "$1")
    .replace(/<[^>]+>/g, "")
    .replace(/[`*~]/g, "")
    .replace(/\\([\\`*{}\[\]()#+.!_>-])/g, "$1")
    .trim();
}

export function slugifyHeading(value) {
  return stripInlineMarkdown(value)
    .normalize("NFKC")
    .toLowerCase()
    .replace(/&[a-z]+;/g, "")
    .replace(/[^\p{L}\p{N}\p{M}\s_-]+/gu, "")
    .replace(/\s+/g, "-")
    .replace(/-+/g, "-")
    .replace(/^-+|-+$/g, "") || "section";
}

export function normalizeSearch(value) {
  return String(value).normalize("NFKC").toLowerCase().trim();
}

export function icon(name, label = "") {
  return `<i data-lucide="${escapeHtml(name)}"${label ? ` aria-label="${escapeHtml(label)}"` : ""}></i>`;
}

export function refreshIcons(root = document) {
  if (window.lucide?.createIcons) {
    window.lucide.createIcons({ root });
  }
}

export function debounce(callback, wait = 120) {
  let timer = null;
  return (...args) => {
    window.clearTimeout(timer);
    timer = window.setTimeout(() => callback(...args), wait);
  };
}

export function readJsonStorage(key, fallback) {
  try {
    const raw = localStorage.getItem(key);
    return raw === null ? fallback : JSON.parse(raw);
  } catch {
    return fallback;
  }
}

export function writeJsonStorage(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // The reader remains functional when storage is disabled.
  }
}

export function routeFor(documentId, section = "") {
  const route = `#/${encodeURI(documentId)}`;
  return section ? `${route}?section=${encodeURIComponent(section)}` : route;
}

export function parseRoute(hash, knownIds) {
  const raw = hash.startsWith("#/") ? hash.slice(2) : "README";
  const [encodedId, query = ""] = raw.split("?");
  let id = "README";
  try {
    id = decodeURI(encodedId || "README");
  } catch {
    id = "README";
  }
  if (!knownIds.has(id)) id = "README";
  const section = new URLSearchParams(query).get("section") || "";
  return { id, section };
}

export function formatNumber(value) {
  return new Intl.NumberFormat("zh-CN").format(value);
}

import { escapeHtml, formatNumber, icon, refreshIcons } from "./utils.js";

const PATH_NODES = [
  {
    id: "request",
    title: "请求 token",
    subtitle: "构造 RadixKey",
    detail: "请求进入调度器后，token 序列连同 extra_key 组成本地树的匹配键。这里的压力主要来自到达速率、prompt 长度和批处理形成速度。",
    pressure: "请求到达与批处理",
  },
  {
    id: "radix",
    title: "Radix 匹配",
    subtitle: "查共享前缀",
    detail: "树遍历返回分层匹配状态和各层索引。树锁持有时间、节点分裂频率和长前缀比较成本会在高并发下放大。",
    pressure: "树锁与元数据操作",
  },
  {
    id: "local",
    title: "本地层判定",
    subtitle: "L1 / L2",
    detail: "先判断可直接计算的 GPU 前缀，再判断 host 副本。L1 容量不足会提高淘汰频率；L2 可用页不足会阻塞后续 prefetch 和写回。",
    pressure: "L1/L2 容量与保护页",
  },
  {
    id: "query",
    title: "L3 query",
    subtitle: "最长连续前缀",
    detail: "Controller 查询连续 page hash。后端查询吞吐、尾延迟、namespace 一致性和 read-after-write 可见性决定这里能否识别可复用数据。",
    pressure: "查询 QPS 与尾延迟",
  },
  {
    id: "allocate",
    title: "分配 L2 页",
    subtitle: "预留传输空间",
    detail: "确认 L3 命中后必须先拿到足够的 host slots。若 in-flight prefetch、写回和锁保护页共同占满 L2，后端即使命中也无法形成有效回载。",
    pressure: "L2 allocation 失败",
  },
  {
    id: "read",
    title: "L3 read",
    subtitle: "远端页进入 host",
    detail: "后端把 page bytes 写入指定 host buffer。关注实际读取页数、失败页数、带宽、平均/尾延迟和读取完成队列积压。",
    pressure: "L3 读带宽与失败",
  },
  {
    id: "h2d",
    title: "H2D 回载",
    subtitle: "L2 → L1",
    detail: "逐层 CUDA copy 把 KV 写回 GPU slots。GPU slot 分配、copy stream 排队、event 同步和 layer-done 计数都会决定首 token 前等待。",
    pressure: "GPU 分配与 H2D stream",
  },
  {
    id: "compute",
    title: "模型计算",
    subtitle: "只算未命中后缀",
    detail: "有效回载前缀可直接复用；其余 token 重新 prefill。端到端命中降低时，这里的 prefill token 量和 GPU 计算时长会增加。",
    pressure: "Prefill token 与 GPU 时间",
  },
  {
    id: "write",
    title: "D2H + L3 write",
    subtitle: "发布新后缀",
    detail: "新算出的 KV 先备份到 host，再异步写入 L3。写入完成前 buffer 和树节点仍受保护；写失败意味着后续请求无法复用这批新页。",
    pressure: "写带宽、ACK 与保护时长",
  },
];

const PATH_SCENARIOS = {
  l1: {
    label: "L1 命中",
    active: ["request", "radix", "local", "compute", "write"],
    selected: "local",
  },
  l2: {
    label: "L2 命中",
    active: ["request", "radix", "local", "h2d", "compute", "write"],
    selected: "h2d",
  },
  l3: {
    label: "L3 命中",
    active: [
      "request",
      "radix",
      "local",
      "query",
      "allocate",
      "read",
      "h2d",
      "compute",
      "write",
    ],
    selected: "allocate",
  },
  miss: {
    label: "全部未命中",
    active: ["request", "radix", "local", "query", "compute", "write"],
    selected: "compute",
  },
};

function createElement(html) {
  const template = document.createElement("template");
  template.innerHTML = html.trim();
  return template.content.firstElementChild;
}

function createConceptFigure(src, alt, caption) {
  return createElement(`
    <figure class="concept-figure">
      <img src="${escapeHtml(src)}" alt="${escapeHtml(alt)}" loading="eager" />
      <figcaption>${icon("info")}<span>${escapeHtml(caption)}</span></figcaption>
    </figure>`);
}

function createOverviewStats(manifest) {
  const totals = manifest.totals;
  return createElement(`
    <div class="book-overview-stats" aria-label="书籍覆盖范围">
      <div><strong>${formatNumber(totals.coreDocuments)}</strong><span>核心章节</span></div>
      <div><strong>${formatNumber(totals.functions)}</strong><span>Python 函数</span></div>
      <div><strong>${formatNumber(totals.classes)}</strong><span>类声明</span></div>
      <div><strong>${formatNumber(totals.diagrams)}</strong><span>源码关系图</span></div>
    </div>`);
}

function createPathLab() {
  const element = createElement(`
    <section class="interactive-lab path-lab" aria-label="请求路径解剖器">
      <div class="lab-header">
        <div><h2>请求路径解剖器</h2><p>切换命中层级，查看真正执行的节点及其压力来源。</p></div>
        <span class="lab-badge">精确路径图</span>
      </div>
      <div class="scenario-tabs" role="group" aria-label="命中场景">
        ${Object.entries(PATH_SCENARIOS)
          .map(
            ([id, scenario], index) =>
              `<button type="button" data-scenario="${id}" class="${index === 2 ? "active" : ""}" aria-pressed="${index === 2}">${scenario.label}</button>`,
          )
          .join("")}
      </div>
      <div class="path-lab-body">
        <div class="path-map">
          ${PATH_NODES.map(
            (node) => `<button class="path-node" type="button" data-path-node="${node.id}">
                <strong>${node.title}</strong><small>${node.subtitle}</small>
              </button>`,
          ).join("")}
        </div>
        <aside class="path-inspector" aria-live="polite"></aside>
      </div>
      <p class="lab-footnote">路径按主分支状态机归纳；并行 rank、逐层 CUDA copy 和异步 ACK 在相应节点内部展开。</p>
    </section>`);

  let scenarioId = "l3";
  let selectedNodeId = PATH_SCENARIOS[scenarioId].selected;
  const inspector = element.querySelector(".path-inspector");

  const update = () => {
    const scenario = PATH_SCENARIOS[scenarioId];
    element.querySelectorAll("[data-scenario]").forEach((button) => {
      const selected = button.dataset.scenario === scenarioId;
      button.classList.toggle("active", selected);
      button.setAttribute("aria-pressed", String(selected));
    });
    element.querySelectorAll("[data-path-node]").forEach((button) => {
      const active = scenario.active.includes(button.dataset.pathNode);
      const selected = active && button.dataset.pathNode === selectedNodeId;
      button.classList.toggle("active", active);
      button.classList.toggle("skipped", !active);
      button.classList.toggle("selected", selected);
      button.setAttribute("aria-disabled", String(!active));
      button.setAttribute("aria-pressed", String(selected));
      button.tabIndex = active ? 0 : -1;
    });
    const node = PATH_NODES.find((candidate) => candidate.id === selectedNodeId);
    inspector.innerHTML = `
      <span class="inspector-label">${escapeHtml(scenario.label)} · 当前节点</span>
      <h4>${escapeHtml(node.title)}</h4>
      <p>${escapeHtml(node.detail)}</p>
      <span class="pressure-tag">${icon("activity")} ${escapeHtml(node.pressure)}</span>`;
    refreshIcons(inspector);
  };

  element.querySelectorAll("[data-scenario]").forEach((button) => {
    button.addEventListener("click", () => {
      scenarioId = button.dataset.scenario;
      selectedNodeId = PATH_SCENARIOS[scenarioId].selected;
      update();
    });
  });
  element.querySelectorAll("[data-path-node]").forEach((button) => {
    button.addEventListener("click", () => {
      if (!PATH_SCENARIOS[scenarioId].active.includes(button.dataset.pathNode)) return;
      selectedNodeId = button.dataset.pathNode;
      update();
    });
  });
  update();
  return element;
}

const RADIX_REQUESTS = {
  a: { label: "请求 A", tokens: "HiCache 是 / 分层 / 缓存", nodes: ["root", "prefix", "a", "a2"] },
  b: { label: "请求 B", tokens: "HiCache 是 / 如何 / 读写", nodes: ["root", "prefix", "b", "b2"] },
  c: { label: "请求 C", tokens: "HiCache 是 / 为什么 / 命中", nodes: ["root", "prefix", "c", "c2"] },
};

function createRadixLab() {
  const nodes = [
    ["root", "ROOT", "root shared"],
    ["prefix", "HiCache 是", "shared"],
    ["a", "分层", ""],
    ["a2", "缓存", ""],
    ["b", "如何", ""],
    ["b2", "读写", ""],
    ["c", "为什么", ""],
    ["c2", "命中", ""],
  ];
  const element = createElement(`
    <section class="interactive-lab radix-lab" aria-label="Radix 共享前缀实验">
      <div class="lab-header">
        <div><h2>Radix 共享前缀实验</h2><p>三个请求共享同一段前缀，树只为分叉后缀创建独立路径。</p></div>
        <span class="lab-badge">节点高亮</span>
      </div>
      <div class="radix-lab-body">
        <div class="request-list">
          ${Object.entries(RADIX_REQUESTS)
            .map(
              ([id, request], index) =>
                `<button type="button" data-request="${id}" class="${index === 0 ? "active" : ""}" aria-pressed="${index === 0}"><span>${request.label}</span><small>${request.tokens}</small></button>`,
            )
            .join("")}
        </div>
        <div class="radix-tree-canvas">
          <div class="radix-tree">
            ${nodes
              .map(
                ([id, label, classes]) =>
                  `<div class="radix-token ${classes}" data-node="${id}">${label}</div>`,
              )
              .join("")}
          </div>
        </div>
      </div>
      <p class="lab-footnote">绿色节点是所有请求共享的前缀；蓝色节点是当前请求实际遍历的后缀。树节点保存索引和状态，不保存 KV 字节。</p>
    </section>`);

  const selectRequest = (requestId) => {
    const request = RADIX_REQUESTS[requestId];
    element.querySelectorAll("[data-request]").forEach((button) => {
      const selected = button.dataset.request === requestId;
      button.classList.toggle("active", selected);
      button.setAttribute("aria-pressed", String(selected));
    });
    element.querySelectorAll("[data-node]").forEach((node) => {
      node.classList.toggle("active", request.nodes.includes(node.dataset.node));
      node.classList.toggle(
        "muted",
        !request.nodes.includes(node.dataset.node) && !node.classList.contains("shared"),
      );
    });
  };
  element.querySelectorAll("[data-request]").forEach((button) => {
    button.addEventListener("click", () => selectRequest(button.dataset.request));
  });
  selectRequest("a");
  return element;
}

function createPressureLab() {
  const element = createElement(`
    <section class="interactive-lab pressure-lab" aria-label="L2 并发压力估算器">
      <div class="lab-header">
        <div><h2>L2 并发压力上界</h2><p>把同时在途的读页和写页放进同一容量预算。</p></div>
        <span class="lab-badge">容量模型</span>
      </div>
      <div class="pressure-lab-body">
        <div class="pressure-controls">
          <div class="range-control"><label for="pressure-capacity"><span>L2 总页数</span><output for="pressure-capacity" data-output="capacity">800</output></label><input id="pressure-capacity" data-range="capacity" type="range" min="400" max="4000" step="100" value="800" aria-label="L2 总页数" /></div>
          <div class="range-control"><label for="pressure-concurrency"><span>并发请求</span><output for="pressure-concurrency" data-output="concurrency">32</output></label><input id="pressure-concurrency" data-range="concurrency" type="range" min="8" max="128" step="8" value="32" aria-label="并发请求数" /></div>
          <div class="range-control"><label for="pressure-read"><span>每请求在途读页</span><output for="pressure-read" data-output="read">8</output></label><input id="pressure-read" data-range="read" type="range" min="1" max="24" step="1" value="8" aria-label="每请求在途读页数" /></div>
          <div class="range-control"><label for="pressure-write"><span>每请求在途写页</span><output for="pressure-write" data-output="write">4</output></label><input id="pressure-write" data-range="write" type="range" min="1" max="16" step="1" value="4" aria-label="每请求在途写页数" /></div>
        </div>
        <div class="pressure-output">
          <div class="pressure-reading"><strong data-pressure-value>48%</strong><span data-pressure-pages>384 / 800 页</span></div>
          <div class="pressure-gauge" data-pressure-gauge role="meter" aria-label="L2 在途容量占比" aria-valuemin="0" aria-valuemax="100" aria-valuenow="48" aria-valuetext="48%，384 / 800 页"><span></span></div>
          <p class="pressure-diagnosis" data-pressure-diagnosis></p>
        </div>
      </div>
      <p class="lab-footnote">这是并发占页上界，不是运行时预测器。真实占用还受实际命中页数、批次重叠、I/O 完成速度、节点锁和释放 ACK 影响。</p>
    </section>`);

  const values = { capacity: 800, concurrency: 32, read: 8, write: 4 };
  const update = () => {
    const occupied = values.concurrency * (values.read + values.write);
    const ratio = occupied / values.capacity;
    const percentage = Math.round(ratio * 100);
    const gauge = element.querySelector("[data-pressure-gauge]");
    const diagnosis = element.querySelector("[data-pressure-diagnosis]");
    gauge.querySelector("span").style.width = `${Math.min(100, percentage)}%`;
    gauge.setAttribute("aria-valuenow", String(Math.min(100, percentage)));
    gauge.setAttribute(
      "aria-valuetext",
      `${percentage}%，${formatNumber(occupied)} / ${formatNumber(values.capacity)} 页`,
    );
    element.querySelector("[data-pressure-value]").textContent = `${percentage}%`;
    element.querySelector("[data-pressure-pages]").textContent = `${formatNumber(occupied)} / ${formatNumber(values.capacity)} 页`;
    let level = "normal";
    let message = "容量仍有余量，但还要为锁保护页、分配碎片和完成队列延迟留出空间。";
    if (ratio >= 1) {
      level = "danger";
      message = "在途需求已经超过 L2 容量上界，prefetch allocation 可能失败，L3 命中也无法全部转化为有效回载。";
    } else if (ratio >= 0.8) {
      level = "warning";
      message = "L2 接近饱和，I/O 尾延迟或 ACK 释放变慢都可能把短暂峰值推成 allocation loss。";
    }
    gauge.dataset.level = level;
    diagnosis.className = `pressure-diagnosis${level === "normal" ? "" : ` ${level}`}`;
    diagnosis.textContent = message;
  };

  element.querySelectorAll("[data-range]").forEach((input) => {
    input.addEventListener("input", () => {
      values[input.dataset.range] = Number(input.value);
      element.querySelector(`[data-output="${input.dataset.range}"]`).textContent = input.value;
      update();
    });
  });
  update();
  return element;
}

function insertSequenceAfter(anchor, elements) {
  let cursor = anchor;
  for (const element of elements) {
    cursor.insertAdjacentElement("afterend", element);
    cursor = element;
  }
}

export function injectChapterEnhancements(article, currentDocument, manifest) {
  const heading = article.querySelector("h1");
  if (!heading) return;
  const elements = [];

  if (currentDocument.id === "README") {
    elements.push(
      createConceptFigure(
        manifest.conceptAssets.dataJourney,
        "GPU、host memory 与分布式 L3 之间传输 KV pages 的概念图",
        "概念图用于建立三层空间感；层级键、状态发布和异步完成顺序以正文和精确流程图为准。",
      ),
      createOverviewStats(manifest),
      createPathLab(),
    );
  } else if (currentDocument.id === "04-radix-tree") {
    elements.push(
      createConceptFigure(
        manifest.conceptAssets.radixMap,
        "共享前缀分叉并指向不同缓存层级的概念图",
        "概念图强调“树是目录而不是 KV 数据”；精确字段和分裂不变量见本章源码分析。",
      ),
      createRadixLab(),
    );
  } else if (currentDocument.id === "05-request-lifecycle") {
    elements.push(createPathLab());
  } else if (currentDocument.id === "13-performance") {
    elements.push(createPressureLab());
  }

  insertSequenceAfter(heading, elements);
  elements.forEach((element) => refreshIcons(element));
}

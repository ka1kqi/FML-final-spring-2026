(function () {
  const ROLES = ["Top", "Jungle", "Mid", "ADC", "Support"];

  const kSelect = document.getElementById("similar-k");
  const neighborsGrid = document.getElementById("similar-grid");
  const emptyBox = document.getElementById("similar-message");
  const errBox = document.getElementById("similar-error");
  const loadingBox = document.getElementById("similar-loading");

  const anchorInput = document.getElementById("similar-search-anchor");
  const anchorDropdown = document.getElementById("similar-grid-anchor");
  const compareInput = document.getElementById("similar-search-compare");
  const compareDropdown = document.getElementById("similar-grid-compare");

  const anchorImg = document.getElementById("similar-anchor-img");
  const anchorName = document.getElementById("similar-anchor-name");
  const anchorRoles = document.getElementById("similar-anchor-roles");
  const wrBadge = document.getElementById("similar-wr-badge");

  const compareEmpty = document.getElementById("similar-compare-empty");
  const compareContent = document.getElementById("similar-compare-content");
  const compareAnchorImg = document.getElementById("similar-compare-anchor-img");
  const compareOtherImg = document.getElementById("similar-compare-other-img");
  const compareTitle = document.getElementById("similar-compare-title");
  const compareDesc = document.getElementById("similar-compare-desc");
  const compareCos = document.getElementById("similar-compare-cos");
  const compareRank = document.getElementById("similar-compare-rank");
  const compareShared = document.getElementById("similar-compare-shared");
  const clearCompareBtn = document.getElementById("similar-clear-compare");
  const compareSuggestions = document.getElementById("similar-compare-suggestions");

  const roleSlicesEmpty = document.getElementById("similar-role-slices-empty");
  const roleSlices = document.getElementById("similar-role-slices");

  let allChamps = [];
  let champByName = new Map(); // lower(name) -> champ

  let anchor = null;
  let compare = null;
  let roleFilter = "all";
  let lastNeighborsAll = []; // fetched with role=all, k=max
  let requestSeq = 0;

  function escapeHtml(s) {
    const d = document.createElement("div");
    d.textContent = s;
    return d.innerHTML;
  }

  function setVisible(el, on) {
    el.classList.toggle("hidden", !on);
  }

  function resetMessages() {
    setVisible(errBox, false);
    setVisible(loadingBox, false);
  }

  function showError(text) {
    errBox.textContent = text;
    setVisible(errBox, true);
    setVisible(emptyBox, true);
    emptyBox.textContent = "Choose a champion to see nearest neighbors.";
    setVisible(loadingBox, false);
    neighborsGrid.innerHTML = "";
    roleSlicesEmpty.classList.remove("hidden");
    roleSlices.classList.add("hidden");
  }

  function showEmpty(text) {
    emptyBox.textContent = text;
    setVisible(emptyBox, true);
    setVisible(errBox, false);
    setVisible(loadingBox, false);
    neighborsGrid.innerHTML = "";
    roleSlicesEmpty.classList.remove("hidden");
    roleSlices.classList.add("hidden");
  }

  function showLoading(on) {
    setVisible(loadingBox, on);
  }

  function formatWrBadge(wr) {
    if (wr == null || Number.isNaN(wr)) return null;
    return `${(wr * 100).toFixed(1)}%`;
  }

  function getChamp(name) {
    if (!name) return null;
    return champByName.get(String(name).toLowerCase()) || null;
  }

  function renderAnchor() {
    if (!anchor) {
      anchorName.textContent = "Select a champion";
      setVisible(anchorImg, false);
      setVisible(wrBadge, false);
      anchorRoles.innerHTML = "";
      return;
    }

    const champ = getChamp(anchor);
    anchorName.textContent = anchor;
    anchorInput.value = anchor;

    if (champ && champ.img) {
      anchorImg.src = champ.img;
      anchorImg.alt = anchor;
      setVisible(anchorImg, true);
    } else {
      setVisible(anchorImg, false);
    }

    anchorRoles.innerHTML = "";
    const roles = (champ && Array.isArray(champ.roles) ? champ.roles : []).slice().sort();
    if (roles.length === 0) {
      anchorRoles.innerHTML = `<span class="similar-chip muted">Unknown</span>`;
    } else {
      roles.forEach((r) => {
        const chip = document.createElement("span");
        chip.className = "similar-chip";
        chip.textContent = r;
        anchorRoles.appendChild(chip);
      });
    }
  }

  function setCompareButtonsVisibility() {
    // no-op placeholder (kept for future UX tweaks)
  }

  function setRoleButtons() {
    document.querySelectorAll(".analysis-role-selector .role-mini-btn").forEach((btn) => {
      const label = (btn.textContent || "").trim();
      const map = { All: "all", T: "Top", J: "Jungle", M: "Mid", A: "ADC", S: "Support" };
      const r = map[label] || "all";
      btn.classList.toggle("active", r === roleFilter);
    });
  }

  function passRoleFilter(name) {
    if (!roleFilter || roleFilter === "all") return true;
    const champ = getChamp(name);
    const roles = champ && Array.isArray(champ.roles) ? champ.roles : [];
    return roles.includes(roleFilter);
  }

  function currentNeighbors() {
    const k = parseInt(kSelect.value, 10) || 8;
    const filtered = lastNeighborsAll.filter((n) => passRoleFilter(n.name));
    return filtered.slice(0, k);
  }

  function renderNeighbors() {
    neighborsGrid.innerHTML = "";
    const list = currentNeighbors();
    if (!anchor) return;
    if (!list || list.length === 0) {
      showEmpty("No champions match this role filter — try All or another role.");
      return;
    }

    setVisible(emptyBox, false);
    resetMessages();
    list.forEach((n) => {
      const card = document.createElement("article");
      card.className = "similar-card clickable";
      const wrLine = formatWrBadge(n.historical_wr);
      card.innerHTML = `
        <div class="similar-card-actions">
          <button type="button" class="similar-compare-btn" aria-label="Compare ${escapeHtml(n.name)}">Compare</button>
        </div>
        <img src="${n.img}" alt="" loading="lazy" width="160" height="160">
        <div class="similar-card-body">
          <div class="similar-card-name">${escapeHtml(n.name)}</div>
          <div class="similar-card-meta">
            <span class="similar-card-sim">cosine ${n.similarity.toFixed(4)}</span>
            ${wrLine ? `<span>${escapeHtml(wrLine)} WR</span>` : ""}
          </div>
        </div>
      `;
      const btn = card.querySelector(".similar-compare-btn");
      if (btn) {
        btn.addEventListener("click", (e) => {
          e.preventDefault();
          e.stopPropagation();
          setCompare(n.name);
        });
      }
      card.addEventListener("click", () => setAnchor(n.name));
      neighborsGrid.appendChild(card);
    });
  }

  function renderRoleSlices() {
    if (!anchor || !lastNeighborsAll.length) {
      roleSlicesEmpty.classList.remove("hidden");
      roleSlices.classList.add("hidden");
      return;
    }

    roleSlicesEmpty.classList.add("hidden");
    roleSlices.classList.remove("hidden");

    ROLES.forEach((role) => {
      const container = document.getElementById(`slice-${role}`);
      if (!container) return;
      container.innerHTML = "";
      const top = lastNeighborsAll.filter((n) => {
        const champ = getChamp(n.name);
        const roles = champ && Array.isArray(champ.roles) ? champ.roles : [];
        return roles.includes(role);
      }).slice(0, 5);

      if (top.length === 0) {
        container.innerHTML = `<div class="similar-slice-empty">No matches</div>`;
        return;
      }

      top.forEach((n) => {
        const item = document.createElement("div");
        item.className = "meta-item";
        item.innerHTML = `
          <img src="${n.img}" alt="" loading="lazy" width="40" height="40">
          <div style="flex:1; min-width:0;">
            <div class="meta-item-name">${escapeHtml(n.name)}</div>
            <div class="meta-item-score" style="color: var(--gold);">cosine ${n.similarity.toFixed(4)}</div>
          </div>
          <button type="button" class="similar-compare-btn small" aria-label="Compare ${escapeHtml(n.name)}">Compare</button>
        `;
        const btn = item.querySelector(".similar-compare-btn");
        if (btn) {
          btn.addEventListener("click", (e) => {
            e.preventDefault();
            e.stopPropagation();
            setCompare(n.name);
          });
        }
        item.addEventListener("click", () => setAnchor(n.name));
        container.appendChild(item);
      });
    });
  }

  function sharedRoles(aName, bName) {
    const a = getChamp(aName);
    const b = getChamp(bName);
    const ar = a && Array.isArray(a.roles) ? a.roles : [];
    const br = b && Array.isArray(b.roles) ? b.roles : [];
    return ar.filter((r) => br.includes(r)).sort();
  }

  async function renderCompare() {
    if (!anchor) {
      compareEmpty.classList.remove("hidden");
      compareContent.classList.add("hidden");
      if (compareSuggestions) compareSuggestions.innerHTML = "";
      return;
    }

    if (!compare) {
      compareEmpty.classList.remove("hidden");
      compareContent.classList.add("hidden");

      if (compareSuggestions) {
        compareSuggestions.innerHTML = "";
        const picks = (lastNeighborsAll || []).slice(0, 8);
        picks.forEach((n) => {
          const chip = document.createElement("button");
          chip.type = "button";
          chip.className = "similar-suggest-chip";
          chip.innerHTML = `
            <img src="${n.img}" alt="" loading="lazy" width="22" height="22">
            <span>${escapeHtml(n.name)}</span>
          `;
          chip.addEventListener("click", () => setCompare(n.name));
          compareSuggestions.appendChild(chip);
        });
        if (picks.length === 0) {
          compareSuggestions.innerHTML = `<span class="similar-chip muted">Pick an anchor to see suggestions</span>`;
        }
      }
      return;
    }

    compareEmpty.classList.add("hidden");
    compareContent.classList.remove("hidden");

    const aChamp = getChamp(anchor);
    const bChamp = getChamp(compare);
    if (aChamp && aChamp.img) compareAnchorImg.src = aChamp.img;
    if (bChamp && bChamp.img) compareOtherImg.src = bChamp.img;

    compareTitle.textContent = `${anchor} vs ${compare}`;
    compareDesc.textContent = "Computing cosine similarity in Champion2Vec space…";
    compareCos.textContent = "—";
    compareRank.textContent = "—";
    compareShared.innerHTML = "";

    const shared = sharedRoles(anchor, compare);
    if (shared.length === 0) {
      compareShared.innerHTML = `<span class="similar-chip muted">None</span>`;
    } else {
      shared.forEach((r) => {
        const chip = document.createElement("span");
        chip.className = "similar-chip";
        chip.textContent = r;
        compareShared.appendChild(chip);
      });
    }

    // Rank in current neighbor list if present
    const idx = lastNeighborsAll.findIndex((n) => n.name === compare);
    if (idx >= 0) compareRank.textContent = `#${idx + 1} / ${lastNeighborsAll.length}`;

    try {
      const params = new URLSearchParams({ a: anchor, b: compare });
      const res = await fetch(`/api/similar_pair?${params}`);
      const data = await res.json();
      if (!res.ok) {
        compareDesc.textContent = data.error || "Failed to compute cosine.";
        return;
      }
      const cos = typeof data.cosine === "number" ? data.cosine : null;
      if (cos == null) {
        compareDesc.textContent = "Failed to compute cosine.";
        return;
      }
      compareCos.textContent = cos.toFixed(4);
      compareDesc.textContent = cos >= 0.55
        ? "Very close in embedding space (strong substitute candidates)."
        : cos >= 0.40
          ? "Moderately close in embedding space."
          : "Weak similarity in embedding space.";
    } catch (e) {
      compareDesc.textContent = "Network error while computing cosine.";
    }
  }

  async function fetchSimilarAll() {
    if (!anchor) return;
    const seq = ++requestSeq;
    resetMessages();
    showLoading(true);
    neighborsGrid.innerHTML = "";
    setVisible(emptyBox, false);
    try {
      const params = new URLSearchParams({ champion: anchor, k: "24" });
      const res = await fetch(`/api/similar?${params}`);
      const data = await res.json();
      if (seq !== requestSeq) return;
      if (!res.ok) {
        showError(data.error || "Request failed.");
        return;
      }
      lastNeighborsAll = Array.isArray(data.neighbors) ? data.neighbors : [];
      showLoading(false);
      setVisible(wrBadge, false);

      renderNeighbors();
      renderRoleSlices();
      renderCompare();
    } catch (e) {
      if (seq !== requestSeq) return;
      showError("Network error — try again.");
    } finally {
      if (seq === requestSeq) showLoading(false);
    }
  }

  function setAnchor(name) {
    const champ = getChamp(name);
    if (!champ) {
      showError("Unknown champion.");
      return;
    }
    anchor = champ.name;
    renderAnchor();
    fetchSimilarAll();
  }

  function setCompare(name) {
    const champ = getChamp(name);
    if (!champ) {
      showError("Unknown champion.");
      return;
    }
    compare = champ.name;
    compareInput.value = compare;
    renderCompare();
  }

  function clearCompare() {
    compare = null;
    compareInput.value = "";
    renderCompare();
  }

  // ---- Dropdown grid (Analysis-style) ----
  function renderGridToDropdown(container, query, onPick) {
    container.innerHTML = "";
    const q = (query || "").toLowerCase();
    const filtered = allChamps.filter((c) => c.name.toLowerCase().includes(q));
    filtered.forEach((champ) => {
      const cell = document.createElement("div");
      cell.className = "dropdown-grid-cell";
      cell.title = champ.name;
      cell.innerHTML = `<img src="${champ.img}" alt="${escapeHtml(champ.name)}">`;
      cell.onclick = () => {
        onPick(champ.name);
        container.classList.add("hidden");
      };
      container.appendChild(cell);
    });
  }

  window.showSimilarGridDropdown = function () {
    renderGridToDropdown(anchorDropdown, "", setAnchor);
    anchorDropdown.classList.remove("hidden");
  };

  window.filterSimilarGridDropdown = function (q) {
    renderGridToDropdown(anchorDropdown, q, setAnchor);
  };

  window.showSimilarCompareDropdown = function () {
    renderGridToDropdown(compareDropdown, "", setCompare);
    compareDropdown.classList.remove("hidden");
  };

  window.filterSimilarCompareDropdown = function (q) {
    renderGridToDropdown(compareDropdown, q, setCompare);
  };

  document.addEventListener("click", (e) => {
    if (!e.target.closest(".analysis-search")) {
      anchorDropdown.classList.add("hidden");
      compareDropdown.classList.add("hidden");
    }
  });

  window.setSimilarRole = function (role) {
    roleFilter = role || "all";
    setRoleButtons();
    renderNeighbors();
  };

  kSelect.addEventListener("change", () => {
    renderNeighbors();
  });

  async function init() {
    try {
      const res = await fetch("/api/champions");
      const data = await res.json();
      allChamps = (Array.isArray(data) ? data : data.champions || []).filter((c) => c && c.name);
      allChamps.sort((a, b) => a.name.localeCompare(b.name));
      champByName = new Map(allChamps.map((c) => [String(c.name).toLowerCase(), c]));

      const params = new URLSearchParams(window.location.search);
      const pre = (params.get("champion") || "").trim();
      if (pre) {
        const match = allChamps.find((c) => c.name.toLowerCase() === pre.toLowerCase());
        if (match) setAnchor(match.name);
        else showEmpty("Choose a champion to see nearest neighbors.");
      } else {
        showEmpty("Choose a champion to see nearest neighbors.");
      }
      setRoleButtons();
      renderCompare();
    } catch (e) {
      showError("Failed to load champions list.");
    }
  }

  clearCompareBtn.addEventListener("click", clearCompare);
  init();
})();

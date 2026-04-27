/* ================================================================
   LoL Draft Screen — Application Logic
   ================================================================ */

const DRAFT_ORDER = [
  { side: "Blue", slot: 0 },  // B1
  { side: "Red",  slot: 0 },  // R1
  { side: "Red",  slot: 1 },  // R2
  { side: "Blue", slot: 1 },  // B2
  { side: "Blue", slot: 2 },  // B3
  { side: "Red",  slot: 2 },  // R3
  { side: "Red",  slot: 3 },  // R4
  { side: "Blue", slot: 3 },  // B4
  { side: "Blue", slot: 4 },  // B5
  { side: "Red",  slot: 4 },  // R5
];

const ROLES = ["Top", "Jungle", "Mid", "ADC", "Support"];
const ROLE_SHORT = { Top: "TOP", Jungle: "JG", Mid: "MID", ADC: "ADC", Support: "SUP" };

// ---------- State ----------
let allChampions = [];
let bluePicks = [null, null, null, null, null];
let redPicks  = [null, null, null, null, null];
let blueBans  = [];
let redBans   = [];
let blueRoles = shuffleArray([...ROLES]);
let redRoles  = shuffleArray([...ROLES]);
let currentPickStep = 0;
let selectedChampion = null;
let searchQuery = "";
let roleFilter = null;
let phase = "ban-blue"; // ban-blue, ban-red, pick, complete
let recommendations = [];
let swapState = null; // { side, slot } if swapping
let recOffset = 0;

// ---------- Init ----------
document.addEventListener("DOMContentLoaded", async () => {
  allChampions = await fetch("/api/champions").then(r => r.json());
  render();
});

// ---------- Utility ----------
function shuffleArray(arr) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

function getUsed() {
  const used = new Set();
  bluePicks.forEach(p => p && used.add(p));
  redPicks.forEach(p => p && used.add(p));
  blueBans.forEach(b => b && b !== "__skip__" && used.add(b));
  redBans.forEach(b => b && b !== "__skip__" && used.add(b));
  return used;
}

function getChampImg(name) {
  const c = allChampions.find(ch => ch.name === name);
  return c ? c.img : "";
}

// ---------- API ----------
async function fetchRecommendations() {
  if (phase !== "pick" || currentPickStep >= DRAFT_ORDER.length) {
    recommendations = [];
    return;
  }
  const { side, slot } = DRAFT_ORDER[currentPickStep];
  const role = side === "Blue" ? blueRoles[slot] : redRoles[slot];

  try {
    const resp = await fetch("/api/recommend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        blue_picks: bluePicks,
        red_picks: redPicks,
        blue_bans: blueBans.filter(b => b && b !== "__skip__"),
        red_bans: redBans.filter(b => b && b !== "__skip__"),
        step: currentPickStep,
        role: role,
      }),
    });
    const data = await resp.json();
    recommendations = data.recommendations || [];
    recOffset = 0;
  } catch (e) {
    console.error("Failed to fetch recommendations:", e);
    recommendations = [];
  }
}

// ---------- Actions ----------
function selectChampion(name) {
  if (getUsed().has(name)) return;
  selectedChampion = name;
  render();
}

function lockIn() {
  if (!selectedChampion) return;

  if (phase === "ban-blue") {
    if (blueBans.length < 5) {
      blueBans.push(selectedChampion);
      selectedChampion = null;
      if (blueBans.length >= 5) phase = "ban-red";
    }
  } else if (phase === "ban-red") {
    if (redBans.length < 5) {
      redBans.push(selectedChampion);
      selectedChampion = null;
      if (redBans.length >= 5) {
        phase = "pick";
        currentPickStep = 0;
        fetchRecommendations().then(render);
      }
    }
  } else if (phase === "pick") {
    const { side, slot } = DRAFT_ORDER[currentPickStep];
    if (side === "Blue") {
      bluePicks[slot] = selectedChampion;
    } else {
      redPicks[slot] = selectedChampion;
    }
    selectedChampion = null;
    currentPickStep++;
    if (currentPickStep >= DRAFT_ORDER.length) {
      phase = "complete";
    }
  }

  render();
  if (phase === "pick") fetchRecommendations().then(render);
}

function skipBan() {
  if (phase === "ban-blue" && blueBans.length < 5) {
    blueBans.push("__skip__");
    if (blueBans.length >= 5) phase = "ban-red";
  } else if (phase === "ban-red" && redBans.length < 5) {
    redBans.push("__skip__");
    if (redBans.length >= 5) {
      phase = "pick";
      currentPickStep = 0;
      fetchRecommendations().then(render);
    }
  }
  selectedChampion = null;
  render();
  if (phase === "pick") fetchRecommendations().then(render);
}

function resetDraft() {
  bluePicks = [null, null, null, null, null];
  redPicks  = [null, null, null, null, null];
  blueBans  = [];
  redBans   = [];
  blueRoles = shuffleArray([...ROLES]);
  redRoles  = shuffleArray([...ROLES]);
  currentPickStep = 0;
  selectedChampion = null;
  phase = "ban-blue";
  recommendations = [];
  swapState = null;
  recOffset = 0;
  render();
}

function setRoleFilter(role) {
  roleFilter = roleFilter === role ? null : role;
  render();
}

function setSearch(q) {
  searchQuery = q.toLowerCase();
  render();
}

function swapRoles(side, slot) {
  const picks = side === "Blue" ? bluePicks : redPicks;

  // Can't swap a locked-in pick
  if (picks[slot]) return;

  if (swapState === null) {
    swapState = { side, slot };
    render();
    return;
  }
  if (swapState.side !== side) {
    swapState = null;
    render();
    return;
  }

  // Can't swap onto a locked-in pick either
  if (picks[swapState.slot]) {
    swapState = null;
    render();
    return;
  }

  const roles = side === "Blue" ? blueRoles : redRoles;
  const a = swapState.slot;
  const b = slot;
  [roles[a], roles[b]] = [roles[b], roles[a]];
  [picks[a], picks[b]] = [picks[b], picks[a]];
  swapState = null;
  render();

  // If we just swapped the currently active picking slot, we need new recommendations
  if (phase === "pick" && currentPickStep < DRAFT_ORDER.length) {
    const cs = DRAFT_ORDER[currentPickStep];
    if (cs.side === side && (cs.slot === a || cs.slot === b)) {
      fetchRecommendations().then(render);
    }
  }
}

// ---------- Render ----------
function render() {
  renderBans();
  renderTeam("blue");
  renderTeam("red");
  renderGrid();
  renderPhaseInfo();
  renderRecommendations();
  renderBottomActions();
  renderCompleteOverlay();
}

function renderBans() {
  const blueContainer = document.getElementById("blue-bans");
  const redContainer = document.getElementById("red-bans");

  blueContainer.innerHTML = "";
  redContainer.innerHTML = "";

  for (let i = 0; i < 5; i++) {
    blueContainer.appendChild(createBanSlot(blueBans[i], "blue", i));
    redContainer.appendChild(createBanSlot(redBans[i], "red", i));
  }
}

function createBanSlot(champ, side, idx) {
  const div = document.createElement("div");
  div.className = `ban-slot ${side}`;

  const bans = side === "blue" ? blueBans : redBans;
  const isActive = (phase === `ban-${side}` && idx === bans.length);
  if (isActive) div.classList.add("active");

  if (champ && champ !== "__skip__") {
    div.classList.add("filled");
    const img = document.createElement("img");
    img.src = getChampImg(champ);
    img.alt = champ;
    div.appendChild(img);
    const x = document.createElement("div");
    x.className = "ban-x";
    x.textContent = "✕";
    div.appendChild(x);
  } else if (champ === "__skip__") {
    div.classList.add("skipped");
    const skipText = document.createElement("div");
    skipText.className = "ban-skip-text";
    skipText.textContent = "NONE";
    div.appendChild(skipText);
  }
  return div;
}

function renderTeam(side) {
  const container = document.getElementById(`${side}-picks`);
  container.innerHTML = "";

  const picks = side === "blue" ? bluePicks : redPicks;
  const roles = side === "blue" ? blueRoles : redRoles;

  for (let i = 0; i < 5; i++) {
    const slot = document.createElement("div");
    slot.className = "pick-slot";

    // Check if this slot is the active pick
    if (phase === "pick" && currentPickStep < DRAFT_ORDER.length) {
      const cs = DRAFT_ORDER[currentPickStep];
      if (cs.side === (side === "blue" ? "Blue" : "Red") && cs.slot === i) {
        slot.classList.add("active");
      }
    }

    if (picks[i]) {
      slot.classList.add("filled");
    }

    if (swapState && swapState.side === (side === "blue" ? "Blue" : "Red") && swapState.slot === i) {
      slot.classList.add("swap-source");
    }

    // Portrait
    const imgDiv = document.createElement("div");
    imgDiv.className = "pick-slot-img";
    if (picks[i]) {
      const img = document.createElement("img");
      img.src = getChampImg(picks[i]);
      img.alt = picks[i];
      imgDiv.appendChild(img);
    }
    slot.appendChild(imgDiv);

    // Info
    const info = document.createElement("div");
    info.className = "pick-slot-info";
    const roleEl = document.createElement("div");
    roleEl.className = "pick-slot-role";
    roleEl.textContent = ROLE_SHORT[roles[i]] || roles[i];
    const nameEl = document.createElement("div");
    nameEl.className = "pick-slot-name";
    nameEl.textContent = picks[i] || "—";
    info.appendChild(roleEl);
    info.appendChild(nameEl);
    slot.appendChild(info);

    // Swap button
    const swapBtn = document.createElement("div");
    swapBtn.className = "pick-slot-swap";
    swapBtn.textContent = "⇄";
    swapBtn.title = "Swap role";
    swapBtn.onclick = (e) => {
      e.stopPropagation();
      swapRoles(side === "blue" ? "Blue" : "Red", i);
    };
    slot.appendChild(swapBtn);

    container.appendChild(slot);
  }
}

function renderGrid() {
  const grid = document.getElementById("champion-grid");
  grid.innerHTML = "";

  // Update role filter button states
  ROLES.forEach(role => {
    const btn = document.getElementById(`filter-${role}`);
    if (btn) {
      btn.classList.toggle("active", roleFilter === role);
    }
  });

  const used = getUsed();
  const bannedSet = new Set([...blueBans, ...redBans].filter(b => b && b !== "__skip__"));

  let filtered = allChampions;

  if (searchQuery) {
    filtered = filtered.filter(c => c.name.toLowerCase().includes(searchQuery));
  }

  if (roleFilter) {
    filtered = filtered.filter(c => c.roles.includes(roleFilter));
  }

  filtered.forEach(champ => {
    const cell = document.createElement("div");
    cell.className = "champ-cell";

    if (bannedSet.has(champ.name)) {
      cell.classList.add("banned-visual");
    } else if (used.has(champ.name)) {
      cell.classList.add("disabled");
    }

    if (selectedChampion === champ.name) {
      cell.classList.add("selected");
    }

    const img = document.createElement("img");
    img.src = champ.img;
    img.alt = champ.name;
    img.loading = "lazy";
    cell.appendChild(img);

    const tooltip = document.createElement("div");
    tooltip.className = "champ-tooltip";
    tooltip.textContent = champ.name;
    cell.appendChild(tooltip);

    if (!used.has(champ.name) && !bannedSet.has(champ.name)) {
      cell.onclick = () => selectChampion(champ.name);
    }

    grid.appendChild(cell);
  });
}

function renderPhaseInfo() {
  const indicator = document.getElementById("phase-indicator");
  const text = document.getElementById("phase-text");

  if (phase === "ban-blue") {
    indicator.textContent = "BAN PHASE";
    indicator.className = "phase-indicator ban-phase-active";
    text.innerHTML = `<span class="side-name blue">Blue</span> team banning (${blueBans.length}/5)`;
  } else if (phase === "ban-red") {
    indicator.textContent = "BAN PHASE";
    indicator.className = "phase-indicator ban-phase-active";
    text.innerHTML = `<span class="side-name red">Red</span> team banning (${redBans.length}/5)`;
  } else if (phase === "pick") {
    indicator.textContent = "PICK PHASE";
    indicator.className = "phase-indicator";
    if (currentPickStep < DRAFT_ORDER.length) {
      const { side, slot } = DRAFT_ORDER[currentPickStep];
      const roles = side === "Blue" ? blueRoles : redRoles;
      const sideClass = side.toLowerCase();
      text.innerHTML = `<span class="side-name ${sideClass}">${side}</span> picks <strong>${roles[slot]}</strong> — Step ${currentPickStep + 1}/10`;
    }
  } else {
    indicator.textContent = "COMPLETE";
    indicator.className = "phase-indicator";
    text.textContent = "Draft complete";
  }
}

function renderRecommendations() {
  const list = document.getElementById("rec-list");
  list.innerHTML = "";

  if (phase !== "pick" || recommendations.length === 0) {
    const empty = document.createElement("div");
    empty.style.cssText = "color: var(--text-dim); font-size: 12px; padding: 8px;";
    empty.textContent = phase === "pick" ? "Loading recommendations..." : "Recommendations appear during pick phase";
    list.appendChild(empty);
    return;
  }

  const visibleRecs = recommendations.slice(recOffset, recOffset + 5);

  if (recOffset > 0) {
    const prevBtn = document.createElement("button");
    prevBtn.className = "load-more-rec-btn";
    prevBtn.textContent = "Prev";
    prevBtn.title = "Show previous 5 recommendations";
    prevBtn.onclick = () => {
      recOffset = Math.max(0, recOffset - 5);
      renderRecommendations();
    };
    list.appendChild(prevBtn);
  }

  visibleRecs.forEach((rec, i) => {
    const card = document.createElement("div");
    card.className = "rec-card";
    card.onclick = () => selectChampion(rec.champion);

    const imgDiv = document.createElement("div");
    imgDiv.className = "rec-card-img";
    const img = document.createElement("img");
    img.src = getChampImg(rec.champion);
    imgDiv.appendChild(img);
    card.appendChild(imgDiv);

    const info = document.createElement("div");
    info.className = "rec-card-info";

    const name = document.createElement("div");
    name.className = "rec-card-name";
    name.textContent = rec.champion;
    info.appendChild(name);

    const prob = document.createElement("div");
    const pct = (rec.win_prob * 100).toFixed(1);
    const score = rec.score.toFixed(1);
    prob.innerHTML = `<span style="font-size: 10px; color: var(--text-dim);">S: ${score}</span> &nbsp;${pct}%`;
    prob.className = "rec-card-prob";
    if (rec.win_prob >= 0.52) prob.classList.add("high");
    else if (rec.win_prob >= 0.48) prob.classList.add("mid");
    else prob.classList.add("low");
    info.appendChild(prob);

    card.appendChild(info);
    list.appendChild(card);
  });

  if (recOffset + 5 < recommendations.length) {
    const moreBtn = document.createElement("button");
    moreBtn.className = "load-more-rec-btn";
    moreBtn.textContent = "Next";
    moreBtn.title = "Show next 5 recommendations";
    moreBtn.onclick = () => {
      recOffset += 5;
      renderRecommendations();
    };
    list.appendChild(moreBtn);
  }
}

function renderBottomActions() {
  const lockBtn = document.getElementById("lock-btn");
  const skipBtn = document.getElementById("skip-ban-btn");

  if (phase === "ban-blue" || phase === "ban-red") {
    lockBtn.textContent = "BAN";
    lockBtn.className = "lock-btn ban-mode";
    lockBtn.disabled = !selectedChampion;
    lockBtn.onclick = lockIn;
    skipBtn.classList.remove("hidden");
    skipBtn.onclick = skipBan;
  } else if (phase === "pick") {
    lockBtn.textContent = "LOCK IN";
    lockBtn.className = "lock-btn";
    lockBtn.disabled = !selectedChampion;
    lockBtn.onclick = lockIn;
    skipBtn.classList.add("hidden");
  } else {
    lockBtn.textContent = "COMPLETE";
    lockBtn.className = "lock-btn";
    lockBtn.disabled = true;
    skipBtn.classList.add("hidden");
  }
}

function renderCompleteOverlay() {
  const overlay = document.getElementById("complete-overlay");
  if (phase !== "complete") {
    overlay.classList.add("hidden");
    return;
  }
  overlay.classList.remove("hidden");

  // Fetch final prediction
  fetch("/api/evaluate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      blue_picks: bluePicks,
      red_picks: redPicks,
    }),
  })
  .then(res => res.json())
  .then(data => {
    const blueBar = document.getElementById("win-bar-blue");
    const redBar = document.getElementById("win-bar-red");
    const bluePct = (data.blue_win_prob * 100).toFixed(1);
    const redPct = (data.red_win_prob * 100).toFixed(1);
    const blueScore = data.blue_score.toFixed(1);
    const redScore = data.red_score.toFixed(1);
    
    blueBar.style.width = `${bluePct}%`;
    blueBar.innerHTML = `<span style="opacity: 0.8; margin-right: 6px;">[S: ${blueScore}]</span> ${bluePct}%`;
    
    redBar.style.width = `${redPct}%`;
    redBar.innerHTML = `${redPct}% <span style="opacity: 0.8; margin-left: 6px;">[S: ${redScore}]</span>`;
  })
  .catch(err => console.error("Evaluate error:", err));

  // Display teams
  const blueTeam = document.getElementById("result-blue-team");
  const redTeam = document.getElementById("result-red-team");
  blueTeam.innerHTML = bluePicks.map(p => `
    <div style="text-align:center">
      <img src="${getChampImg(p)}" style="width:48px;height:48px;border-radius:50%;border:2px solid var(--blue-team)">
      <div style="font-size:10px;margin-top:4px">${p}</div>
    </div>
  `).join("");
  redTeam.innerHTML = redPicks.map(p => `
    <div style="text-align:center">
      <img src="${getChampImg(p)}" style="width:48px;height:48px;border-radius:50%;border:2px solid var(--red-team)">
      <div style="font-size:10px;margin-top:4px">${p}</div>
    </div>
  `).join("");
}

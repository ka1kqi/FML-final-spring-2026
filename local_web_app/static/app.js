/* ==========================================================================
   LoL Draft AI — Vanilla-JS frontend.

   Talks to the Flask backend on the same origin:
     GET  /api/meta
     GET  /api/champions
     POST /api/recommend
     POST /api/evaluate

   The client owns the draft order (Tournament Draft) and the selection state.
   Every UI action that changes draft state triggers an /api/recommend or
   /api/evaluate call to keep the win-prob gauges and recommendation list
   live.
   ========================================================================== */

const ROLES = ['top', 'jungle', 'mid', 'adc', 'support'];
const ROLE_EMOJI = { top: '⚔️', jungle: '🌲', mid: '✨', adc: '🏹', support: '🛡️' };

// Tournament Draft order: B1R1B2R2B3R3 → B1 R1 R2 B2 B3 R3 → R4 B4 R5 B5 → R4 B4 B5 R5
const DRAFT_ORDER = [
  { type: 'ban',  side: 'blue', slot: 0 }, { type: 'ban',  side: 'red',  slot: 0 },
  { type: 'ban',  side: 'blue', slot: 1 }, { type: 'ban',  side: 'red',  slot: 1 },
  { type: 'ban',  side: 'blue', slot: 2 }, { type: 'ban',  side: 'red',  slot: 2 },
  { type: 'pick', side: 'blue', slot: 0 }, { type: 'pick', side: 'red',  slot: 0 }, { type: 'pick', side: 'red', slot: 1 },
  { type: 'pick', side: 'blue', slot: 1 }, { type: 'pick', side: 'blue', slot: 2 }, { type: 'pick', side: 'red', slot: 2 },
  { type: 'ban',  side: 'red',  slot: 3 }, { type: 'ban',  side: 'blue', slot: 3 },
  { type: 'ban',  side: 'red',  slot: 4 }, { type: 'ban',  side: 'blue', slot: 4 },
  { type: 'pick', side: 'red',  slot: 3 }, { type: 'pick', side: 'blue', slot: 3 },
  { type: 'pick', side: 'blue', slot: 4 }, { type: 'pick', side: 'red',  slot: 4 },
];

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
const state = {
  step: 0,
  blueBans: [null, null, null, null, null],
  redBans:  [null, null, null, null, null],
  // Pick slot index -> {role, champion}; role is fixed per slot order
  blueRoleForSlot: [...ROLES],
  redRoleForSlot:  [...ROLES],
  bluePicks: [null, null, null, null, null],
  redPicks:  [null, null, null, null, null],
  // UI-side
  champions: [],
  championsByName: {},
  selected: null,           // candidate awaiting Lock
  search: '',
  roleFilter: 'all',
  available: [],            // model names from /api/meta
  model: null,
  algorithm: 'greedy',
};

// ---------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------
async function boot() {
  const [meta, champs] = await Promise.all([
    fetch('/api/meta').then(r => r.json()),
    fetch('/api/champions').then(r => r.json()),
  ]);
  state.champions = champs;
  state.championsByName = Object.fromEntries(champs.map(c => [c.name, c]));
  state.available = meta.available_models || [];
  state.model = state.available.includes('wide_deep') ? 'wide_deep' : state.available[0];

  hydrateMetaBar(meta);
  hydrateModelSelector();
  document.getElementById('sel-algo').addEventListener('change', e => {
    state.algorithm = e.target.value;
    document.getElementById('rec-algo-tag').textContent = state.algorithm;
    refresh();
  });

  renderBans();
  renderPicks();
  renderGrid();
  await refresh();
}

function hydrateMetaBar(meta) {
  const summary = meta.summary || {};
  const models = summary.models || {};
  const schema = summary.schema || {};
  const rec = summary.recommender || {};

  const best = summary.best_model;
  const auc = summary.best_test_auc;
  document.getElementById('m-auc').textContent     = auc != null ? auc.toFixed(4) : '—';
  document.getElementById('m-recall').textContent  = rec['recall@5'] != null ? (rec['recall@5'] * 100).toFixed(1) + '%' : '—';
  document.getElementById('m-matches').textContent = schema.matches != null ? schema.matches.toLocaleString() : '—';
  document.getElementById('m-split').textContent   = (schema.split && schema.split.method) ? schema.split.method.replace('_', '-') : '—';
}

function hydrateModelSelector() {
  const sel = document.getElementById('sel-model');
  sel.innerHTML = '';
  state.available.forEach(m => {
    const opt = document.createElement('option');
    opt.value = m;
    opt.textContent = m;
    if (m === state.model) opt.selected = true;
    sel.appendChild(opt);
  });
  sel.addEventListener('change', e => { state.model = e.target.value; refresh(); });
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------
function renderBans() {
  for (const side of ['blue', 'red']) {
    const root = document.getElementById(`${side}-bans`);
    const arr = side === 'blue' ? state.blueBans : state.redBans;
    root.innerHTML = '';
    for (let i = 0; i < 5; i++) {
      const slot = document.createElement('div');
      slot.className = 'ban-slot' + (arr[i] ? '' : ' empty');
      const action = DRAFT_ORDER[state.step];
      if (action && action.type === 'ban' && action.side === side && action.slot === i) {
        slot.classList.add('active');
      }
      if (arr[i]) {
        const c = state.championsByName[arr[i]];
        if (c) slot.innerHTML = `<img src="${c.img}" alt="${c.name}">`;
        else   slot.textContent = arr[i];
      }
      root.appendChild(slot);
    }
  }
}

function renderPicks() {
  for (const side of ['blue', 'red']) {
    const root = document.getElementById(`${side}-picks`);
    const picks = side === 'blue' ? state.bluePicks : state.redPicks;
    const roles = side === 'blue' ? state.blueRoleForSlot : state.redRoleForSlot;
    root.innerHTML = '';
    for (let i = 0; i < 5; i++) {
      const slot = document.createElement('div');
      slot.className = 'pick-slot';
      const action = DRAFT_ORDER[state.step];
      if (action && action.type === 'pick' && action.side === side && action.slot === i) {
        slot.classList.add('active');
      }
      const role = roles[i];
      const champ = picks[i];
      const c = champ ? state.championsByName[champ] : null;
      slot.innerHTML = `
        <div class="pick-portrait">${c ? `<img src="${c.img}">` : ROLE_EMOJI[role]}</div>
        <div class="pick-info">
          <div class="pick-role">${role}</div>
          <div class="pick-name${champ ? '' : ' empty'}">${champ || 'Empty'}</div>
        </div>`;
      root.appendChild(slot);
    }
  }
}

function renderGrid() {
  const grid = document.getElementById('champion-grid');
  const usedSet = currentUsedChampions();
  const q = state.search.trim().toLowerCase();
  grid.innerHTML = '';
  for (const c of state.champions) {
    if (q && !c.name.toLowerCase().includes(q)) continue;
    if (state.roleFilter !== 'all') {
      // Show champions with no role data (treat as universal) AND those with the role
      if (c.roles && c.roles.length && !c.roles.includes(state.roleFilter)) continue;
    }
    const cell = document.createElement('div');
    cell.className = 'champ-cell';
    if (state.selected === c.name) cell.classList.add('selected');
    if (usedSet.has(c.name)) cell.classList.add('disabled');
    cell.innerHTML = `<img src="${c.img}" alt="${c.name}"
                       onerror="this.style.display='none'">
                      <div class="champ-name-overlay">${c.name}</div>`;
    cell.title = c.name;
    cell.addEventListener('click', () => {
      if (usedSet.has(c.name)) return;
      state.selected = (state.selected === c.name) ? null : c.name;
      renderGrid();
      updateLockButton();
    });
    grid.appendChild(cell);
  }
}

function renderRecommendations(payload) {
  const list = document.getElementById('rec-list');
  document.getElementById('rec-algo-tag').textContent = state.algorithm;
  if (!payload || !payload.recommendations || !payload.recommendations.length) {
    list.innerHTML = `<div class="rec-empty">${
      DRAFT_ORDER[state.step]?.type === 'pick'
        ? 'No legal candidates.'
        : 'Recommendations appear during pick phase.'
    }</div>`;
    return;
  }
  list.innerHTML = '';
  for (const r of payload.recommendations) {
    const c = state.championsByName[r.champion] || { img: '', name: r.champion };
    const card = document.createElement('div');
    card.className = 'rec-card';
    const deltaCls = r.delta > 0 ? 'rec-delta-pos' : (r.delta < 0 ? 'rec-delta-neg' : '');
    const deltaTxt = (r.delta >= 0 ? '+' : '') + r.delta.toFixed(3);
    card.innerHTML = `
      <div class="rec-portrait"><img src="${c.img}" onerror="this.style.display='none'"></div>
      <div>
        <div class="rec-meta-line">
          <span class="rec-name">${r.champion}</span>
          <span class="rec-wp">${(r.win_prob * 100).toFixed(1)}%</span>
          <span class="${deltaCls}">${deltaTxt}</span>
        </div>
        <div class="rec-notes">${r.notes || ''}</div>
      </div>`;
    card.addEventListener('click', () => {
      const usedSet = currentUsedChampions();
      if (usedSet.has(r.champion)) return;
      state.selected = r.champion;
      renderGrid();
      updateLockButton();
    });
    list.appendChild(card);
  }
}

function renderPhase(payload) {
  const action = DRAFT_ORDER[state.step];
  const ind = document.getElementById('phase-indicator');
  const txt = document.getElementById('phase-text');
  if (!action) {
    ind.textContent = 'COMPLETE';
    ind.className = 'phase-indicator';
    txt.innerHTML = 'All 20 actions locked. See final win prob below.';
    return;
  }
  const verb = action.type === 'ban' ? 'BAN' : 'PICK';
  ind.className = 'phase-indicator' + (action.type === 'pick'
    ? ' pick'
    : (action.side === 'red' ? ' banred' : ''));
  ind.textContent = `${action.side.toUpperCase()} ${verb}`;
  let detail = `Step ${state.step + 1} / 20 — ${action.side} ${verb} #${action.slot + 1}`;
  if (action.type === 'pick') {
    const role = (action.side === 'blue' ? state.blueRoleForSlot : state.redRoleForSlot)[action.slot];
    detail += ` (${ROLE_EMOJI[role]} <strong>${role}</strong>)`;
  }
  txt.innerHTML = detail;
}

function renderWinProbs(payload) {
  const blue = payload?.current_blue_winprob;
  if (blue == null) return;
  document.getElementById('blue-wp').textContent = (blue * 100).toFixed(1) + '%';
  document.getElementById('red-wp').textContent = ((1 - blue) * 100).toFixed(1) + '%';
}

// ---------------------------------------------------------------------------
// API
// ---------------------------------------------------------------------------
function buildPayload() {
  const blue_picks = {}, red_picks = {};
  state.bluePicks.forEach((c, i) => { if (c) blue_picks[state.blueRoleForSlot[i]] = c; });
  state.redPicks.forEach((c, i)  => { if (c) red_picks[state.redRoleForSlot[i]]  = c; });
  const bans = [...state.blueBans, ...state.redBans].filter(Boolean);
  return { blue_picks, red_picks, bans, model: state.model };
}

async function refresh() {
  renderBans();
  renderPicks();
  renderGrid();
  const action = DRAFT_ORDER[state.step];
  if (!action) {
    renderPhase(null);
    showCompletionOverlay();
    return;
  }
  document.getElementById('complete-overlay').classList.add('hidden');
  let payload;
  if (action.type === 'pick') {
    const role = (action.side === 'blue' ? state.blueRoleForSlot : state.redRoleForSlot)[action.slot];
    const body = {
      ...buildPayload(),
      side: action.side, role,
      top_k: 5,
      algorithm: state.algorithm,
      beam_width: 5, beam_depth: 2,
      mcts_simulations: 64,
    };
    payload = await fetch('/api/recommend', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }).then(r => r.json()).catch(_ => ({}));
  } else {
    // Ban phase: just evaluate current win prob, hide AI picks.
    payload = await fetch('/api/evaluate', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(buildPayload()),
    }).then(r => r.json()).catch(_ => ({}));
    payload.current_blue_winprob = payload?.blue_win_prob;
  }
  renderPhase(payload);
  renderWinProbs(payload);
  renderRecommendations(action.type === 'pick' ? payload : null);
  updateLockButton();
}

async function showCompletionOverlay() {
  const overlay = document.getElementById('complete-overlay');
  overlay.classList.remove('hidden');
  const finalEval = await fetch('/api/evaluate', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(buildPayload()),
  }).then(r => r.json()).catch(_ => ({ blue_win_prob: 0.5, red_win_prob: 0.5 }));
  const bluePct = (finalEval.blue_win_prob * 100).toFixed(1);
  const redPct  = (finalEval.red_win_prob * 100).toFixed(1);
  document.getElementById('win-bar-blue').style.width = bluePct + '%';
  document.getElementById('win-bar-blue').textContent = bluePct + '%';
  document.getElementById('win-bar-red').style.width = redPct + '%';
  document.getElementById('win-bar-red').textContent = redPct + '%';

  const fillRow = (id, picks) => {
    const root = document.getElementById(id);
    root.innerHTML = '';
    for (const c of picks) {
      const champ = c ? state.championsByName[c] : null;
      const div = document.createElement('div');
      div.className = 'pick-portrait';
      div.innerHTML = champ ? `<img src="${champ.img}">` : '?';
      root.appendChild(div);
    }
  };
  fillRow('result-blue-team', state.bluePicks);
  fillRow('result-red-team',  state.redPicks);
}

// ---------------------------------------------------------------------------
// UI controls
// ---------------------------------------------------------------------------
function currentUsedChampions() {
  return new Set([...state.bluePicks, ...state.redPicks, ...state.blueBans, ...state.redBans].filter(Boolean));
}

function updateLockButton() {
  const btn = document.getElementById('lock-btn');
  const action = DRAFT_ORDER[state.step];
  if (!action) {
    btn.disabled = true;
    btn.textContent = 'DONE';
    btn.className = 'lock-btn';
    return;
  }
  btn.disabled = !state.selected;
  btn.textContent = action.type === 'ban' ? 'BAN' : 'LOCK PICK';
  btn.className = 'lock-btn ' + (action.type === 'ban' ? 'ban-mode' : 'pick-mode');
}

function setSearch(v) { state.search = v; renderGrid(); }
function setRoleFilter(r) {
  state.roleFilter = r;
  document.querySelectorAll('.role-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.role === r);
  });
  renderGrid();
}

function lockAction() {
  if (!state.selected) return;
  const action = DRAFT_ORDER[state.step];
  if (!action) return;
  const arr =
    action.type === 'ban'
      ? (action.side === 'blue' ? state.blueBans : state.redBans)
      : (action.side === 'blue' ? state.bluePicks : state.redPicks);
  arr[action.slot] = state.selected;
  state.selected = null;
  state.step += 1;
  refresh();
}

function skipBan() {
  const action = DRAFT_ORDER[state.step];
  if (!action || action.type !== 'ban') return;
  state.step += 1;
  state.selected = null;
  refresh();
}

function resetDraft() {
  state.step = 0;
  state.blueBans = [null, null, null, null, null];
  state.redBans  = [null, null, null, null, null];
  state.bluePicks = [null, null, null, null, null];
  state.redPicks  = [null, null, null, null, null];
  state.selected = null;
  document.getElementById('complete-overlay').classList.add('hidden');
  refresh();
}

document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('lock-btn').addEventListener('click', lockAction);
  boot();
});

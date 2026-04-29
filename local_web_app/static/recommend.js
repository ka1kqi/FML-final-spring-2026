/* ============================================================
   Recommendation Analyzer — click-driven UI.

   State machine
   -------------
   activeTarget can be:
     null                                  — nothing selected
     { type: 'pick', side, role }          — empty pick slot waiting to be filled
     { type: 'ban' }                       — "ban mode": next champion click bans

   Click flow
   ----------
     • empty slot         → activeTarget = pick(side, role)
     • filled slot        → clear it, activeTarget = pick(side, role)
     • "+ Add ban" button → activeTarget = ban (or toggle off)
     • banned champion ×  → remove
     • champion in grid   → fill activeTarget (pick or ban)
     • Recommend button   → call /api/recommend with the active pick slot
     • result card click  → fill activeTarget pick slot with that champion
   ============================================================ */

const ROLES = ['top', 'jungle', 'mid', 'adc', 'support'];
const ROLE_EMOJI = { top: '⚔️', jungle: '🌲', mid: '✨', adc: '🏹', support: '🛡️' };

const state = {
  bluePicks:    [null, null, null, null, null],   // index = ROLES order
  redPicks:     [null, null, null, null, null],
  bans:         [],
  activeTarget: null,
  champions:    [],
  byName:       {},
  available:    [],
  lastResponse: null,
  selectedIndex: -1,
  search:       '',
  roleFilter:   'all',
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
  state.byName    = Object.fromEntries(champs.map(c => [c.name, c]));
  state.available = meta.available_models || [];

  hydrateMetaBar(meta);
  hydrateModelSelector();

  document.getElementById('run-btn').addEventListener('click', runRecommend);
  // ban-add-btn handler is wired in renderBans (it gets re-created on each render)

  render();
  refreshWinProb();
}

function hydrateMetaBar(meta) {
  const summary = meta.summary || {};
  const schema  = summary.schema || {};
  const rec     = summary.recommender || {};
  document.getElementById('m-auc').textContent     = summary.best_test_auc != null ? summary.best_test_auc.toFixed(4) : '—';
  document.getElementById('m-recall').textContent  = rec['recall@5'] != null ? (rec['recall@5'] * 100).toFixed(1) + '%' : '—';
  document.getElementById('m-matches').textContent = schema.matches != null ? schema.matches.toLocaleString() : '—';
  document.getElementById('m-split').textContent   = schema.split?.method ? schema.split.method.replace('_', '-') : '—';

  // If any models were skipped at server startup (e.g. PyTorch missing for
  // wide_deep), surface why so the user can fix their env instead of seeing
  // a confusing default model.
  const skipped = meta.skipped_models || {};
  const names = Object.keys(skipped);
  const old = document.getElementById('skip-banner');
  if (old) old.remove();
  if (names.length) {
    const div = document.createElement('div');
    div.id = 'skip-banner';
    div.className = 'flat-notice';
    div.innerHTML = '<b>Some models are not loaded on this server:</b><br>' +
      names.map(n => `<code>${n}</code> — ${skipped[n]}`).join('<br>');
    document.querySelector('.rec-page').prepend(div);
  }
}

function hydrateModelSelector() {
  const sel = document.getElementById('pick-model');
  sel.innerHTML = '';
  for (const m of state.available) {
    const opt = document.createElement('option');
    opt.value = m;
    opt.textContent = m;
    if (m === 'wide_deep') opt.selected = true;
    sel.appendChild(opt);
  }
}

// ---------------------------------------------------------------------------
// Renderers
// ---------------------------------------------------------------------------
function render() {
  renderSlots();
  renderBans();
  renderGrid();
  updateActiveInfo();
  updateRunButton();
}

function renderSlots() {
  for (const side of ['blue', 'red']) {
    const root = document.getElementById(`${side}-rec-slots`);
    const arr  = state[`${side}Picks`];
    root.innerHTML = '';
    ROLES.forEach((role, i) => {
      const filled   = !!arr[i];
      const isActive = state.activeTarget?.type === 'pick'
                    && state.activeTarget.side === side
                    && state.activeTarget.role === role;
      const slot     = document.createElement('div');
      slot.className = 'rec-pick-slot'
                     + (filled ? ' filled' : ' empty')
                     + (isActive ? ' active' : '');
      const c        = filled ? state.byName[arr[i]] : null;
      const portrait = c
        ? `<img src="${c.img}" onerror="this.style.display='none'">`
        : `<span class="role-glyph">${ROLE_EMOJI[role]}</span>`;
      slot.innerHTML = `
        <div class="rec-portrait">${portrait}</div>
        <div class="rec-role">${role}</div>
        <div class="rec-name">${arr[i] || (isActive ? '— picking —' : 'empty')}</div>`;
      slot.onclick = () => onSlotClick(side, role, i);
      root.appendChild(slot);
    });
  }
}

function renderBans() {
  const list = document.getElementById('rec-bans-list');
  list.innerHTML = '';
  for (const ban of state.bans) {
    const c    = state.byName[ban];
    const chip = document.createElement('div');
    chip.className = 'ban-chip';
    chip.innerHTML = `
      <div class="ban-chip-img">${c ? `<img src="${c.img}" onerror="this.style.display='none'">` : ''}</div>
      <span class="ban-chip-name">${ban}</span>
      <span class="ban-chip-x" title="remove ban">×</span>`;
    chip.querySelector('.ban-chip-x').onclick = (ev) => {
      ev.stopPropagation();
      state.bans = state.bans.filter(b => b !== ban);
      hideResults();
      render();
      refreshWinProb();
    };
    list.appendChild(chip);
  }
  const btn = document.createElement('button');
  btn.className = 'ban-add-btn' + (state.activeTarget?.type === 'ban' ? ' active' : '');
  btn.textContent = state.activeTarget?.type === 'ban'
    ? '× ban mode — click a champion below to ban'
    : '+ click here, then click champion to ban';
  btn.onclick = () => {
    state.activeTarget = state.activeTarget?.type === 'ban' ? null : { type: 'ban' };
    render();
  };
  list.appendChild(btn);
}

function renderGrid() {
  const grid = document.getElementById('champion-grid');
  const used = currentUsedChampions();
  const q    = state.search.trim().toLowerCase();
  grid.innerHTML = '';
  for (const c of state.champions) {
    if (q && !c.name.toLowerCase().includes(q)) continue;
    if (state.roleFilter !== 'all'
        && c.roles && c.roles.length
        && !c.roles.includes(state.roleFilter)) continue;
    const cell = document.createElement('div');
    cell.className = 'champ-cell';
    if (used.has(c.name)) cell.classList.add('disabled');
    cell.innerHTML = `
      <img src="${c.img}" onerror="this.style.display='none'">
      <div class="champ-name-overlay">${c.name}</div>`;
    cell.title = c.name;
    cell.addEventListener('click', () => onChampClick(c.name));
    grid.appendChild(cell);
  }
}

// ---------------------------------------------------------------------------
// Click handlers
// ---------------------------------------------------------------------------
function onSlotClick(side, role, i) {
  const arr = state[`${side}Picks`];
  if (arr[i]) {
    // Filled → clear it AND make this slot the active target.
    arr[i] = null;
    state.activeTarget = { type: 'pick', side, role };
  } else {
    // Empty → toggle target.
    const same = state.activeTarget?.type === 'pick'
              && state.activeTarget.side === side
              && state.activeTarget.role === role;
    state.activeTarget = same ? null : { type: 'pick', side, role };
  }
  hideResults();
  render();
  refreshWinProb();
}

function onChampClick(name) {
  if (currentUsedChampions().has(name)) return;
  if (!state.activeTarget) {
    // No target chosen — auto-pick the first empty blue slot.
    const i = state.bluePicks.findIndex(x => !x);
    if (i < 0) return;
    state.activeTarget = { type: 'pick', side: 'blue', role: ROLES[i] };
  }
  if (state.activeTarget.type === 'pick') {
    const arr = state[`${state.activeTarget.side}Picks`];
    const i   = ROLES.indexOf(state.activeTarget.role);
    arr[i] = name;
    // After filling, keep activeTarget in case the user wants to immediately
    // re-recommend a different fill (they can clear by clicking the slot).
  } else if (state.activeTarget.type === 'ban') {
    state.bans.push(name);
  }
  hideResults();
  render();
  refreshWinProb();
}

window.setRoleFilter = function (r) {
  state.roleFilter = r;
  document.querySelectorAll('.role-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.role === r);
  });
  renderGrid();
};

window.onSearchInput = function (v) {
  state.search = v;
  renderGrid();
};

function currentUsedChampions() {
  return new Set([...state.bluePicks, ...state.redPicks, ...state.bans].filter(Boolean));
}

// ---------------------------------------------------------------------------
// API calls
// ---------------------------------------------------------------------------
function buildPayload() {
  const blue_picks = {};
  const red_picks  = {};
  state.bluePicks.forEach((c, i) => { if (c) blue_picks[ROLES[i]] = c; });
  state.redPicks.forEach((c, i)  => { if (c) red_picks[ROLES[i]]  = c; });
  return { blue_picks, red_picks, bans: state.bans };
}

async function refreshWinProb() {
  const model = document.getElementById('pick-model')?.value
              || (state.available.includes('wide_deep') ? 'wide_deep' : state.available[0]);
  if (!model) return;
  try {
    const httpRes = await fetch('/api/evaluate', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ ...buildPayload(), model }),
    });
    if (!httpRes.ok || !(httpRes.headers.get('content-type') || '').includes('application/json')) {
      console.warn('refreshWinProb: non-JSON response', httpRes.status, await httpRes.text());
      return;
    }
    const res = await httpRes.json();
    const blue = res.blue_win_prob;
    if (blue != null) {
      document.getElementById('blue-wp').textContent = (blue * 100).toFixed(2) + '%';
      document.getElementById('red-wp').textContent  = ((1 - blue) * 100).toFixed(2) + '%';
    }
  } catch (e) { console.warn('refreshWinProb failed', e); }
}

async function runRecommend() {
  const t = state.activeTarget;
  if (!t || t.type !== 'pick') return;
  const arr = state[`${t.side}Picks`];
  if (arr[ROLES.indexOf(t.role)]) return;

  const btn = document.getElementById('run-btn');
  btn.disabled = true;
  btn.textContent = 'Computing…';

  const body = {
    ...buildPayload(),
    side:      t.side,
    role:      t.role,
    top_k:     parseInt(document.getElementById('pick-topk').value, 10),
    model:     document.getElementById('pick-model').value,
    algorithm: document.getElementById('pick-algo').value,
    beam_width:       5,
    beam_depth:       2,
    mcts_simulations: 64,
    include_breakdown: true,
  };

  let res;
  try {
    const httpRes = await fetch('/api/recommend', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(body),
    });
    const ct = httpRes.headers.get('content-type') || '';
    if (!httpRes.ok || !ct.includes('application/json')) {
      // Server returned HTML (Flask error page) or non-2xx. Surface the
      // real text so the user/dev can diagnose instead of the cryptic
      // "Unexpected token '<'" SyntaxError from .json().
      const txt = await httpRes.text();
      const snippet = txt.length > 800 ? txt.slice(0, 800) + ' …' : txt;
      throw new Error(`HTTP ${httpRes.status} ${httpRes.statusText}; body: ${snippet}`);
    }
    res = await httpRes.json();
  } catch (e) {
    alert('Request failed: ' + e.message);
    console.error('recommend failed', e);
    btn.disabled = false;
    btn.textContent = '🎯 Recommend';
    return;
  }
  if (res.error) {
    alert('Error: ' + res.error);
    btn.disabled = false;
    btn.textContent = '🎯 Recommend';
    return;
  }
  state.lastResponse = res;
  renderResults(res, t.side);
  btn.disabled = false;
  btn.textContent = '🎯 Recommend';
}

// ---------------------------------------------------------------------------
// Results rendering
// ---------------------------------------------------------------------------
function renderResults(res, side) {
  document.getElementById('results-card').classList.remove('hidden');
  const cur = side === 'blue' ? res.current_blue_winprob : (1 - res.current_blue_winprob);
  document.getElementById('cur-wp').textContent = (cur * 100).toFixed(2) + '%';

  const wps   = res.recommendations.map(r => r.win_prob);
  const range = wps.length ? Math.max(...wps) - Math.min(...wps) : 0;
  const old   = document.getElementById('flat-notice');
  if (old) old.remove();
  if (range < 0.0005 && wps.length > 1) {
    const div = document.createElement('div');
    div.id = 'flat-notice';
    div.className = 'flat-notice';
    div.textContent =
      `Note: model output is nearly constant across these candidates ` +
      `(spread ${(range * 100).toFixed(3)} pp). Use synergy / counter or the per-pair ` +
      `breakdown panel below to choose. See FINAL_REPORT.md §4.3 for the data-side cause.`;
    document.getElementById('results-card').insertBefore(div, document.getElementById('rec-grid'));
  }

  const grid = document.getElementById('rec-grid');
  grid.innerHTML = '';
  res.recommendations.forEach((r, i) => {
    const c       = state.byName[r.champion] || { img: '' };
    const wpPct   = (r.win_prob * 100).toFixed(2);
    const barPct  = Math.max(0, Math.min(100, ((r.win_prob - 0.40) / 0.20) * 100));
    const dCls    = r.delta   > 0 ? 'pos' : (r.delta   < 0 ? 'neg' : '');
    const synCls  = r.synergy > 0 ? 'pos' : (r.synergy < 0 ? 'neg' : '');
    const ctrCls  = r.counter > 0 ? 'pos' : (r.counter < 0 ? 'neg' : '');

    const card = document.createElement('div');
    card.className = 'rec-card-big';
    card.innerHTML = `
      <div class="rank">#${i + 1}</div>
      <div class="head">
        <div class="portrait"><img src="${c.img}" onerror="this.style.display='none'"></div>
        <div><div class="name">${r.champion}</div></div>
      </div>
      <div class="wp-bar"><div class="wp-fill" style="width:${barPct}%"></div></div>
      <div class="wp-bar-axis">40% &nbsp;&nbsp;&nbsp; <b>50%</b> &nbsp;&nbsp;&nbsp; 60%</div>
      <div class="metrics">
        <div class="metric"><span class="k">model wp</span><span class="v gold">${wpPct}%</span></div>
        <div class="metric"><span class="k">Δ</span><span class="v ${dCls}">${r.delta >= 0 ? '+' : ''}${r.delta.toFixed(4)}</span></div>
        <div class="metric"><span class="k">syn / ctr</span><span class="v">
          <span class="${synCls}">${r.synergy >= 0 ? '+' : ''}${r.synergy.toFixed(3)}</span> /
          <span class="${ctrCls}">${r.counter >= 0 ? '+' : ''}${r.counter.toFixed(3)}</span>
        </span></div>
      </div>
      <div class="notes">${r.notes || ''}</div>
      <div class="rec-card-cta">click to lock pick →</div>`;
    card.addEventListener('click', () => {
      // Single-click: select for detail view.
      selectRec(i);
    });
    card.addEventListener('dblclick', () => {
      // Double-click: lock this champion into the active slot.
      acceptRec(i);
    });
    grid.appendChild(card);
  });
  selectRec(0);
}

function acceptRec(idx) {
  const r = state.lastResponse?.recommendations?.[idx];
  if (!r) return;
  const t = state.activeTarget;
  if (!t || t.type !== 'pick') return;
  if (currentUsedChampions().has(r.champion)) return;
  const arr = state[`${t.side}Picks`];
  const i   = ROLES.indexOf(t.role);
  arr[i] = r.champion;
  hideResults();
  render();
  refreshWinProb();
}

function selectRec(idx) {
  state.selectedIndex = idx;
  document.querySelectorAll('.rec-card-big').forEach((el, i) => {
    el.classList.toggle('selected', i === idx);
  });
  const r = state.lastResponse?.recommendations?.[idx];
  if (!r) return;

  document.getElementById('detail-card').classList.remove('hidden');
  document.getElementById('detail-name').textContent = r.champion;
  document.getElementById('detail-sub').textContent =
    `algorithm: ${state.lastResponse.algorithm} · model: ${state.lastResponse.model}` +
    (r.mcts_visits ? ` · mcts visits: ${r.mcts_visits}` : '');
  document.getElementById('d-wp').textContent    = (r.win_prob * 100).toFixed(2) + '%';
  document.getElementById('d-delta').textContent = (r.delta >= 0 ? '+' : '') + r.delta.toFixed(4);
  document.getElementById('d-syn').textContent   = (r.synergy >= 0 ? '+' : '') + r.synergy.toFixed(3);
  document.getElementById('d-ctr').textContent   = (r.counter >= 0 ? '+' : '') + r.counter.toFixed(3);
  document.getElementById('d-notes').textContent = r.notes || '—';

  renderBars(document.getElementById('syn-bars'), r.ally_breakdown  || [], 'synergy');
  renderBars(document.getElementById('ctr-bars'), r.enemy_breakdown || [], 'counter');
}

function renderBars(root, rows, key) {
  root.innerHTML = '';
  if (!rows.length) {
    root.innerHTML = `<div class="bar-empty">No ${key === 'synergy' ? 'allies' : 'enemies'} set.</div>`;
    return;
  }
  const maxAbs = Math.max(0.05, ...rows.map(x => Math.abs(x[key] || 0)));
  for (const x of rows) {
    const v   = x[key] || 0;
    const pct = Math.min(50, (Math.abs(v) / maxAbs) * 50);
    const cls = v >= 0 ? 'pos' : 'neg';
    const c   = state.byName[x.champion];
    const row = document.createElement('div');
    row.className = 'bar-row';
    row.innerHTML = `
      <div class="bar-who">
        <div class="bar-portrait">${c ? `<img src="${c.img}" onerror="this.style.display='none'">` : ''}</div>
        <div>
          <div class="role">${x.role}</div>
          <div class="who">${x.champion}</div>
        </div>
      </div>
      <div class="gauge">
        <div class="gauge-fill ${cls}" style="${
          v >= 0 ? `left:50%; width:${pct}%` : `right:50%; width:${pct}%`
        }"></div>
      </div>
      <div class="val ${cls}">${v >= 0 ? '+' : ''}${v.toFixed(3)}</div>`;
    root.appendChild(row);
  }
}

function hideResults() {
  document.getElementById('results-card').classList.add('hidden');
  document.getElementById('detail-card').classList.add('hidden');
  state.lastResponse = null;
}

function updateActiveInfo() {
  const info = document.getElementById('active-info');
  const t    = state.activeTarget;
  info.classList.toggle('active', !!t);
  if (!t) {
    info.textContent = 'Click any empty role above to select your recommendation target.';
  } else if (t.type === 'pick') {
    info.innerHTML = `Selected: <strong>${t.side} ${t.role}</strong> — click 🎯 Recommend for top-K, or click a champion below to lock manually.`;
  } else {
    info.textContent = 'Ban mode — click any champion below to add to ban list.';
  }
  document.getElementById('grid-hint').textContent = t
    ? (t.type === 'pick' ? `Filling: ${t.side} ${t.role}` : 'Click champion to ban')
    : 'No active slot — click a slot above to start.';
}

function updateRunButton() {
  const btn = document.getElementById('run-btn');
  const t   = state.activeTarget;
  const can = t?.type === 'pick' && !state[`${t.side}Picks`][ROLES.indexOf(t.role)];
  btn.disabled = !can;
}

document.addEventListener('DOMContentLoaded', boot);

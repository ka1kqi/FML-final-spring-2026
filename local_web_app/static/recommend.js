/* ============================================================
   Recommendation Analyzer — form-driven UI.
   Populates dropdowns, posts to /api/recommend with
   include_breakdown=true, renders top-K big cards + detail panel.
   ============================================================ */

const ROLES = ['top', 'jungle', 'mid', 'adc', 'support'];
const ROLE_EMOJI = { top: '⚔️', jungle: '🌲', mid: '✨', adc: '🏹', support: '🛡️' };

const state = {
  champions: [],
  byName: {},
  available: [],
  lastResponse: null,
  selectedIndex: -1,
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
  state.byName = Object.fromEntries(champs.map(c => [c.name, c]));
  state.available = meta.available_models || [];
  hydrateMetaBar(meta);
  hydrateRoleRows();
  hydrateModelSelector();
  hydrateChampDatalist();

  document.getElementById('run-btn').addEventListener('click', runRecommend);
}

function hydrateMetaBar(meta) {
  const summary = meta.summary || {};
  const schema = summary.schema || {};
  const rec = summary.recommender || {};
  document.getElementById('m-auc').textContent     = summary.best_test_auc != null ? summary.best_test_auc.toFixed(4) : '—';
  document.getElementById('m-recall').textContent  = rec['recall@5'] != null ? (rec['recall@5'] * 100).toFixed(1) + '%' : '—';
  document.getElementById('m-matches').textContent = schema.matches != null ? schema.matches.toLocaleString() : '—';
  document.getElementById('m-split').textContent   = schema.split?.method ? schema.split.method.replace('_', '-') : '—';
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

function hydrateChampDatalist() {
  // Single shared datalist that all champion <input>s autocomplete from.
  let dl = document.getElementById('champ-datalist');
  if (!dl) {
    dl = document.createElement('datalist');
    dl.id = 'champ-datalist';
    document.body.appendChild(dl);
  }
  dl.innerHTML = '';
  for (const c of state.champions) {
    const opt = document.createElement('option');
    opt.value = c.name;
    dl.appendChild(opt);
  }
}

function hydrateRoleRows() {
  for (const side of ['blue', 'red']) {
    const root = document.getElementById(`${side}-role-rows`);
    root.innerHTML = '';
    for (const role of ROLES) {
      const row = document.createElement('div');
      row.className = 'role-row';
      row.innerHTML = `
        <label>${ROLE_EMOJI[role]} ${role}</label>
        <input list="champ-datalist" id="${side}-${role}"
               type="text" placeholder="(empty)">
      `;
      root.appendChild(row);
    }
  }
}

// ---------------------------------------------------------------------------
// Submit
// ---------------------------------------------------------------------------
function buildPayload() {
  const blue_picks = {};
  const red_picks = {};
  for (const role of ROLES) {
    const b = document.getElementById(`blue-${role}`).value.trim();
    const r = document.getElementById(`red-${role}`).value.trim();
    if (b) blue_picks[role] = b;
    if (r) red_picks[role] = r;
  }
  const bans = (document.getElementById('ban-input').value || '')
    .split(',').map(s => s.trim()).filter(Boolean);
  const side = document.getElementById('pick-side').value;
  const role = document.getElementById('pick-role').value;
  const top_k = parseInt(document.getElementById('pick-topk').value, 10);
  const model = document.getElementById('pick-model').value;
  const algorithm = document.getElementById('pick-algo').value;
  return {
    blue_picks, red_picks, bans,
    side, role, top_k, model, algorithm,
    beam_width: 5, beam_depth: 2,
    mcts_simulations: 64,
    include_breakdown: true,
  };
}

async function runRecommend() {
  const btn = document.getElementById('run-btn');
  btn.disabled = true;
  btn.textContent = 'Computing…';

  const payload = buildPayload();
  if (payload.blue_picks[payload.role] && payload.side === 'blue') {
    alert(`Blue ${payload.role} is already filled with ${payload.blue_picks[payload.role]}. Clear it before recommending.`);
    btn.disabled = false; btn.textContent = 'Recommend'; return;
  }
  if (payload.red_picks[payload.role] && payload.side === 'red') {
    alert(`Red ${payload.role} is already filled with ${payload.red_picks[payload.role]}. Clear it before recommending.`);
    btn.disabled = false; btn.textContent = 'Recommend'; return;
  }

  let res;
  try {
    res = await fetch('/api/recommend', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }).then(r => r.json());
  } catch (e) {
    alert('Request failed: ' + e);
    btn.disabled = false; btn.textContent = 'Recommend'; return;
  }
  if (res.error) {
    alert('Error: ' + res.error);
    btn.disabled = false; btn.textContent = 'Recommend'; return;
  }
  state.lastResponse = res;
  renderResults(res, payload.side);
  btn.disabled = false;
  btn.textContent = 'Recommend';
}

// ---------------------------------------------------------------------------
// Render results
// ---------------------------------------------------------------------------
function renderResults(res, side) {
  const card = document.getElementById('results-card');
  card.classList.remove('hidden');
  const cur = side === 'blue' ? res.current_blue_winprob : (1 - res.current_blue_winprob);
  document.getElementById('cur-wp').textContent = (cur * 100).toFixed(2) + '%';

  // Detect a degenerate case where every recommendation has the same wp:
  // surface a clear note rather than letting the user think the page is broken.
  const wps = res.recommendations.map(r => r.win_prob);
  const range = wps.length ? Math.max(...wps) - Math.min(...wps) : 0;
  const notice = document.getElementById('flat-notice');
  if (notice) notice.remove();
  if (range < 0.0005 && wps.length > 1) {
    const div = document.createElement('div');
    div.id = 'flat-notice';
    div.className = 'flat-notice';
    div.textContent =
      'Note: the model returns a near-constant win probability across these candidates ' +
      `(spread ${(range * 100).toFixed(3)} pp). Use synergy / counter columns and the per-pair ` +
      'breakdown panel below to pick. This is a known limitation of training on 7.9k matches with ' +
      'no rank/mastery features — see FINAL_REPORT.md §4.3.';
    document.getElementById('results-card').insertBefore(div, document.getElementById('rec-grid'));
  }

  const grid = document.getElementById('rec-grid');
  grid.innerHTML = '';
  res.recommendations.forEach((r, i) => {
    const c = state.byName[r.champion] || { img: '' };
    // Display 2 decimal places of % so 51.32 vs 51.01 is visible.
    const wpPct = (r.win_prob * 100).toFixed(2);
    // Zoom the bar to [40%, 60%] so a 0.5 pt difference = 2.5% bar width.
    const barPct = Math.max(0, Math.min(100, ((r.win_prob - 0.40) / 0.20) * 100));
    const deltaCls = r.delta > 0 ? 'pos' : (r.delta < 0 ? 'neg' : '');
    const synCls = r.synergy > 0 ? 'pos' : (r.synergy < 0 ? 'neg' : '');
    const ctrCls = r.counter > 0 ? 'pos' : (r.counter < 0 ? 'neg' : '');
    const card = document.createElement('div');
    card.className = 'rec-card-big';
    card.innerHTML = `
      <div class="rank">#${i + 1}</div>
      <div class="head">
        <div class="portrait"><img src="${c.img}" onerror="this.style.display='none'"></div>
        <div>
          <div class="name">${r.champion}</div>
        </div>
      </div>
      <div class="wp-bar"><div class="wp-fill" style="width:${barPct}%"></div></div>
      <div class="wp-bar-axis">40% &nbsp;&nbsp;&nbsp; <b>50%</b> &nbsp;&nbsp;&nbsp; 60%</div>
      <div class="metrics">
        <div class="metric"><span class="k">model wp</span><span class="v gold">${wpPct}%</span></div>
        <div class="metric"><span class="k">Δ</span><span class="v ${deltaCls}">${r.delta >= 0 ? '+' : ''}${r.delta.toFixed(4)}</span></div>
        <div class="metric"><span class="k">syn / ctr</span><span class="v"><span class="${synCls}">${r.synergy >= 0 ? '+' : ''}${r.synergy.toFixed(3)}</span> / <span class="${ctrCls}">${r.counter >= 0 ? '+' : ''}${r.counter.toFixed(3)}</span></span></div>
      </div>
      <div class="notes">${r.notes || ''}</div>`;
    card.addEventListener('click', () => selectRec(i));
    grid.appendChild(card);
  });
  selectRec(0);
}

function selectRec(idx) {
  state.selectedIndex = idx;
  const all = document.querySelectorAll('.rec-card-big');
  all.forEach((el, i) => el.classList.toggle('selected', i === idx));
  const r = state.lastResponse?.recommendations?.[idx];
  if (!r) return;

  const card = document.getElementById('detail-card');
  card.classList.remove('hidden');
  document.getElementById('detail-name').textContent = r.champion;
  document.getElementById('detail-sub').textContent =
    `${r.notes && r.notes.includes('blind') ? 'no enemy info — ' : ''}` +
    `algorithm: ${state.lastResponse.algorithm}, model: ${state.lastResponse.model}` +
    (r.mcts_visits ? `, mcts visits: ${r.mcts_visits}` : '');
  document.getElementById('d-wp').textContent    = (r.win_prob * 100).toFixed(2) + '%';
  document.getElementById('d-delta').textContent = (r.delta >= 0 ? '+' : '') + r.delta.toFixed(3);
  document.getElementById('d-syn').textContent   = (r.synergy >= 0 ? '+' : '') + r.synergy.toFixed(3);
  document.getElementById('d-ctr').textContent   = (r.counter >= 0 ? '+' : '') + r.counter.toFixed(3);
  document.getElementById('d-notes').textContent = r.notes || '—';

  renderBars(document.getElementById('syn-bars'), r.ally_breakdown || [], 'synergy');
  renderBars(document.getElementById('ctr-bars'), r.enemy_breakdown || [], 'counter');
}

function renderBars(root, rows, key) {
  root.innerHTML = '';
  if (!rows.length) {
    root.innerHTML = `<div class="bar-empty">No ${key === 'synergy' ? 'allies' : 'enemies'} set in form.</div>`;
    return;
  }
  // Use a fixed scale so different rows are comparable
  const maxAbs = Math.max(0.05, ...rows.map(x => Math.abs(x[key] || 0)));
  for (const x of rows) {
    const v = x[key] || 0;
    const pct = Math.min(50, (Math.abs(v) / maxAbs) * 50);
    const cls = v >= 0 ? 'pos' : 'neg';
    const row = document.createElement('div');
    row.className = 'bar-row';
    row.innerHTML = `
      <div>
        <div class="role">${x.role}</div>
        <div class="who">${x.champion}</div>
      </div>
      <div class="gauge">
        <div class="gauge-fill ${cls}" style="${v >= 0 ? `left:50%; width:${pct}%` : `right:50%; width:${pct}%`}"></div>
      </div>
      <div class="val ${cls}">${v >= 0 ? '+' : ''}${v.toFixed(3)}</div>`;
    root.appendChild(row);
  }
}

document.addEventListener('DOMContentLoaded', boot);

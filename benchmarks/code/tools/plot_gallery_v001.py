"""Write an exclusive, dependency-free HTML gallery for frozen N1–N9 figures.

The central plot generator owns every value, image, data file and provenance
artifact. This module only presents its manifest; it performs no measurements,
statistical calculations, network requests or figure selection from results.
"""
from __future__ import annotations

import json
from pathlib import Path


def write_gallery(output_dir: Path, manifest: dict) -> Path:
    """Create ``index.html`` once, beside the generator's figure/data artifacts.

    ``manifest`` has charts, sources and coverage fields. Chart entries contain
    id, experiment, title, subtitle, takeaway, scope, cohort, detail, files,
    data_file and source_run_ids. Relative files are optional. Never overwrite
    a previously published gallery; the caller must use a new output directory.
    """
    if not isinstance(manifest, dict) or not isinstance(manifest.get("charts"), list):
        raise ValueError("Gallery manifest must contain a charts list")
    for key in ("sources", "coverage"):
        if not isinstance(manifest.get(key), dict):
            raise ValueError(f"Gallery manifest must contain a {key} object")
    payload = json.dumps(manifest, ensure_ascii=False, allow_nan=False)
    # Keep arbitrary manifest text inside its inert JSON script element.
    payload = payload.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "index.html"
    with destination.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(_PAGE.replace("__MANIFEST_JSON__", payload))
    return destination


_PAGE = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light">
  <title>Strassen MM Focus · N1–N9 results</title>
  <style>
    :root { --ink:#172433; --muted:#526170; --line:#d9e1e8; --paper:#fff;
      --wash:#f4f7fa; --blue:#2463a6; --orange:#c66b17; --green:#23804c; --accent:#204c77; }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--wash); color:var(--ink); font:16px/1.55 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
    a { color:var(--accent); text-underline-offset:.2em; }
    a:hover { color:#102f50; }
    button, select, input { font:inherit; }
    button, select { border:1px solid var(--line); border-radius:8px; background:var(--paper); color:var(--ink); }
    button { padding:.45rem .8rem; cursor:pointer; }
    :focus-visible { outline:3px solid #6598d3; outline-offset:3px; }
    .skip { position:absolute; left:1rem; top:-5rem; padding:.7rem; z-index:10; background:var(--paper); }
    .skip:focus { top:.5rem; }
    .wrap { max-width:1460px; margin:auto; padding:0 clamp(18px,4vw,56px); }
    header { background:var(--paper); border-bottom:1px solid var(--line); padding:40px 0 26px; }
    .eyebrow { margin:0 0 8px; font-size:.76rem; text-transform:uppercase; letter-spacing:.14em; color:var(--muted); font-weight:750; }
    h1 { margin:0; font-size:clamp(1.8rem,3.5vw,2.75rem); line-height:1.15; letter-spacing:-.04em; }
    .lede { margin:13px 0 19px; max-width:850px; color:var(--muted); }
    .resources { display:flex; flex-wrap:wrap; gap:10px 22px; font-size:.9rem; }
    .resources a { font-weight:650; }
    .guide { margin:22px 0 0; padding:16px 18px; border:1px solid var(--line); border-radius:12px; background:#fafcfe; }
    .legend { display:flex; flex-wrap:wrap; gap:9px 21px; margin-bottom:8px; font-size:.9rem; }
    .legend span { display:inline-flex; align-items:center; gap:7px; }
    .dot { width:11px; height:11px; border-radius:50%; display:inline-block; }
    .native { background:var(--blue); } .cubic { background:var(--orange); } .strassen { background:var(--green); }
    .guide p { margin:5px 0 0; color:var(--muted); font-size:.88rem; max-width:1160px; }
    .toolbar { background:var(--paper); border-bottom:1px solid var(--line); }
    nav { display:flex; gap:8px; overflow-x:auto; padding:17px 0 13px; }
    nav a { display:inline-flex; justify-content:center; min-width:54px; padding:8px 14px; border:1px solid var(--line); border-radius:9px; font-weight:750; text-decoration:none; }
    nav a[aria-current="page"] { color:white; background:var(--accent); border-color:var(--accent); }
    .filters { display:flex; align-items:flex-end; flex-wrap:wrap; gap:15px 27px; padding:0 0 18px; }
    fieldset { border:0; padding:0; margin:0; min-width:0; }
    legend, .select-label { display:block; margin-bottom:5px; font-size:.77rem; color:var(--muted); font-weight:750; }
    .options { display:flex; flex-wrap:wrap; gap:5px; }
    .options label { cursor:pointer; position:relative; }
    .options input { position:absolute; width:1px; height:1px; opacity:0; }
    .options span { display:block; padding:6px 11px; border:1px solid var(--line); border-radius:7px; font-size:.88rem; }
    .options input:checked + span { background:#eaf1f8; color:#183e64; border-color:#93afcb; font-weight:700; }
    .options input:focus-visible + span { outline:3px solid #6598d3; outline-offset:3px; }
    select { min-height:36px; max-width:340px; padding:5px 30px 5px 10px; font-size:.88rem; }
    main { padding-top:28px; padding-bottom:40px; }
    .section-heading { display:flex; flex-wrap:wrap; align-items:baseline; gap:8px 20px; margin-bottom:19px; }
    h2 { margin:0; font-size:1.35rem; letter-spacing:-.02em; }
    #count { margin:0; color:var(--muted); font-size:.9rem; }
    #cards { display:grid; gap:25px; }
    .card { border:1px solid var(--line); border-radius:14px; background:var(--paper); overflow:hidden; box-shadow:0 3px 12px #17243306; }
    .card-head { padding:21px 24px 12px; }
    .card h3 { margin:7px 0 5px; font-size:1.12rem; line-height:1.35; }
    .subtitle { margin:0; color:var(--muted); font-size:.9rem; }
    .tags { display:flex; flex-wrap:wrap; gap:6px; }
    .tag { border:1px solid #e0e7ed; background:#f5f8fb; border-radius:5px; padding:2px 7px; color:#465b6d; font-size:.72rem; overflow-wrap:anywhere; }
    .view-note { margin:9px 0 0; font-size:.8rem; color:#765616; }
    .figure-link { display:block; padding:0 10px; }
    .figure-link img { width:100%; height:auto; display:block; background:white; }
    .takeaway { margin:12px 24px 16px; padding:11px 13px; border-left:3px solid #9ab7d3; background:#f5f8fc; font-size:.94rem; }
    .card-foot { display:flex; flex-wrap:wrap; align-items:flex-start; gap:12px 24px; border-top:1px solid #e7edf2; padding:13px 24px 15px; }
    .downloads { display:flex; gap:16px; flex-wrap:wrap; font-size:.84rem; font-weight:650; }
    details { font-size:.83rem; color:var(--muted); }
    summary { cursor:pointer; }
    .evidence { flex:1; min-width:180px; }
    .evidence ul { margin:7px 0 0; padding-left:18px; overflow-wrap:anywhere; font: .76rem/1.5 ui-monospace,SFMono-Regular,Consolas,monospace; }
    .empty { border:1px dashed #b7c8d7; padding:30px; border-radius:12px; background:var(--paper); }
    .empty p { margin-top:0; }
    .image-error { padding:22px; color:var(--muted); }
    footer { border-top:1px solid var(--line); padding:23px 0 36px; color:var(--muted); font-size:.82rem; }
    footer p { margin:0 0 9px; }
    #coverage-text { white-space:pre-wrap; overflow-wrap:anywhere; max-height:400px; overflow:auto; border:1px solid var(--line); padding:14px; background:white; border-radius:8px; font-size:.75rem; }
    noscript { display:block; padding:20px; background:#fff1d0; }
    @media (max-width:600px) {
      header { padding-top:27px; } .card-head { padding:17px 16px 10px; }
      .card-foot { padding:12px 16px; } .takeaway { margin:10px 16px 14px; }
      .figure-link { padding:0; } .filters { gap:13px 18px; } select { max-width:280px; }
      nav { gap:5px; } nav a { min-width:48px; padding:7px 10px; }
    }
    @media print {
      .toolbar, .downloads, .resources, .skip, footer { display:none; }
      body, header { background:white; } .wrap { max-width:none; padding:0; }
      header { padding:0 0 16px; } .card { break-inside:avoid; box-shadow:none; }
    }
  </style>
</head>
<body>
  <a class="skip" href="#results">Skip to figures</a>
  <header><div class="wrap">
    <p class="eyebrow">Strassen MM Focus · frozen experiment evidence</p>
    <h1>Matrix multiplication, then applications</h1>
    <p class="lede">Explore the N1–N9 comparisons, from basic kernels to real-model checks. Every figure links to its data; timing scopes and runtime cohorts remain separate.</p>
    <div class="resources">
      <a href="N1_N9_key_comparisons.pdf">Key comparisons · PDF</a>
      <a href="sources.json">Source provenance · JSON</a>
      <a href="coverage.json">Coverage · JSON</a>
    </div>
    <div class="guide" aria-label="How to read these figures">
      <div class="legend">
        <span><i class="dot native" aria-hidden="true"></i>Native · blue</span>
        <span><i class="dot cubic" aria-hidden="true"></i>Cubic · orange</span>
        <span><i class="dot strassen" aria-hidden="true"></i>Strassen · green</span>
        <span>Extra controls use distinct labels and styles.</span>
      </div>
      <p>Lower latency or error is better. For speedup ratios, higher is better and 1× is the reference. Check each axis and caption: error bars show the uncertainty or spread stated in that figure.</p>
      <p>Complete-call and prepared measurements are separate. OOM and unmeasured timings are missing values, never zero. Numerical failures remain visible as hatched error bars. N4 supports accuracy comparisons only. Gemma was not measured in N9 because access was blocked.</p>
    </div>
  </div></header>
  <div class="toolbar"><div class="wrap">
    <nav id="experiment-nav" aria-label="Experiment"></nav>
    <div class="filters" aria-label="Figure filters">
      <fieldset><legend>Figure coverage</legend><div class="options">
        <label><input type="radio" name="detail" value="overview" checked><span>Overview</span></label>
        <label><input type="radio" name="detail" value="all"><span>All details</span></label>
      </div></fieldset>
      <div><label class="select-label" for="scope">Timing / measurement scope</label><select id="scope"><option value="all">All scopes</option></select></div>
      <fieldset><legend>Figure view</legend><div class="options">
        <label><input type="radio" name="view" value="absolute" checked><span>Absolute</span></label>
        <label><input type="radio" name="view" value="relative"><span>Relative</span></label>
      </div></fieldset>
    </div>
  </div></div>
  <main id="results" class="wrap" tabindex="-1">
    <div class="section-heading"><h2 id="section-title">N1 figures</h2><p id="count" role="status" aria-live="polite"></p></div>
    <div id="cards"></div>
    <noscript>Enable JavaScript to filter this local gallery, or open the PDF, source provenance and coverage links above. No external services are required.</noscript>
  </main>
  <footer><div class="wrap">
    <p id="totals"></p>
    <p>Figures present the supplied measurements. A numerical pass or a completed run does not by itself establish a performance gain. Selection results and independent confirmation must be read according to their captions.</p>
    <details><summary>Recorded coverage</summary><pre id="coverage-text"></pre></details>
  </div></footer>
  <script id="gallery-manifest" type="application/json">__MANIFEST_JSON__</script>
  <script>
  (() => {
    'use strict';
    const manifest = JSON.parse(document.getElementById('gallery-manifest').textContent);
    const charts = manifest.charts;
    const experiments = Array.from({length:9}, (_, i) => 'N' + (i + 1));
    const state = {experiment:'N1', detail:'overview', scope:'all', view:'absolute'};
    const nav = document.getElementById('experiment-nav');
    const cards = document.getElementById('cards');
    const scopeSelect = document.getElementById('scope');
    const scopeName = value => ({call:'Complete call', prepared_kernel:'Prepared kernel',
      resident:'Resident layer', streamed:'Streamed full forward', device:'Profiled TPU module',
      accuracy:'Accuracy', quality:'Model quality', preparation:'Preparation'})[String(value)] || String(value || 'Unspecified');
    const el = (tag, className, text) => {
      const node = document.createElement(tag);
      if (className) node.className = className;
      if (text !== undefined && text !== null) node.textContent = String(text);
      return node;
    };
    const safePath = value => {
      if (typeof value !== 'string' || !value || /^[a-z][a-z0-9+.-]*:/i.test(value)
          || /^[\\/]/.test(value) || value.split(/[\\/]/).includes('..')) return null;
      return value;
    };
    const fileLink = (parent, label, path) => {
      const safe = safePath(path);
      if (!safe) return;
      const link = el('a', '', label); link.href = safe;
      link.download = safe.split('/').pop(); parent.appendChild(link);
    };
    const experimentCharts = () => charts.filter(c => c.experiment === state.experiment);
    const selectScopeOptions = () => {
      const values = [...new Set(experimentCharts().map(c => String(c.scope || 'Unspecified')))];
      if (state.scope !== 'all' && !values.includes(state.scope)) state.scope = 'all';
      scopeSelect.replaceChildren();
      const all = el('option', '', 'All scopes'); all.value = 'all'; scopeSelect.appendChild(all);
      values.forEach(value => { const option = el('option', '', scopeName(value)); option.value = value; scopeSelect.appendChild(option); });
      scopeSelect.value = state.scope;
    };
    const renderCard = chart => {
      const card = el('article', 'card');
      const head = el('div', 'card-head');
      const tags = el('div', 'tags');
      tags.appendChild(el('span', 'tag', chart.experiment));
      tags.appendChild(el('span', 'tag', 'Scope: ' + scopeName(chart.scope)));
      tags.appendChild(el('span', 'tag', 'Cohort: ' + String(chart.cohort || 'Not specified')));
      if (chart.detail) tags.appendChild(el('span', 'tag', 'Detail'));
      head.appendChild(tags);
      head.appendChild(el('h3', '', chart.title || chart.id));
      if (chart.subtitle) head.appendChild(el('p', 'subtitle', chart.subtitle));
      const variants = chart.files || {};
      const hasFigure = v => v && (safePath(v.png) || safePath(v.svg));
      let mode = state.view;
      if (!hasFigure(variants[mode])) mode = hasFigure(variants.absolute) ? 'absolute' : 'relative';
      const files = variants[mode] || {};
      if (mode !== state.view && hasFigure(files)) {
        head.appendChild(el('p', 'view-note', state.view === 'relative'
          ? 'No relative variant supplied; showing the absolute / primary figure.'
          : 'No absolute variant supplied; showing the relative figure.'));
      }
      card.appendChild(head);
      const imagePath = safePath(files.png) || safePath(files.svg);
      if (imagePath) {
        const imageLink = el('a', 'figure-link'); imageLink.href = imagePath;
        imageLink.target = '_blank'; imageLink.rel = 'noopener';
        imageLink.setAttribute('aria-label', 'Open full-size figure: ' + String(chart.title || chart.id));
        const image = document.createElement('img'); image.src = imagePath;
        image.alt = String(chart.title || chart.id) + ' — ' + scopeName(chart.scope) + ', ' + mode + ' view';
        image.loading = 'lazy'; image.decoding = 'async';
        image.addEventListener('error', () => {
          imageLink.replaceWith(el('p', 'image-error', 'Figure could not be loaded. Check the PNG/SVG links and source data below.'));
        }, {once:true});
        imageLink.appendChild(image); card.appendChild(imageLink);
      } else card.appendChild(el('p', 'image-error', 'No figure file supplied for this comparison.'));
      if (chart.takeaway) card.appendChild(el('p', 'takeaway', chart.takeaway));
      const foot = el('div', 'card-foot');
      const downloads = el('div', 'downloads'); downloads.setAttribute('aria-label', 'Figure downloads');
      fileLink(downloads, 'PNG', files.png); fileLink(downloads, 'SVG', files.svg);
      fileLink(downloads, 'Figure data · JSON', chart.data_file); foot.appendChild(downloads);
      const evidence = el('details', 'evidence');
      evidence.appendChild(el('summary', '', 'Source executions · ' + String(chart.id || 'figure')));
      const list = el('ul');
      (Array.isArray(chart.source_run_ids) ? chart.source_run_ids : []).forEach(id => list.appendChild(el('li', '', id)));
      if (!list.children.length) list.appendChild(el('li', '', 'No source run IDs supplied; see sources.json.'));
      evidence.appendChild(list); foot.appendChild(evidence); card.appendChild(foot);
      return card;
    };
    const render = () => {
      nav.querySelectorAll('a').forEach(a => {
        if (a.dataset.experiment === state.experiment) a.setAttribute('aria-current', 'page');
        else a.removeAttribute('aria-current');
      });
      const all = experimentCharts();
      const shown = all.filter(c => (state.detail === 'all' || !c.detail)
        && (state.scope === 'all' || String(c.scope || 'Unspecified') === state.scope));
      document.getElementById('section-title').textContent = state.experiment + ' figures';
      document.getElementById('count').textContent = shown.length + ' of ' + all.length + ' figures · '
        + (state.detail === 'all' ? 'all details' : 'overview') + ' · '
        + (state.scope === 'all' ? 'all scopes' : scopeName(state.scope));
      cards.replaceChildren();
      if (!shown.length) {
        const empty = el('div', 'empty');
        empty.appendChild(el('p', '', all.length ? 'No figures match these filters.' : 'No figures were supplied for this experiment.'));
        if (all.length) {
          const reset = el('button', '', 'Show all details and scopes');
          reset.type = 'button'; reset.addEventListener('click', () => {
            state.detail = 'all'; state.scope = 'all';
            document.querySelector('input[name="detail"][value="all"]').checked = true;
            scopeSelect.value = 'all'; render();
          }); empty.appendChild(reset);
        }
        cards.appendChild(empty);
      } else shown.forEach(c => cards.appendChild(renderCard(c)));
    };
    const fromHash = () => {
      const candidate = location.hash.slice(1);
      // The skip-link anchor is not an experiment change.
      if (candidate === 'results' && cards.childElementCount) return;
      state.experiment = experiments.includes(candidate) ? candidate : 'N1';
      selectScopeOptions(); render();
    };
    experiments.forEach(name => {
      const link = el('a', '', name); link.href = '#' + name; link.dataset.experiment = name;
      link.setAttribute('aria-label', 'Experiment ' + name); nav.appendChild(link);
      link.addEventListener('click', () => {
        // Re-selecting the same hash must still leave the results visible.
        if (location.hash === '#' + name) { state.experiment = name; selectScopeOptions(); render(); }
      });
    });
    document.querySelectorAll('input[name="detail"]').forEach(input => input.addEventListener('change', () => { state.detail = input.value; render(); }));
    document.querySelectorAll('input[name="view"]').forEach(input => input.addEventListener('change', () => { state.view = input.value; render(); }));
    scopeSelect.addEventListener('change', () => { state.scope = scopeSelect.value; render(); });
    window.addEventListener('hashchange', fromHash);
    const runIDs = new Set(charts.flatMap(c => Array.isArray(c.source_run_ids) ? c.source_run_ids : []));
    document.getElementById('totals').textContent = charts.length + ' supplied figures · ' + runIDs.size + ' referenced source executions · local, self-contained gallery';
    document.getElementById('coverage-text').textContent = JSON.stringify(manifest.coverage, null, 2);
    fromHash();
  })();
  </script>
</body>
</html>
'''

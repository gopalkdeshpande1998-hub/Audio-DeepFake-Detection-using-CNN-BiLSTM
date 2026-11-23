"""Spectrogram Lab: an interactive waveform + spectrogram workstation with a custom audio player.

One self-contained HTML component (iframe) holding one lane (Detect) or two lanes (Compare). Each lane has its own
audio element; the toolbar zooms, pans, switches views and frequency ranges, sets volume and speed; dragging selects
a region (played, zoomed to, mirrored on every lane), hovering reads out time, frequency and level. Every number comes
from ui.signal (the decoded audio) and, when given, the detector's own per-step scores and flagged segments.
"""

from __future__ import annotations

import json

import numpy as np
import streamlit.components.v1 as components

from ui import signal
from ui.theme import AMBER, CYAN, GREEN, RED

TONES = {"main": CYAN, "A": CYAN, "B": AMBER}


def lane(x: np.ndarray, sr: int, title: str, tone: str = "main", res: dict | None = None) -> dict:
    d = {"title": title, "tone": TONES.get(tone, CYAN), "tag": tone, "dur": len(x) / sr,
         "audio": signal.audio_uri(x, sr), "peaks": signal.peaks(x), "spec": signal.spectrogram(x, sr)}
    if res is not None and res.get("per_second"):
        d["steps"] = [[w["start_s"], w["end_s"], w["p_fake"]] for w in res["per_second"]]
        d["segs"] = [[s["start_s"], s["end_s"], f"S{i}", s["p_fake"]] for i, s in enumerate(res.get("segments", []), 1)]
    return d


def render(lanes: list[dict], key: str, view_h: int | None = None) -> None:
    """Draw the lab. Height is fixed per lane count so the page never jumps when views change."""
    n = len(lanes)
    view_h = view_h or (330 if n == 1 else 220)
    has_strip = any("steps" in ln for ln in lanes)
    height = 64 + n * (view_h + 58 + (12 if has_strip else 0)) + 72
    cfg = {"lanes": lanes, "viewH": view_h, "heat": signal.HEAT_STOPS, "key": key,
           "colors": {"real": GREEN, "syn": RED, "amber": AMBER, "cyan": CYAN}}
    payload = json.dumps(cfg).replace("</", "<\\/")
    components.html(TEMPLATE.replace("__CFG__", payload), height=height, scrolling=False)


TEMPLATE = r"""<!doctype html><html><head><meta charset="utf-8">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{--bg:#0d1524;--panel:#111827;--raise:#152033;--line:#1f2a3c;--line2:#2b3a52;--text:#e6ebf2;--mute:#8b98ad;--dim:#5d6b82;
--cyan:#22d3ee;--real:#34d399;--syn:#f87171;--amber:#fbbf24}
*{box-sizing:border-box}
html,body{margin:0;background:transparent;color:var(--text);font:13px/1.4 Inter,system-ui,sans-serif;overflow:hidden}
.ws{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:10px 12px 12px;height:100vh;display:flex;flex-direction:column}
.tb{display:flex;align-items:center;gap:6px;flex-wrap:wrap;padding-bottom:10px;border-bottom:1px solid var(--line)}
.grp{display:flex;align-items:center;gap:2px;background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:2px}
button,select{font:inherit;color:var(--text);background:transparent;border:0;border-radius:6px;height:30px;min-width:30px;
  padding:0 8px;display:inline-flex;align-items:center;justify-content:center;gap:6px;cursor:pointer;
  transition:background .15s ease,color .15s ease,transform .1s ease}
button:hover,select:hover{background:var(--raise)}
button:active{transform:scale(.95)}
button:focus-visible,select:focus-visible,input:focus-visible{outline:2px solid var(--cyan);outline-offset:1px}
button.on{background:rgba(34,211,238,.14);color:var(--cyan)}
button.play{background:var(--cyan);color:#05202a;min-width:36px;height:32px;border-radius:8px}
button.play:hover{background:#67e8f9}
button:disabled{opacity:.35;cursor:default}
select{appearance:none;padding:0 22px 0 8px;background:var(--bg) url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='10' viewBox='0 0 24 24' fill='none' stroke='%238b98ad' stroke-width='3'%3E%3Cpath d='m6 9 6 6 6-6'/%3E%3C/svg%3E") no-repeat right 7px center;border:1px solid var(--line)}
option{background:#0d1524}
.time{font:500 12px "JetBrains Mono",ui-monospace,monospace;color:var(--mute);padding:0 6px;white-space:nowrap}
.time b{color:var(--text);font-weight:500}
.sp{flex:1}
input[type=range]{width:78px;accent-color:var(--cyan)}
.lbl{font-size:11px;color:var(--dim);padding:0 2px 0 6px}
svg{width:15px;height:15px;stroke:currentColor;fill:none;stroke-width:2;stroke-linecap:round;stroke-linejoin:round}
.lanes{flex:1;display:flex;flex-direction:column;gap:6px;padding-top:8px;min-height:0}
.lane{position:relative}
.lh{display:flex;align-items:center;gap:8px;height:24px;font-size:12px;color:var(--mute)}
.tag{font-weight:700;font-size:10px;letter-spacing:.08em;color:#05202a;padding:2px 6px;border-radius:4px}
.lt{color:var(--text);font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:46%}
.legend{margin-left:auto;display:flex;align-items:center;gap:6px;font-size:10.5px;color:var(--dim)}
.legend canvas{width:90px;height:8px;border-radius:2px}
.view{position:relative;cursor:crosshair;touch-action:none;border-radius:6px;overflow:hidden;background:#070b14}
.view canvas{display:block;width:100%}
.axis{height:22px}
.tip{position:absolute;pointer-events:none;background:rgba(8,12,22,.94);border:1px solid var(--line2);border-radius:6px;
  padding:6px 9px;font:500 11.5px "JetBrains Mono",ui-monospace,monospace;color:var(--text);white-space:nowrap;
  display:none;z-index:4;box-shadow:0 8px 24px rgba(0,0,0,.5)}
.tip i{font-style:normal;color:var(--mute)}
.lp{height:24px;min-width:24px;padding:0 6px}
.foot{display:flex;align-items:center;gap:10px;flex-wrap:wrap;padding-top:8px;border-top:1px solid var(--line);margin-top:6px}
.mini{flex:1;min-width:180px;height:30px;position:relative;cursor:pointer;border-radius:5px;overflow:hidden;background:#070b14;touch-action:none}
.mini canvas{width:100%;height:30px;display:block}
.sel{font-size:12px;color:var(--mute);display:flex;align-items:center;gap:4px;flex-wrap:wrap}
.sel b{color:var(--cyan);font:500 12px "JetBrains Mono",monospace}
.hint{font-size:11px;color:var(--dim)}
@media (max-width:620px){.hide-sm{display:none!important}input[type=range]{width:60px}.lt{max-width:60%}}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
</style></head><body>
<div class="ws" id="ws">
 <div class="tb" role="toolbar" aria-label="Audio lab controls">
  <button class="play" id="play" title="Play / pause (Space)" aria-label="Play"></button>
  <span class="time" id="time"><b>0:00.0</b> / 0:00.0</span>
  <div class="grp" role="group" aria-label="Zoom">
   <button id="zin" title="Zoom in (+)" aria-label="Zoom in"></button>
   <button id="zout" title="Zoom out (−)" aria-label="Zoom out"></button>
   <button id="zreset" title="Reset zoom (0)" aria-label="Reset zoom"></button>
  </div>
  <div class="grp" role="group" aria-label="Time navigation">
   <button id="pl" title="Pan left (←)" aria-label="Pan left"></button>
   <button id="pr" title="Pan right (→)" aria-label="Pan right"></button>
  </div>
  <div class="grp" role="group" aria-label="View">
   <button id="vw" title="Toggle waveform" aria-pressed="true">Wave</button>
   <button id="vs" title="Toggle spectrogram" aria-pressed="true">Spectrum</button>
  </div>
  <select id="fr" title="Frequency range" aria-label="Frequency range">
   <option value="0,8000">0–8 kHz</option><option value="0,4000">0–4 kHz speech</option>
   <option value="4000,8000">4–8 kHz high band</option><option value="0,1000">0–1 kHz pitch</option>
  </select>
  <span class="sp"></span>
  <span id="linkWrap" class="hide-sm" style="display:none"><button id="link" class="on" title="Play both recordings together" aria-pressed="true">Linked</button></span>
  <span class="lbl hide-sm">Vol</span><input id="vol" type="range" min="0" max="1" step="0.05" value="0.9" aria-label="Volume">
  <select id="spd" title="Playback speed" aria-label="Playback speed">
   <option>0.5</option><option>0.75</option><option selected>1</option><option>1.25</option><option>1.5</option><option>2</option>
  </select>
 </div>
 <div class="lanes" id="lanes"></div>
 <div class="foot">
  <div class="mini" id="mini" title="Overview: click or drag to move the view"><canvas id="minic"></canvas></div>
  <div class="sel" id="sel"><span class="hint">Drag across the view to select a region · click to seek · Ctrl + wheel to zoom</span></div>
 </div>
</div>
<script id="cfg" type="application/json">__CFG__</script>
<script>
(function(){
const C = JSON.parse(document.getElementById('cfg').textContent);
const IC = {
 play:'<svg viewBox="0 0 24 24"><polygon points="6 3 20 12 6 21 6 3" fill="currentColor"/></svg>',
 pause:'<svg viewBox="0 0 24 24"><rect x="14" y="4" width="4" height="16" rx="1" fill="currentColor"/><rect x="6" y="4" width="4" height="16" rx="1" fill="currentColor"/></svg>',
 zin:'<svg viewBox="0 0 24 24"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3M11 8v6M8 11h6"/></svg>',
 zout:'<svg viewBox="0 0 24 24"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3M8 11h6"/></svg>',
 reset:'<svg viewBox="0 0 24 24"><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/></svg>',
 left:'<svg viewBox="0 0 24 24"><path d="m15 18-6-6 6-6"/></svg>',
 right:'<svg viewBox="0 0 24 24"><path d="m9 18 6-6-6-6"/></svg>'};
const $ = id => document.getElementById(id);
$('play').innerHTML = IC.play; $('zin').innerHTML = IC.zin; $('zout').innerHTML = IC.zout;
$('zreset').innerHTML = IC.reset; $('pl').innerHTML = IC.left; $('pr').innerHTML = IC.right;
const L = 46;                                     // left gutter for axis labels
const DPR = Math.min(window.devicePixelRatio || 1, 2);
const DUR = Math.max(...C.lanes.map(l => l.dur));
let v0 = 0, v1 = DUR, fLo = 0, fHi = 8000, showW = true, showS = true, linked = true;
let sel = null, hover = null, active = 0, playEnd = null, dirty = true;
if (C.lanes.length > 1) $('linkWrap').style.display = '';

// colour map lookup table
const LUT = new Uint8ClampedArray(256 * 3);
(function(){ const s = C.heat.map(([t, h]) => [t, parseInt(h.slice(1,3),16), parseInt(h.slice(3,5),16), parseInt(h.slice(5,7),16)]);
 for (let i = 0; i < 256; i++){ const t = i/255; let k = 0; while (k < s.length - 2 && t > s[k+1][0]) k++;
  const a = s[k], b = s[k+1], u = (t - a[0]) / (b[0] - a[0] || 1);
  for (let c = 0; c < 3; c++) LUT[i*3+c] = a[c+1] + (b[c+1] - a[c+1]) * u; } })();
const b64f = s => new Float32Array(Uint8Array.from(atob(s), c => c.charCodeAt(0)).buffer);
const b64u = s => Uint8Array.from(atob(s), c => c.charCodeAt(0));
const fmt = t => { t = Math.max(0, t); const m = Math.floor(t / 60), s = t - 60 * m; return m + ':' + (s < 10 ? '0' : '') + s.toFixed(1); };
const fmtHz = f => f >= 1000 ? (f/1000).toFixed(f % 1000 ? 1 : 0) + ' kHz' : Math.round(f) + ' Hz';
const pcol = p => p == null ? '#475569' : (p >= 0.5 ? C.colors.syn : C.colors.real);

const lanesEl = $('lanes');
const LS = C.lanes.map((d, i) => {
  const el = document.createElement('div'); el.className = 'lane';
  const stripH = d.steps ? 12 : 0;
  el.innerHTML = `<div class="lh">${C.lanes.length > 1 ? `<span class="tag" style="background:${d.tone}">${d.tag}</span>` : ''}
    <span class="lt" title="${d.title.replace(/"/g,'&quot;')}">${d.title.replace(/</g,'&lt;')}</span>
    ${C.lanes.length > 1 ? `<button class="lp" aria-label="Play ${d.tag} only" title="Play this recording only">${IC.play}</button>` : ''}
    <span class="legend hide-sm"><span>−80 dB</span><canvas width="90" height="8"></canvas><span>0 dB</span></span></div>
    <div class="view" tabindex="0" aria-label="Waveform and spectrogram of ${d.title.replace(/"/g,'')}"><canvas class="main"></canvas><div class="tip"></div></div>
    <canvas class="axis"></canvas>`;
  lanesEl.appendChild(el);
  const audio = new Audio(d.audio); audio.preload = 'auto';
  const spec = d.spec, img = b64u(spec.data);
  const off = document.createElement('canvas'); off.width = spec.cols; off.height = spec.rows;
  const octx = off.getContext('2d'), id = octx.createImageData(spec.cols, spec.rows);
  for (let k = 0; k < img.length; k++){ const v = img[k]*3; id.data[k*4] = LUT[v]; id.data[k*4+1] = LUT[v+1]; id.data[k*4+2] = LUT[v+2]; id.data[k*4+3] = 255; }
  octx.putImageData(id, 0, 0);
  const lg = el.querySelector('.legend canvas').getContext('2d');
  for (let x = 0; x < 90; x++){ const v = Math.round(x/89*255)*3; lg.fillStyle = `rgb(${LUT[v]},${LUT[v+1]},${LUT[v+2]})`; lg.fillRect(x, 0, 1, 8); }
  const L_ = {d, el, audio, off, img, spec, stripH, mn: b64f(d.peaks.min), mx: b64f(d.peaks.max),
    view: el.querySelector('.view'), cv: el.querySelector('canvas.main'), ax: el.querySelector('canvas.axis'),
    tip: el.querySelector('.tip'), lp: el.querySelector('.lp')};
  audio.addEventListener('ended', () => { syncPlayBtns(); dirty = true; });
  audio.addEventListener('pause', syncPlayBtns); audio.addEventListener('play', syncPlayBtns);
  if (L_.lp) L_.lp.onclick = e => { e.stopPropagation(); active = i; toggle(i); };
  return L_;
});

let VH = C.viewH;
function size(){
  // fit the lanes into the space the (possibly wrapped) toolbar and footer leave
  const ws = $('ws'), tb = document.querySelector('.tb'), ft = document.querySelector('.foot'), n = LS.length;
  const strips = LS.reduce((a, o) => a + o.stripH, 0);
  const free = ws.clientHeight - 22 - tb.offsetHeight - 8 - ft.offsetHeight - 6 - n * 46 - (n - 1) * 6 - strips - 4;
  VH = Math.max(100, Math.min(C.viewH + 40, Math.floor(free / n)));
  LS.forEach(o => {
    const w = o.view.clientWidth, h = VH + o.stripH;
    o.cv.width = w * DPR; o.cv.height = h * DPR; o.cv.style.height = h + 'px';
    o.ax.width = w * DPR; o.ax.height = 22 * DPR; o.ax.style.height = '22px';
  });
  const m = $('minic'); m.width = m.clientWidth * DPR; m.height = 30 * DPR;
  dirty = true;
}
const tx = (t, W) => L + (t - v0) / (v1 - v0) * (W - L);
const xt = (x, W) => v0 + (x - L) / (W - L) * (v1 - v0);
function niceStep(span, px){ const raw = span / Math.max(2, px / 90); const p = Math.pow(10, Math.floor(Math.log10(raw)));
  for (const m of [1, 2, 5, 10]) if (m * p >= raw) return m * p; return 10 * p; }

function layout(o){ const H = VH; const top = o.stripH;
  if (showW && showS) return {wy: top, wh: Math.round(H * 0.32), sy: top + Math.round(H * 0.32) + 4, sh: H - Math.round(H * 0.32) - 4};
  if (showW) return {wy: top, wh: H, sy: 0, sh: 0};
  return {wy: 0, wh: 0, sy: top, sh: H}; }

function draw(o){
  const c = o.cv.getContext('2d'), W = o.cv.width / DPR, H = o.cv.height / DPR;
  c.setTransform(DPR, 0, 0, DPR, 0, 0); c.clearRect(0, 0, W, H);
  c.fillStyle = '#070b14'; c.fillRect(0, 0, W, H);
  const g = layout(o), d = o.d;
  // local-score strip
  if (d.steps){ for (const [a, b, p] of d.steps){ if (b < v0 || a > v1) continue;
      const x0 = Math.max(L, tx(a, W)), x1 = Math.min(W, tx(b, W)); c.fillStyle = pcol(p); c.globalAlpha = p == null ? .5 : .85;
      c.fillRect(x0, 2, Math.max(1, x1 - x0), 7); }
    c.globalAlpha = 1; c.fillStyle = '#5d6b82'; c.font = '9.5px Inter'; c.textBaseline = 'middle'; c.fillText('LOCAL', 6, 6); }
  // spectrogram
  if (g.sh > 0){
    const fmax = o.spec.fmax, rows = o.spec.rows, cols = o.spec.cols;
    const c0 = v0 / d.dur * cols, c1 = v1 / d.dur * cols;
    const r0 = (1 - Math.min(fHi, fmax) / fmax) * rows, r1 = (1 - fLo / fmax) * rows;
    const visEnd = Math.min(c1, cols);
    if (visEnd > c0){ c.imageSmoothingEnabled = (c1 - c0) < (W - L); c.imageSmoothingQuality = 'high';
      c.drawImage(o.off, c0, r0, visEnd - c0, r1 - r0, L, g.sy, (tx(Math.min(v1, d.dur), W) - L), g.sh); }
    // frequency grid and labels
    const span = fHi - fLo, st = span <= 1000 ? 200 : span <= 4000 ? 1000 : 2000;
    c.font = '10px "JetBrains Mono",monospace'; c.textBaseline = 'middle'; c.textAlign = 'right';
    for (let f = Math.ceil(fLo / st) * st; f <= fHi; f += st){ const y = g.sy + g.sh - (f - fLo) / span * g.sh;
      c.strokeStyle = 'rgba(230,235,242,.08)'; c.beginPath(); c.moveTo(L, y); c.lineTo(W, y); c.stroke();
      c.fillStyle = '#8b98ad'; c.fillText(f >= 1000 ? (f/1000) + 'k' : f, L - 6, Math.min(Math.max(y, g.sy + 6), g.sy + g.sh - 6)); }
    c.save(); c.translate(10, g.sy + g.sh / 2); c.rotate(-Math.PI/2); c.textAlign = 'center'; c.fillStyle = '#5d6b82'; c.font = '9.5px Inter';
    c.fillText('Hz', 0, 0); c.restore();
  }
  // waveform
  if (g.wh > 0){
    const mid = g.wy + g.wh / 2, amp = (g.wh / 2 - 4) / d.peaks.peak, n = o.mn.length, pos = o.audio.currentTime;
    c.strokeStyle = 'rgba(230,235,242,.07)'; c.beginPath(); c.moveTo(L, mid); c.lineTo(W, mid); c.stroke();
    for (let x = L; x < W; x++){
      const ta = xt(x, W), tb = xt(x + 1, W); if (ta > d.dur) break;
      let i0 = Math.floor(ta / d.dur * n), i1 = Math.max(i0 + 1, Math.ceil(tb / d.dur * n));
      i0 = Math.max(0, i0); i1 = Math.min(n, i1); let lo = 0, hi = 0;
      for (let i = i0; i < i1; i++){ if (o.mn[i] < lo) lo = o.mn[i]; if (o.mx[i] > hi) hi = o.mx[i]; }
      let col = d.tone;
      if (d.steps){ const tm = (ta + tb) / 2; let p = null; for (const s of d.steps) if (tm >= s[0] && tm < s[1]) { p = s[2]; break; } col = pcol(p); }
      c.fillStyle = col; c.globalAlpha = ta <= pos ? 1 : .62;
      c.fillRect(x, mid - hi * amp, 1, Math.max(1, (hi - lo) * amp));
    }
    c.globalAlpha = 1;
    c.fillStyle = '#5d6b82'; c.font = '9.5px Inter'; c.textAlign = 'left'; c.textBaseline = 'top'; c.fillText('AMP', 6, g.wy + 4);
  }
  // flagged segments
  if (d.segs) for (const [a, b, nm, p] of d.segs){ if (b < v0 || a > v1) continue;
    const x0 = Math.max(L, tx(a, W)), x1 = Math.min(W, tx(b, W));
    c.fillStyle = 'rgba(248,113,113,.10)'; c.fillRect(x0, o.stripH, x1 - x0, H - o.stripH);
    c.strokeStyle = 'rgba(248,113,113,.75)'; c.setLineDash([3, 3]); c.beginPath();
    c.moveTo(x0 + .5, o.stripH); c.lineTo(x0 + .5, H); c.moveTo(x1 - .5, o.stripH); c.lineTo(x1 - .5, H); c.stroke(); c.setLineDash([]);
    c.fillStyle = C.colors.syn; c.font = '600 10.5px Inter'; c.textAlign = 'left'; c.textBaseline = 'top';
    c.fillText(nm + ' ' + Math.round(p * 100) + '%', x0 + 4, o.stripH + 4); }
  // selection
  if (sel){ const x0 = Math.max(L, tx(sel[0], W)), x1 = Math.min(W, tx(sel[1], W));
    if (x1 > x0){ c.fillStyle = 'rgba(34,211,238,.13)'; c.fillRect(x0, 0, x1 - x0, H);
      c.strokeStyle = C.colors.cyan; c.lineWidth = 1.5; c.beginPath(); c.moveTo(x0, 0); c.lineTo(x0, H); c.moveTo(x1, 0); c.lineTo(x1, H); c.stroke(); c.lineWidth = 1; } }
  // hover crosshair
  if (hover && hover.o === o){ c.strokeStyle = 'rgba(230,235,242,.45)'; c.setLineDash([2, 3]); c.beginPath();
    c.moveTo(hover.x, 0); c.lineTo(hover.x, H); if (hover.f != null){ c.moveTo(L, hover.y); c.lineTo(W, hover.y); } c.stroke(); c.setLineDash([]); }
  else if (hover){ const x = tx(hover.t, W); c.strokeStyle = 'rgba(230,235,242,.3)'; c.setLineDash([2, 3]); c.beginPath(); c.moveTo(x, 0); c.lineTo(x, H); c.stroke(); c.setLineDash([]); }
  // playhead
  const ph = tx(o.audio.currentTime, W);
  if (ph >= L && ph <= W){ c.strokeStyle = '#ffffff'; c.lineWidth = 1.5; c.beginPath(); c.moveTo(ph, 0); c.lineTo(ph, H); c.stroke(); c.lineWidth = 1;
    c.fillStyle = '#ffffff'; c.beginPath(); c.moveTo(ph - 5, 0); c.lineTo(ph + 5, 0); c.lineTo(ph, 6); c.fill(); }
  c.fillStyle = '#111827'; c.fillRect(0, 0, L - 2, o.stripH);
  // time axis
  const a = o.ax.getContext('2d'); a.setTransform(DPR, 0, 0, DPR, 0, 0); a.clearRect(0, 0, W, 22);
  const step = niceStep(v1 - v0, W - L); a.font = '10px "JetBrains Mono",monospace'; a.fillStyle = '#8b98ad'; a.textAlign = 'center'; a.textBaseline = 'top';
  for (let t = Math.ceil(v0 / step) * step; t <= v1 + 1e-9; t += step){ const x = tx(t, W);
    a.fillStyle = '#2b3a52'; a.fillRect(x, 0, 1, 5); a.fillStyle = '#8b98ad';
    a.fillText(step < 1 ? t.toFixed(step < .1 ? 2 : 1) : (t >= 60 ? fmt(t).replace(/\.0$/, '') : t.toFixed(0) + 's'), Math.min(Math.max(x, L + 14), W - 16), 7); }
}

function drawMini(){
  const m = $('minic'), c = m.getContext('2d'), W = m.width / DPR, H = 30, o = LS[0];
  c.setTransform(DPR, 0, 0, DPR, 0, 0); c.clearRect(0, 0, W, H); c.fillStyle = '#070b14'; c.fillRect(0, 0, W, H);
  const n = o.mn.length, amp = (H / 2 - 3) / o.d.peaks.peak;
  c.fillStyle = 'rgba(139,152,173,.55)';
  for (let x = 0; x < W; x++){ const ta = x / W * DUR, tb = (x + 1) / W * DUR; if (ta > o.d.dur) break;
    const i0 = Math.floor(ta / o.d.dur * n), i1 = Math.min(n, Math.max(i0 + 1, Math.ceil(tb / o.d.dur * n))); let lo = 0, hi = 0;
    for (let i = i0; i < i1; i++){ if (o.mn[i] < lo) lo = o.mn[i]; if (o.mx[i] > hi) hi = o.mx[i]; }
    c.fillRect(x, H / 2 - hi * amp, 1, Math.max(1, (hi - lo) * amp)); }
  if (sel){ c.fillStyle = 'rgba(34,211,238,.25)'; c.fillRect(sel[0] / DUR * W, 0, (sel[1] - sel[0]) / DUR * W, H); }
  const x0 = v0 / DUR * W, x1 = v1 / DUR * W;
  c.fillStyle = 'rgba(34,211,238,.10)'; c.fillRect(x0, 0, x1 - x0, H);
  c.strokeStyle = C.colors.cyan; c.lineWidth = 1.5; c.strokeRect(x0 + .75, .75, Math.max(2, x1 - x0 - 1.5), H - 1.5); c.lineWidth = 1;
  c.fillStyle = '#ffffff'; c.fillRect(LS[active].audio.currentTime / DUR * W, 0, 1.5, H);
}

function frame(){
  const playing = LS.some(o => !o.audio.paused);
  if (playing){
    const o = LS[active].audio.paused ? LS.find(q => !q.audio.paused) : LS[active];
    if (playEnd != null && o.audio.currentTime >= playEnd){ LS.forEach(q => q.audio.pause()); playEnd = null; }
    const t = o.audio.currentTime, span = v1 - v0;
    if (span < DUR - 1e-6 && (t > v1 || t < v0)){ v0 = Math.max(0, Math.min(t - span * .1, DUR - span)); v1 = v0 + span; }
    dirty = true;
  }
  if (dirty){ LS.forEach(draw); drawMini(); updTime(); dirty = false; }
  requestAnimationFrame(frame);
}
function updTime(){ const o = LS[active]; $('time').innerHTML = `<b>${fmt(o.audio.currentTime)}</b> / ${fmt(o.d.dur)}`; }
function syncPlayBtns(){ const any = LS.some(o => !o.audio.paused);
  $('play').innerHTML = any ? IC.pause : IC.play; $('play').setAttribute('aria-label', any ? 'Pause' : 'Play');
  LS.forEach(o => { if (o.lp) o.lp.innerHTML = o.audio.paused ? IC.play : IC.pause; }); dirty = true; }
function targets(i){ return (linked && LS.length > 1) ? LS : [LS[i == null ? active : i]]; }
function toggle(i){
  const ts = i == null ? targets() : [LS[i]];
  if (ts.some(o => !o.audio.paused)){ LS.forEach(o => o.audio.pause()); return; }
  const t0 = LS[i == null ? active : i].audio.currentTime;
  ts.forEach(o => { if (o.audio.currentTime >= o.d.dur - .05) o.audio.currentTime = 0; if (ts.length > 1) o.audio.currentTime = Math.min(t0, o.d.dur);
    o.audio.play().catch(() => {}); });
}
function seek(t){ targets().forEach(o => { o.audio.currentTime = Math.max(0, Math.min(t, o.d.dur)); }); dirty = true; }
function zoom(f, at){ const span = v1 - v0, ns = Math.max(Math.min(DUR, span * f), Math.min(.05, DUR));
  at = at == null ? (v0 + v1) / 2 : at; const r = (at - v0) / span;
  v0 = Math.max(0, Math.min(at - r * ns, DUR - ns)); v1 = v0 + ns; dirty = true; }
function pan(f){ const span = v1 - v0; v0 = Math.max(0, Math.min(v0 + f * span, DUR - span)); v1 = v0 + span; dirty = true; }
function setSel(s){ sel = s; const el = $('sel');
  if (!s){ el.innerHTML = '<span class="hint">Drag across the view to select a region · click to seek · Ctrl + wheel to zoom</span>'; dirty = true; return; }
  el.innerHTML = `Selection <b>${fmt(s[0])}–${fmt(s[1])}</b> (${(s[1] - s[0]).toFixed(2)} s)
    <button id="sp" title="Play the selected region">${IC.play} Play</button><button id="sz" title="Zoom to the selection">${IC.zin} Zoom</button><button id="sc" title="Clear the selection" aria-label="Clear selection">✕</button>`;
  $('sp').onclick = () => { LS.forEach(o => o.audio.pause()); targets().forEach(o => { o.audio.currentTime = Math.min(s[0], o.d.dur); o.audio.play().catch(() => {}); }); playEnd = s[1]; };
  $('sz').onclick = () => { v0 = s[0]; v1 = Math.max(s[1], s[0] + .05); dirty = true; };
  $('sc').onclick = () => setSel(null); dirty = true; }

LS.forEach((o, i) => {
  let down = null;
  const pos = e => { const r = o.view.getBoundingClientRect(); return {x: e.clientX - r.left, y: e.clientY - r.top, W: r.width}; };
  o.view.addEventListener('pointerdown', e => { const p = pos(e); if (p.x < L) return; active = i; down = {x: p.x, t: xt(p.x, p.W)}; o.view.setPointerCapture(e.pointerId); });
  o.view.addEventListener('pointermove', e => {
    const p = pos(e), t = Math.max(0, Math.min(xt(p.x, p.W), DUR));
    if (down && Math.abs(p.x - down.x) > 4){ const a = Math.min(down.t, t), b = Math.max(down.t, t); sel = [Math.max(0, a), Math.min(DUR, b)]; }
    const g = layout(o); let f = null, db = null, line = '';
    if (p.x >= L && g.sh > 0 && p.y >= g.sy && p.y <= g.sy + g.sh && t <= o.d.dur){
      f = fLo + (1 - (p.y - g.sy) / g.sh) * (fHi - fLo);
      const col = Math.min(o.spec.cols - 1, Math.floor(t / o.d.dur * o.spec.cols)), row = Math.min(o.spec.rows - 1, Math.max(0, Math.round((1 - f / o.spec.fmax) * (o.spec.rows - 1))));
      db = o.spec.db_lo + o.img[row * o.spec.cols + col] / 255 * (o.spec.db_hi - o.spec.db_lo);
      line = `<br><i>freq</i> ${fmtHz(f)}<br><i>level</i> ${db.toFixed(1)} dB`;
    } else if (p.x >= L && g.wh > 0 && p.y >= g.wy && p.y <= g.wy + g.wh && t <= o.d.dur){
      const n = o.mn.length, k = Math.min(n - 1, Math.floor(t / o.d.dur * n)); const a = Math.max(Math.abs(o.mn[k]), Math.abs(o.mx[k]));
      line = `<br><i>peak</i> ${(20 * Math.log10(a + 1e-9)).toFixed(1)} dBFS`; }
    if (o.d.steps && t <= o.d.dur){ const s = o.d.steps.find(s => t >= s[0] && t < s[1]);
      if (s) line += `<br><i>local</i> ${s[2] == null ? 'no speech' : (s[2] >= .5 ? '<span style="color:#f87171">' : '<span style="color:#34d399">') + (s[2] == null ? '' : Math.round(s[2] * 100) + '% synthetic</span>')}`; }
    if (o.d.segs){ const s = o.d.segs.find(s => t >= s[0] && t <= s[1]); if (s) line += `<br><span style="color:#f87171">flagged ${s[2]}</span>`; }
    hover = p.x >= L ? {o, x: p.x, y: p.y, t, f} : null;
    if (hover){ o.tip.style.display = 'block'; o.tip.innerHTML = `<i>time</i> ${fmt(t)}${line}`;
      const tw = o.tip.offsetWidth, th = o.tip.offsetHeight; o.tip.style.left = (p.x + 14 + tw > p.W ? p.x - tw - 12 : p.x + 14) + 'px';
      o.tip.style.top = Math.max(4, Math.min(p.y - th / 2, o.view.clientHeight - th - 4)) + 'px'; }
    else o.tip.style.display = 'none';
    dirty = true; });
  o.view.addEventListener('pointerup', e => { const p = pos(e);
    if (down){ if (Math.abs(p.x - down.x) <= 4){ seek(xt(p.x, p.W)); } else setSel(sel); } down = null; });
  o.view.addEventListener('pointerleave', () => { hover = null; o.tip.style.display = 'none'; dirty = true; });
  o.view.addEventListener('wheel', e => { if (e.ctrlKey || e.metaKey){ e.preventDefault(); const p = pos(e); zoom(e.deltaY > 0 ? 1.25 : .8, xt(p.x, p.W)); }
    else if (e.shiftKey){ e.preventDefault(); pan((e.deltaY || e.deltaX) > 0 ? .15 : -.15); } }, {passive: false});
});
(function(){ const m = $('mini'); let drag = false;
  const go = e => { const r = m.getBoundingClientRect(), t = (e.clientX - r.left) / r.width * DUR, span = v1 - v0;
    v0 = Math.max(0, Math.min(t - span / 2, DUR - span)); v1 = v0 + span; dirty = true; };
  m.addEventListener('pointerdown', e => { drag = true; m.setPointerCapture(e.pointerId); go(e); });
  m.addEventListener('pointermove', e => { if (drag) go(e); });
  m.addEventListener('pointerup', () => drag = false); })();

$('play').onclick = () => toggle();
$('zin').onclick = () => zoom(.5); $('zout').onclick = () => zoom(2);
$('zreset').onclick = () => { v0 = 0; v1 = DUR; dirty = true; };
$('pl').onclick = () => pan(-.25); $('pr').onclick = () => pan(.25);
function setView(){ $('vw').classList.toggle('on', showW); $('vs').classList.toggle('on', showS);
  $('vw').setAttribute('aria-pressed', showW); $('vs').setAttribute('aria-pressed', showS); $('fr').disabled = !showS; dirty = true; }
$('vw').onclick = () => { if (showW && !showS) return; showW = !showW; setView(); };
$('vs').onclick = () => { if (showS && !showW) return; showS = !showS; setView(); };
$('fr').onchange = e => { [fLo, fHi] = e.target.value.split(',').map(Number); dirty = true; };
$('vol').oninput = e => LS.forEach(o => o.audio.volume = +e.target.value);
$('spd').onchange = e => LS.forEach(o => o.audio.playbackRate = +e.target.value);
$('link').onclick = () => { linked = !linked; $('link').classList.toggle('on', linked); $('link').setAttribute('aria-pressed', linked);
  $('link').textContent = linked ? 'Linked' : 'Independent'; };
LS.forEach(o => o.audio.volume = .9);
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'SELECT' || e.target.tagName === 'INPUT') return;
  if (e.code === 'Space'){ e.preventDefault(); toggle(); }
  else if (e.key === '+' || e.key === '='){ zoom(.5); } else if (e.key === '-'){ zoom(2); }
  else if (e.key === '0'){ v0 = 0; v1 = DUR; dirty = true; }
  else if (e.key === 'ArrowLeft'){ pan(-.25); } else if (e.key === 'ArrowRight'){ pan(.25); }
  else if (e.key === 'Escape'){ setSel(null); } });
setView(); syncPlayBtns();
new ResizeObserver(size).observe(document.body); size(); requestAnimationFrame(frame);
})();
</script></body></html>"""

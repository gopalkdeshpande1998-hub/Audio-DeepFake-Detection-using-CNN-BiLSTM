"""VoiceGuard visual system: colour tokens, the CSS layer and small HTML components shared by every page.

Dark forensic workstation: navy ground, slate panels, one cyan accent; green means REAL, coral means SYNTHETIC,
amber means caution. All motion is short and switched off under prefers-reduced-motion.
"""

from __future__ import annotations

import html

import streamlit as st

from ui.icons import icon

# colour tokens (names kept for the existing chart and report code)
CYAN = "#22d3ee"
GREEN = "#34d399"          # REAL
RED = "#f87171"            # SYNTHETIC
AMBER = "#fbbf24"          # caution
GREY = "#64748b"           # no speech / not scored
BLUE = CYAN                # the old accent maps to the new one
INK = "#e6ebf2"
MUTED = "#8b98ad"
LINE = "#1f2a3c"
PANEL = "#111827"
BG = "#0a0f1a"

VERDICT_WORD = {"Fake": "SYNTHETIC", "Real": "REAL"}
VERDICT_COLOR = {"Fake": RED, "Real": GREEN}

CSS = """
<style>
:root{--vg-bg:#0a0f1a;--vg-panel:#111827;--vg-panel2:#0d1524;--vg-raise:#152033;--vg-line:#1f2a3c;--vg-line2:#2b3a52;
--vg-text:#e6ebf2;--vg-mute:#8b98ad;--vg-dim:#5d6b82;--vg-cyan:#22d3ee;--vg-real:#34d399;--vg-syn:#f87171;
--vg-amber:#fbbf24;--vg-ease:cubic-bezier(.2,.8,.2,1)}
.stApp{background:radial-gradient(1200px 500px at 70% -200px,rgba(34,211,238,.06),transparent 70%),var(--vg-bg)}
header[data-testid="stHeader"]{background:rgba(10,15,26,.72);backdrop-filter:blur(8px);border-bottom:1px solid var(--vg-line)}
.block-container{padding-top:3.2rem;padding-bottom:4rem;max-width:1320px}
h1,h2,h3,h4{letter-spacing:-.015em}
.vg-i{flex:none;vertical-align:-.15em}
.vg-num{font-variant-numeric:tabular-nums}

/* page and section headers */
.vg-hdr{padding:10px 0 18px}
.vg-eyebrow{display:inline-flex;align-items:center;gap:8px;font-size:.72rem;font-weight:600;letter-spacing:.12em;
  text-transform:uppercase;color:var(--vg-cyan)}
.vg-hdr h1{font-size:clamp(1.7rem,3.2vw,2.35rem);line-height:1.12;font-weight:700;margin:.45rem 0 .5rem;padding:0;color:var(--vg-text)}
.vg-hdr p{font-size:1rem;max-width:70ch;color:var(--vg-mute);margin:0;line-height:1.55}
.vg-sec{display:flex;align-items:flex-end;justify-content:space-between;gap:16px;margin:30px 0 12px;flex-wrap:wrap}
.vg-sec .t{display:flex;align-items:center;gap:12px}
.vg-sec .ib{width:34px;height:34px;border-radius:8px;display:grid;place-items:center;background:rgba(34,211,238,.08);
  border:1px solid rgba(34,211,238,.22);color:var(--vg-cyan)}
.vg-sec h3{font-size:1.08rem;font-weight:650;margin:0;padding:0;color:var(--vg-text)}
.vg-sec .s{font-size:.85rem;color:var(--vg-mute);margin-top:2px}
.vg-sec .m{font-size:.78rem;color:var(--vg-dim)}
.vg-sec.big h3{font-size:1.35rem;letter-spacing:.01em;text-transform:uppercase}

/* panels, cards, stats */
.vg-panel{background:var(--vg-panel);border:1px solid var(--vg-line);border-radius:12px;padding:18px 20px}
.vg-card{background:var(--vg-panel);border:1px solid var(--vg-line);border-radius:12px;padding:18px;height:100%;
  transition:transform .18s var(--vg-ease),border-color .18s ease,background .18s ease}
.vg-card:hover{transform:translateY(-2px);border-color:var(--vg-line2);background:var(--vg-raise)}
.vg-card .ib{width:32px;height:32px;border-radius:8px;display:grid;place-items:center;margin-bottom:12px;
  background:rgba(34,211,238,.08);color:var(--vg-cyan)}
.vg-card h4{margin:0 0 6px;font-size:.98rem;font-weight:600;padding:0;color:var(--vg-text)}
.vg-card p{margin:0;font-size:.88rem;color:var(--vg-mute);line-height:1.5}
.vg-stat{background:var(--vg-panel);border:1px solid var(--vg-line);border-radius:12px;padding:16px 18px;height:100%}
.vg-stat .k{display:flex;align-items:center;gap:8px;font-size:.8rem;color:var(--vg-mute)}
.vg-stat .v{font-size:clamp(1.45rem,2.3vw,1.9rem);font-weight:700;line-height:1.1;margin:8px 0 4px;
  font-variant-numeric:tabular-nums;color:var(--vg-text);white-space:nowrap}
.vg-stat .s{font-size:.74rem;color:var(--vg-dim);line-height:1.4}

/* chips and dots */
.vg-chip{display:inline-flex;align-items:center;gap:6px;font-size:.72rem;font-weight:600;letter-spacing:.04em;
  padding:3px 9px;border-radius:999px;border:1px solid currentColor;white-space:nowrap;line-height:1.4}
.vg-chip.solid{border:0;color:#0a0f1a}
.vg-chip.real{color:var(--vg-real);background:rgba(52,211,153,.08)}
.vg-chip.syn{color:var(--vg-syn);background:rgba(248,113,113,.08)}
.vg-chip.amber{color:var(--vg-amber);background:rgba(251,191,36,.08)}
.vg-chip.cyan{color:var(--vg-cyan);background:rgba(34,211,238,.08)}
.vg-chip.mute{color:var(--vg-mute);background:rgba(139,152,173,.06);border-color:var(--vg-line2)}
.vg-dot{width:8px;height:8px;border-radius:50%;display:inline-block;background:currentColor;box-shadow:0 0 0 3px rgba(255,255,255,.04)}

/* workflow stepper */
.vg-steps{display:flex;gap:6px;flex-wrap:wrap;margin:4px 0 18px}
.vg-step{display:flex;align-items:center;gap:8px;padding:6px 12px 6px 6px;border-radius:999px;border:1px solid var(--vg-line);
  background:var(--vg-panel2);font-size:.8rem;color:var(--vg-dim);transition:all .2s ease}
.vg-step .n{width:22px;height:22px;border-radius:50%;display:grid;place-items:center;font-size:.72rem;font-weight:700;
  background:var(--vg-line);color:var(--vg-mute)}
.vg-step.done{color:var(--vg-mute)} .vg-step.done .n{background:rgba(52,211,153,.15);color:var(--vg-real)}
.vg-step.now{color:var(--vg-text);border-color:rgba(34,211,238,.45);background:rgba(34,211,238,.06)}
.vg-step.now .n{background:var(--vg-cyan);color:#0a0f1a}

/* file card */
.vg-file{display:flex;align-items:center;gap:14px;background:var(--vg-panel);border:1px solid var(--vg-line);
  border-radius:12px;padding:14px 16px;animation:vg-in .35s var(--vg-ease) both}
.vg-file .ib{width:42px;height:42px;border-radius:10px;display:grid;place-items:center;background:rgba(34,211,238,.1);
  color:var(--vg-cyan)}
.vg-file .nm{font-weight:600;color:var(--vg-text);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.vg-file .meta{display:flex;flex-wrap:wrap;gap:6px 14px;font-size:.78rem;color:var(--vg-mute);margin-top:3px}
.vg-file .meta b{color:var(--vg-text);font-weight:500}
.vg-file .body{min-width:0;flex:1}
.vg-file.a .ib{background:rgba(34,211,238,.1);color:var(--vg-cyan)}
.vg-file.b .ib{background:rgba(251,191,36,.1);color:var(--vg-amber)}

/* verdict */
.vg-verdict{position:relative;display:grid;grid-template-columns:auto 1fr auto;gap:28px;align-items:center;
  background:linear-gradient(105deg,var(--tint) 0%,rgba(17,24,39,0) 55%),var(--vg-panel);border:1px solid var(--vg-line);
  border-left:3px solid var(--c);border-radius:14px;padding:24px 28px;animation:vg-in .45s var(--vg-ease) both}
.vg-verdict .k{display:flex;align-items:center;gap:8px;font-size:.72rem;font-weight:600;letter-spacing:.12em;
  text-transform:uppercase;color:var(--vg-mute)}
.vg-verdict .w{font-size:clamp(2.4rem,5.2vw,3.6rem);font-weight:800;letter-spacing:-.025em;line-height:1;color:var(--c);
  margin:8px 0 8px}
.vg-verdict .d{font-size:.95rem;color:var(--vg-mute);max-width:52ch;line-height:1.5}
.vg-verdict .d b{color:var(--vg-text)}
.vg-facts{display:grid;grid-template-columns:repeat(2,minmax(120px,auto));gap:14px 26px;border-left:1px solid var(--vg-line);
  padding-left:26px}
.vg-facts .fk{font-size:.72rem;color:var(--vg-dim);text-transform:uppercase;letter-spacing:.08em}
.vg-facts .fv{font-size:1.05rem;font-weight:600;color:var(--vg-text);margin-top:3px;font-variant-numeric:tabular-nums;
  display:flex;align-items:center;gap:8px}
.vg-ring{position:relative;display:grid;place-items:center}
.vg-ring svg{transform:rotate(-90deg)}
.vg-ring .arc{animation:vg-ring 1.1s var(--vg-ease) both}
.vg-ring .lbl{position:absolute;text-align:center;line-height:1.1}
.vg-ring .lbl b{display:block;font-size:1.45rem;font-weight:700;color:var(--vg-text);font-variant-numeric:tabular-nums}
.vg-ring .lbl span{font-size:.64rem;color:var(--vg-mute);text-transform:uppercase;letter-spacing:.08em}
.vg-ring.sm .lbl b{font-size:1rem}
.vg-meter{margin-top:16px}
.vg-meter .lab{display:flex;justify-content:space-between;font-size:.8rem;color:var(--vg-mute);margin-bottom:6px}
.vg-meter .lab b{font-variant-numeric:tabular-nums}
.vg-track{position:relative;display:flex;height:10px;border-radius:6px;overflow:hidden;background:var(--vg-line)}
.vg-track .r{background:var(--vg-real);width:var(--w);animation:vg-grow 1s var(--vg-ease) both}
.vg-track .f{background:var(--vg-syn);width:var(--w);animation:vg-grow 1s var(--vg-ease) both;margin-left:auto}
.vg-track .mid{position:absolute;left:50%;top:-3px;bottom:-3px;width:2px;background:var(--vg-text);opacity:.55}

/* consensus and warnings */
.vg-cons{display:flex;align-items:center;gap:18px;flex-wrap:wrap}
.vg-pips{display:flex;gap:6px}
.vg-pips i{width:28px;height:8px;border-radius:4px;background:var(--c);display:block;animation:vg-in .4s var(--vg-ease) both;
  animation-delay:var(--d)}
.vg-warn{display:flex;gap:14px;align-items:flex-start;border:1px solid rgba(251,191,36,.35);background:rgba(251,191,36,.06);
  border-radius:12px;padding:14px 18px;color:var(--vg-text);margin-top:14px;animation:vg-in .4s var(--vg-ease) both}
.vg-warn .ib{color:var(--vg-amber);margin-top:1px}
.vg-warn b{color:var(--vg-amber)}
.vg-warn p{margin:3px 0 0;font-size:.88rem;color:var(--vg-mute);line-height:1.5}
.vg-note{display:flex;gap:12px;align-items:flex-start;border:1px solid var(--vg-line);background:var(--vg-panel2);
  border-radius:12px;padding:12px 16px;font-size:.86rem;color:var(--vg-mute);line-height:1.5}
.vg-note .ib{color:var(--vg-cyan);margin-top:2px}

/* model cards */
.vg-model{background:var(--vg-panel);border:1px solid var(--vg-line);border-radius:12px;padding:18px;height:100%;
  animation:vg-in .5s var(--vg-ease) both;animation-delay:var(--d,0s);transition:border-color .18s ease,transform .18s var(--vg-ease)}
.vg-model:hover{border-color:var(--vg-line2);transform:translateY(-2px)}
.vg-model .top{display:flex;justify-content:space-between;align-items:flex-start;gap:10px}
.vg-model .nm{font-weight:600;font-size:.95rem;color:var(--vg-text)}
.vg-model .sub{font-size:.75rem;color:var(--vg-dim);margin-top:2px}
.vg-model .mid{display:flex;align-items:center;gap:16px;margin:16px 0 14px}
.vg-model .bars{flex:1;display:grid;gap:10px}
.vg-mbar .l{display:flex;justify-content:space-between;font-size:.75rem;color:var(--vg-mute);margin-bottom:4px}
.vg-mbar .l b{color:var(--vg-text);font-variant-numeric:tabular-nums;font-weight:600}
.vg-mbar .t{height:6px;border-radius:4px;background:var(--vg-line);overflow:hidden}
.vg-mbar .t>div{height:100%;width:var(--w);background:var(--c);animation:vg-grow .9s var(--vg-ease) both;animation-delay:var(--d,0s)}
.vg-model .foot{display:flex;justify-content:space-between;align-items:center;gap:8px;padding-top:12px;
  border-top:1px solid var(--vg-line);font-size:.76rem;color:var(--vg-mute);flex-wrap:wrap}
.vg-model .st{display:inline-flex;align-items:center;gap:6px}

/* comparison chart */
.vg-cmp{display:grid;gap:12px}
.vg-cmp .row{display:grid;grid-template-columns:minmax(120px,200px) 1fr;gap:16px;align-items:center}
.vg-cmp .nm{font-size:.85rem;color:var(--vg-text);font-weight:500;line-height:1.25}
.vg-cmp .nm span{display:block;font-size:.72rem;color:var(--vg-dim);font-weight:400}
.vg-cmp .row.ens .nm{color:var(--vg-cyan);font-weight:600}
.vg-cmp .bar{position:relative;display:flex;height:26px;border-radius:6px;background:var(--vg-line)}
.vg-cmp .seg{position:relative;height:100%;width:var(--w);display:flex;align-items:center;font-size:.74rem;font-weight:600;
  color:#06231a;padding:0 8px;white-space:nowrap;animation:vg-grow .9s var(--vg-ease) both;animation-delay:var(--d,0s);
  cursor:default;transition:filter .15s ease}
.vg-cmp .seg.r{background:var(--vg-real);border-radius:6px 0 0 6px}
.vg-cmp .seg.f{background:var(--vg-syn);color:#2a0b0b;justify-content:flex-end;border-radius:0 6px 6px 0;margin-left:auto}
.vg-cmp .seg:hover{filter:brightness(1.12)}
.vg-cmp .seg[data-tip]:hover::after{content:attr(data-tip);position:absolute;bottom:calc(100% + 8px);left:50%;
  transform:translateX(-50%);background:#0b1220;border:1px solid var(--vg-line2);color:var(--vg-text);padding:6px 10px;
  border-radius:6px;font-weight:500;font-size:.75rem;white-space:nowrap;z-index:5;box-shadow:0 8px 24px rgba(0,0,0,.45)}
.vg-cmp .thr{position:absolute;left:50%;top:-5px;bottom:-5px;border-left:1.5px dashed rgba(230,235,242,.6);pointer-events:none}
.vg-cmp .axis{display:grid;grid-template-columns:minmax(120px,200px) 1fr;gap:16px;font-size:.7rem;color:var(--vg-dim)}
.vg-cmp .axis div{display:flex;justify-content:space-between}
.vg-legend2{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:.78rem;color:var(--vg-mute)}
.vg-legend2 span{display:inline-flex;align-items:center;gap:7px}
.vg-legend2 i{width:12px;height:12px;border-radius:3px;display:inline-block}

/* explainability cards */
.vg-x{background:var(--vg-panel);border:1px solid var(--vg-line);border-radius:12px;padding:16px 18px;height:100%;
  border-top:2px solid var(--c);animation:vg-in .45s var(--vg-ease) both;animation-delay:var(--d,0s);
  transition:border-color .18s ease,background .18s ease}
.vg-x:hover{background:var(--vg-raise)}
.vg-x .top{display:flex;justify-content:space-between;align-items:center;gap:10px}
.vg-x .tt{display:flex;align-items:center;gap:10px;font-weight:600;font-size:.92rem;color:var(--vg-text)}
.vg-x .tt .ib{width:30px;height:30px;border-radius:8px;display:grid;place-items:center;color:var(--c);
  background:color-mix(in srgb,var(--c) 12%,transparent)}
.vg-x .val{font-size:1.35rem;font-weight:700;color:var(--vg-text);margin:12px 0 2px;font-variant-numeric:tabular-nums}
.vg-x .val small{font-size:.78rem;font-weight:500;color:var(--vg-mute);margin-left:4px}
.vg-x .val.na{font-size:.95rem;color:var(--vg-dim);font-weight:500}
.vg-x .txt{font-size:.84rem;color:var(--vg-mute);line-height:1.5;margin-top:6px}
.vg-x details{margin-top:10px;border-top:1px solid var(--vg-line);padding-top:8px}
.vg-x summary{cursor:pointer;font-size:.78rem;color:var(--vg-cyan);list-style:none;display:flex;align-items:center;gap:6px}
.vg-x summary::-webkit-details-marker{display:none}
.vg-x summary::before{content:"+";font-weight:700;width:12px}
.vg-x details[open] summary::before{content:"\\2212"}
.vg-x details div{font-size:.8rem;color:var(--vg-mute);line-height:1.55;margin-top:6px}
.vg-x details ul{margin:4px 0 0;padding-left:1rem}
.vg-pct{position:relative;height:6px;border-radius:4px;background:var(--vg-line);margin:12px 0 4px}
.vg-pct .band{position:absolute;left:5%;right:5%;top:0;bottom:0;background:rgba(52,211,153,.22);border-radius:4px}
.vg-pct .dot{position:absolute;top:50%;left:var(--p);width:12px;height:12px;border-radius:50%;background:var(--c);
  border:2px solid var(--vg-panel);transform:translate(-50%,-50%);animation:vg-slide .9s var(--vg-ease) both}
.vg-pct-l{display:flex;justify-content:space-between;font-size:.66rem;color:var(--vg-dim)}
.vg-sev{font-size:.68rem;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:var(--c);
  display:inline-flex;align-items:center;gap:6px}
.vg-mini{display:flex;gap:2px;align-items:flex-end;height:28px;margin-top:10px}
.vg-mini i{flex:1;min-width:2px;border-radius:2px 2px 0 0;background:var(--c);opacity:.9}
.vg-seg{position:relative;height:24px;border-radius:6px;background:#070b14;overflow:hidden;margin-top:12px}
.vg-seg i{position:absolute;top:0;bottom:0;background:var(--c);border-right:1px solid var(--vg-panel);font:700 .58rem Inter,
  sans-serif;font-style:normal;letter-spacing:.05em;color:rgba(5,12,22,.82);display:flex;align-items:center;
  justify-content:center;overflow:hidden;white-space:nowrap;opacity:.92}
.vg-seg-ax{position:relative;height:14px;font-size:.62rem;color:var(--vg-dim);font-variant-numeric:tabular-nums}
.vg-seg-ax span{position:absolute;top:2px;transform:translateX(-50%)}

/* empty state, skeleton, loader */
.vg-empty{border:1px dashed var(--vg-line2);border-radius:14px;padding:34px 24px;text-align:center;background:var(--vg-panel2)}
.vg-empty .ib{width:46px;height:46px;border-radius:12px;display:grid;place-items:center;margin:0 auto 12px;
  background:rgba(34,211,238,.08);color:var(--vg-cyan)}
.vg-empty h4{margin:0 0 6px;font-size:1rem;font-weight:600;color:var(--vg-text);padding:0}
.vg-empty p{margin:0 auto;font-size:.86rem;color:var(--vg-mute);max-width:56ch;line-height:1.5}
.vg-skel{border-radius:12px;border:1px solid var(--vg-line);height:var(--h,120px);
  background:linear-gradient(90deg,var(--vg-panel) 0%,var(--vg-raise) 40%,var(--vg-panel) 80%);background-size:200% 100%;
  animation:vg-shimmer 1.4s linear infinite}
.vg-skel-row{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-top:14px}
.vg-loader{display:flex;align-items:center;gap:22px;background:var(--vg-panel);border:1px solid rgba(34,211,238,.25);
  border-radius:14px;padding:20px 24px;animation:vg-in .35s var(--vg-ease) both}
.vg-bars{display:flex;align-items:center;gap:3px;height:44px}
.vg-bars i{width:4px;border-radius:2px;background:var(--vg-cyan);height:100%;transform-origin:center;
  animation:vg-eq 1.1s ease-in-out infinite;animation-delay:var(--d)}
.vg-loader .t{font-weight:600;color:var(--vg-text)}
.vg-loader .s{font-size:.85rem;color:var(--vg-mute);margin-top:3px;line-height:1.5}
.vg-pulse{width:10px;height:10px;border-radius:50%;background:var(--vg-cyan);animation:vg-pulse 1.6s ease-out infinite}

/* legacy helpers still used by charts and the report */
.adfd-legend{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:.8rem;color:var(--vg-mute);margin:4px 0 6px}
.adfd-legend span.sw{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:7px;vertical-align:-1px}
.adfd-bar{height:6px;border-radius:4px;background:var(--vg-line);overflow:hidden;margin:4px 0 10px}
.adfd-bar>div{height:100%;background:var(--c);animation:vg-grow .9s var(--vg-ease) both;width:var(--w)}
.adfd-trust .lvl{font-size:1.2rem;font-weight:700}

/* Streamlit widgets in the system */
.stButton>button,.stDownloadButton>button,[data-testid="stPageLink"] a{transition:transform .15s var(--vg-ease),
  border-color .15s ease,background .15s ease,box-shadow .15s ease}
.stButton>button:hover,.stDownloadButton>button:hover{transform:translateY(-1px)}
.stButton>button:active,.stDownloadButton>button:active{transform:translateY(0) scale(.98)}
.stButton>button[kind="primary"]{color:#06202a;font-weight:600;box-shadow:0 0 0 1px rgba(34,211,238,.4),0 6px 20px -8px rgba(34,211,238,.55)}
[data-testid="stPageLink"] a{border:1px solid var(--vg-line);background:var(--vg-panel);border-radius:8px;padding:6px 12px}
[data-testid="stPageLink"] a:hover{border-color:rgba(34,211,238,.45);background:var(--vg-raise)}
[data-testid="stSidebarNav"] a{border-radius:8px}
[data-testid="stFileUploaderDropzone"]{border:1.5px dashed var(--vg-line2)!important;background:var(--vg-panel2)!important;
  border-radius:12px!important;padding:26px 20px!important;transition:border-color .2s ease,background .2s ease}
[data-testid="stFileUploaderDropzone"]:hover{border-color:var(--vg-cyan)!important;background:rgba(34,211,238,.04)!important}
[data-testid="stExpander"] details{border-color:var(--vg-line)!important;border-radius:12px!important;background:var(--vg-panel)}
[data-testid="stExpander"] summary:hover{color:var(--vg-cyan)}
.stTabs [data-baseweb="tab-list"]{gap:4px;border-bottom:1px solid var(--vg-line)}
.stTabs [data-baseweb="tab"]{padding:8px 14px}
[data-testid="stVegaLiteChart"],[data-testid="stArrowVegaLiteChart"]{background:var(--vg-panel);border:1px solid var(--vg-line);
  border-radius:12px;padding:12px 10px 6px}
iframe[title="st.iframe"],[data-testid="stIFrame"],[data-testid="stCustomComponentV1"]{border-radius:12px}

@keyframes vg-in{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
@keyframes vg-grow{from{width:0}}
@keyframes vg-slide{from{left:0;opacity:0}}
@keyframes vg-ring{from{stroke-dashoffset:var(--circ)}}
@keyframes vg-shimmer{from{background-position:200% 0}to{background-position:-200% 0}}
@keyframes vg-eq{0%,100%{transform:scaleY(.25)}50%{transform:scaleY(1)}}
@keyframes vg-pulse{0%{box-shadow:0 0 0 0 rgba(34,211,238,.55)}100%{box-shadow:0 0 0 12px rgba(34,211,238,0)}}

@media (max-width:900px){
  .vg-verdict{grid-template-columns:auto 1fr;padding:20px}
  .vg-facts{grid-column:1/-1;border-left:0;padding-left:0;border-top:1px solid var(--vg-line);padding-top:16px;
    grid-template-columns:repeat(2,1fr)}
}
@media (max-width:640px){
  .block-container{padding-left:1rem;padding-right:1rem}
  .vg-verdict{grid-template-columns:1fr;justify-items:start;gap:18px}
  .vg-cmp .row,.vg-cmp .axis{grid-template-columns:1fr;gap:6px}
  .vg-cmp .axis>span{display:none}
  .vg-skel-row{grid-template-columns:1fr}
  .vg-loader{flex-direction:column;align-items:flex-start}
  .vg-step .lb{display:none}
  .vg-step.now .lb{display:inline}
}
@media (prefers-reduced-motion:reduce){
  *,*::before,*::after{animation-duration:.001ms!important;animation-iteration-count:1!important;transition:none!important}
}
</style>
"""


def inject() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def esc(s) -> str:
    return html.escape(str(s))


# ---------------------------------------------------------------------------------------
# headers and layout
# ---------------------------------------------------------------------------------------
def hero(eyebrow: str, title: str, text: str, ico: str = "audio-waveform") -> None:
    st.markdown(f"<div class='vg-hdr'><div class='vg-eyebrow'>{icon(ico, 14)}{esc(eyebrow)}</div>"
                f"<h1>{esc(title)}</h1><p>{esc(text)}</p></div>", unsafe_allow_html=True)


page_header = hero


def section(title: str, sub: str = "", ico: str = "activity", meta: str = "", big: bool = False) -> None:
    st.markdown(section_html(title, sub, ico, meta, big), unsafe_allow_html=True)


def section_html(title: str, sub: str = "", ico: str = "activity", meta: str = "", big: bool = False) -> str:
    return (f"<div class='vg-sec{' big' if big else ''}'><div class='t'><div class='ib'>{icon(ico, 17)}</div><div>"
            f"<h3>{esc(title)}</h3>" + (f"<div class='s'>{esc(sub)}</div>" if sub else "") + "</div></div>"
            + (f"<div class='m'>{esc(meta)}</div>" if meta else "") + "</div>")


def stat(value: str, key: str, sub: str = "", color: str | None = None, ico: str | None = None) -> str:
    style = f" style='color:{color}'" if color else ""
    ic = f"<span style='color:{color or CYAN}'>{icon(ico, 15)}</span>" if ico else ""
    return (f"<div class='vg-stat'><div class='k'>{ic}{esc(key)}</div><div class='v'{style}>{esc(value)}</div>"
            + (f"<div class='s'>{esc(sub)}</div>" if sub else "") + "</div>")


def card(title: str, text: str, ico: str | None = None) -> str:
    ib = f"<div class='ib'>{icon(ico, 17)}</div>" if ico else ""
    return f"<div class='vg-card'>{ib}<h4>{esc(title)}</h4><p>{esc(text)}</p></div>"


def row(htmls: list[str], gap: str = "medium") -> None:
    for c, h in zip(st.columns(len(htmls), gap=gap), htmls):
        with c:
            st.markdown(h, unsafe_allow_html=True)


def bar(label: str, pct: float, color: str) -> str:
    """Labelled percentage bar (used by the trust panel and the downloadable report)."""
    w = max(0.0, min(pct, 100.0))
    return (f"<div style='font-size:.85rem;display:flex;justify-content:space-between;gap:12px'><span>{esc(label)}"
            f"</span><b>{pct:.0f}%</b></div><div class='adfd-bar' style='--c:{color};--w:{w:.1f}%'>"
            f"<div style='width:{w:.1f}%'></div></div>")


def chip(text: str, kind: str = "mute", ico: str | None = None) -> str:
    return f"<span class='vg-chip {kind}'>{icon(ico, 12) if ico else ''}{esc(text)}</span>"


def verdict_chip(verdict: str | None) -> str:
    if verdict is None:
        return chip("UNAVAILABLE", "mute", "info")
    return chip(VERDICT_WORD.get(verdict, verdict.upper()), "syn" if verdict == "Fake" else "real",
                "alert" if verdict == "Fake" else "check")



def note(text: str, ico: str = "info") -> None:
    st.markdown(f"<div class='vg-note'><span class='ib'>{icon(ico, 16)}</span><div>{text}</div></div>",
                unsafe_allow_html=True)


def warn_html(title: str, text: str, ico: str = "alert") -> str:
    return (f"<div class='vg-warn'><span class='ib'>{icon(ico, 18)}</span><div><b>{esc(title)}</b>"
            f"<p>{text}</p></div></div>")


def empty(title: str, text: str, ico: str = "audio-waveform") -> None:
    st.markdown(f"<div class='vg-empty'><div class='ib'>{icon(ico, 22)}</div><h4>{esc(title)}</h4>"
                f"<p>{esc(text)}</p></div>", unsafe_allow_html=True)


STEPS = ("Upload audio", "Preview waveform", "Analyze", "View results", "Explore analysis")


def stepper_html(active: int, labels=STEPS) -> str:
    """Workflow indicator; steps before `active` (1-based) show as done."""
    out = []
    for i, lb in enumerate(labels, 1):
        cls = "done" if i < active else "now" if i == active else ""
        n = icon("check", 12, stroke=3) if i < active else str(i)
        cur = " aria-current='step'" if i == active else ""
        out.append(f"<div class='vg-step {cls}' role='listitem'{cur}><span class='n'>{n}</span>"
                   f"<span class='lb'>{esc(lb)}</span></div>")
    return "<div class='vg-steps' role='list' aria-label='Analysis steps'>" + "".join(out) + "</div>"


def stepper(active: int, labels=STEPS) -> None:
    st.markdown(stepper_html(active, labels), unsafe_allow_html=True)


def ring(pct: float, color: str, size: int = 120, label: str = "", stroke: int = 9, small: bool = False) -> str:
    """Circular score indicator for a 0-100 value, drawn as an SVG arc that sweeps in."""
    r = (size - stroke) / 2
    circ = 2 * 3.14159265 * r
    off = circ * (1 - max(0.0, min(pct, 100.0)) / 100)
    return (f"<div class='vg-ring{' sm' if small else ''}' style='width:{size}px;height:{size}px'>"
            f"<svg width='{size}' height='{size}' viewBox='0 0 {size} {size}' aria-hidden='true'>"
            f"<circle cx='{size / 2}' cy='{size / 2}' r='{r:.1f}' fill='none' stroke='{LINE}' stroke-width='{stroke}'/>"
            f"<circle class='arc' cx='{size / 2}' cy='{size / 2}' r='{r:.1f}' fill='none' stroke='{color}' "
            f"stroke-width='{stroke}' stroke-linecap='round' stroke-dasharray='{circ:.1f}' "
            f"stroke-dashoffset='{off:.1f}' style='--circ:{circ:.1f}'/></svg>"
            f"<div class='lbl'><b>{pct:.0f}%</b>" + (f"<span>{esc(label)}</span>" if label else "") + "</div></div>")


def skeleton(h: int = 120) -> str:
    return f"<div class='vg-skel' style='--h:{h}px'></div>"


def loader_html(title: str, text: str) -> str:
    bars = "".join(f"<i style='--d:{(i % 7) * 0.09 + (i // 7) * 0.05:.2f}s'></i>" for i in range(22))
    return (f"<div class='vg-loader' role='status'><div class='vg-bars' aria-hidden='true'>{bars}</div><div>"
            f"<div class='t' style='display:flex;align-items:center;gap:10px'><span class='vg-pulse'></span>"
            f"{esc(title)}</div><div class='s'>{esc(text)}</div></div></div>")


def results_skeleton() -> str:
    return (skeleton(150) + "<div class='vg-skel-row'>" + skeleton(190) * 3 + "</div>"
            + "<div style='margin-top:14px'>" + skeleton(260) + "</div>")


def fmt_size(n: int) -> str:
    return f"{n / 2**20:.1f} MB" if n >= 2**20 else f"{n / 1024:.0f} KB"


def fmt_dur(s: float) -> str:
    m, sec = divmod(max(s, 0.0), 60)
    return f"{int(m)}:{sec:04.1f}" if m else f"{sec:.1f} s"


def file_card(name: str, meta: list[tuple[str, str]], tone: str = "", ico: str = "file-audio",
              tag: str = "") -> str:
    items = "".join(f"<span>{esc(k)} <b>{esc(v)}</b></span>" for k, v in meta)
    tg = f"<div style='margin-left:auto'>{tag}</div>" if tag else ""
    return (f"<div class='vg-file {tone}'><div class='ib'>{icon(ico, 20)}</div><div class='body'>"
            f"<div class='nm' title='{esc(name)}'>{esc(name)}</div><div class='meta'>{items}</div></div>{tg}</div>")

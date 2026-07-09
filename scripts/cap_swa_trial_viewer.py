"""
CAP-SWA Trial Viewer — interactive whole-night browser with clickable trials.

Companion to scripts/session_viewer.py (runs on a different port so both can be up).
Everything is driven by the pre-computed trial table, so it starts instantly:
    reports/slow_wave/cap_swa/trials/trial_epochs.parquet   (per-epoch, all sessions)
    reports/slow_wave/cap_swa/trials/trials.csv             (per-trial inventory)

What you get
  • Whole-night overview per session: hypnogram + capacitive channel means (CLE/CRE/CH)
    + the two hold criteria (C1 single-channel slow-drift, C3 slow thorax) + heart rate
    + EEG delta, with every CAP-SWA trial shaded green and a clickable onset marker.
  • Click a trial marker (or pick from the dropdown) → its pre/trial/post detail pops
    into a panel below; the whole-night view stays visible above.
  • PNG export: the whole-night plot auto-saves on each session load; "Export ALL"
    writes every session's whole-night PNG and every trial-window PNG. (PNG via
    matplotlib since kaleido isn't installed; the live views stay interactive.)

Run:
    .venv/Scripts/python.exe scripts/cap_swa_trial_viewer.py            # server on :8051
    .venv/Scripts/python.exe scripts/cap_swa_trial_viewer.py --export-all   # headless save, no server
Exports -> reports/slow_wave/cap_swa/trials/viewer_exports/
"""

import sys
import os
import argparse
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output, State, ctx, no_update

from sleep_monitor.config import STAGE_LABELS, STAGE_COLORS, STAGE_ORDER

ROOT = Path(__file__).resolve().parents[1]
TRIALS_DIR = ROOT / 'reports' / 'slow_wave' / 'cap_swa' / 'trials'
EXPORT_DIR = TRIALS_DIR / 'viewer_exports'
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

EPOCH_SEC = 30.0
N3_CODE = 1
Q_HOLD = 0.50
PAD_EPOCHS = 10        # 5 min pre/post context in the detail view

# hypnogram heatmap colorscale (discrete per stage code 0..4)
STAGE_CSCALE = []
for _i in range(5):
    STAGE_CSCALE.append([_i / 5.0, STAGE_COLORS[_i]])
    STAGE_CSCALE.append([(_i + 1) / 5.0, STAGE_COLORS[_i]])

# ── Data (loaded once) ───────────────────────────────────────────────────────

EPOCHS = pd.read_parquet(TRIALS_DIR / 'trial_epochs.parquet')
TRIALS = pd.read_csv(TRIALS_DIR / 'trials.csv')
SESSIONS = sorted(EPOCHS['session'].unique())


def sess_epochs(sess):
    return EPOCHS[EPOCHS['session'] == sess].sort_values('t_hr').reset_index(drop=True)


def sess_trials(sess):
    return TRIALS[TRIALS['session'] == sess].sort_values('t_start_hr').reset_index(drop=True)


def trial_span_hr(g, trial_id):
    """(onset_hr, offset_hr) for a trial id within a session epoch frame g."""
    idx = np.where(g['trial_id'].values == trial_id)[0]
    a, b = idx[0], idx[-1]
    return float(g['t_hr'].iloc[a]), float(g['t_hr'].iloc[b] + EPOCH_SEC / 3600.0)


# ── Whole-night interactive figure ───────────────────────────────────────────

def wholenight_fig(sess):
    g = sess_epochs(sess)
    ts = sess_trials(sess)
    t = g['t_hr'].values

    titles = ['Hypnogram + trials (click a marker)',
              'Capacitive channel means (CLE/CRE/CH)',
              'Hold criteria  C1 slow-drift / C3 slow-thorax',
              'Heart rate (BPM)', 'EEG delta ratio']
    heights = [0.13, 0.24, 0.22, 0.21, 0.20]
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, row_heights=heights,
                        vertical_spacing=0.03, subplot_titles=titles)

    # 1 hypnogram ribbon
    codes = g['stage_code'].astype(float).values.copy()
    codes[codes < 0] = np.nan
    fig.add_trace(go.Heatmap(
        z=codes.reshape(1, -1), x=t, y=[''], colorscale=STAGE_CSCALE,
        zmin=-0.5, zmax=4.5, showscale=False,
        text=[[STAGE_LABELS.get(int(c), '?') if np.isfinite(c) else '?' for c in codes]],
        hovertemplate='%{x:.2f} hr  %{text}<extra></extra>'), row=1, col=1)
    # clickable trial-onset markers
    if len(ts):
        on = ts['t_start_hr'].values
        fig.add_trace(go.Scatter(
            x=on, y=[0.5] * len(on), mode='markers',
            marker=dict(symbol='triangle-down', size=13, color='#111',
                        line=dict(width=1, color='#2ECC71')),
            customdata=ts['trial_id'].values,
            hovertemplate=('trial %{customdata}<br>%{x:.2f} hr<br>'
                           'click to open<extra></extra>'),
            name='trials', showlegend=False), row=1, col=1)
    for code in STAGE_ORDER:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
            marker=dict(color=STAGE_COLORS[code], size=9, symbol='square'),
            name=STAGE_LABELS[code], legendgroup='stages'), row=1, col=1)
    fig.update_yaxes(showticklabels=False, row=1, col=1)

    # 2 capacitive means
    for ch, c in [('cle_mean', '#2980B9'), ('cre_mean', '#C0392B'), ('ch_mean', '#16A085')]:
        fig.add_trace(go.Scattergl(x=t, y=g[ch], mode='lines', name=ch.split('_')[0].upper(),
            line=dict(color=c, width=1), legendgroup='cap'), row=2, col=1)
    # 3 criteria
    fig.add_trace(go.Scattergl(x=t, y=g['c1_slow_drift'], mode='lines', name='C1 slow-drift',
        line=dict(color='#2980B9', width=1), legendgroup='crit'), row=3, col=1)
    fig.add_trace(go.Scattergl(x=t, y=g['c3_slow_thorax'], mode='lines', name='C3 slow-thorax',
        line=dict(color='#E67E22', width=1), legendgroup='crit'), row=3, col=1)
    fig.add_hline(y=Q_HOLD, line=dict(color='#888', dash='dot', width=1), row=3, col=1)
    fig.update_yaxes(range=[0, 1.02], row=3, col=1)
    # 4 HR
    fig.add_trace(go.Scattergl(x=t, y=g['ecg_hr_hz'] * 60, mode='lines', name='HR',
        line=dict(color='#C0392B', width=1), showlegend=False), row=4, col=1)
    # 5 EEG delta
    fig.add_trace(go.Scattergl(x=t, y=g['eeg_delta_ratio'], mode='lines', name='EEG delta',
        line=dict(color='#8E44AD', width=1), showlegend=False), row=5, col=1)

    # shade every trial across all rows
    for _, tr in ts.iterrows():
        x0, x1 = trial_span_hr(g, tr['trial_id'])
        for r in range(1, 6):
            fig.add_vrect(x0=x0, x1=x1, fillcolor='#2ECC71', opacity=0.16,
                          line_width=0, row=r, col=1)

    fig.update_xaxes(title_text='Time (hr)', row=5, col=1)
    fig.update_layout(
        height=760, margin=dict(l=60, r=15, t=60, b=40),
        title=dict(text=f'{sess} — {len(ts)} CAP-SWA trials '
                        f'(green = trial · click a marker to open its window)', font=dict(size=15)),
        legend=dict(orientation='h', y=1.04, x=0, font=dict(size=10)),
        plot_bgcolor='#ffffff', hovermode='closest')
    return fig


# ── Trial detail interactive figure ──────────────────────────────────────────

def trial_detail_fig(sess, trial_id):
    g = sess_epochs(sess)
    idx = np.where(g['trial_id'].values == trial_id)[0]
    if len(idx) == 0:
        return go.Figure()
    a, b = idx[0], idx[-1] + 1
    lo, hi = max(0, a - PAD_EPOCHS), min(len(g), b + PAD_EPOCHS)
    w = g.iloc[lo:hi]
    onset = g['t_hr'].iloc[a]
    x = (w['t_hr'].values - onset) * 60.0
    x_off = (g['t_hr'].iloc[b - 1] - onset) * 60.0 + EPOCH_SEC / 60.0
    move_x = (w.loc[w['head_move'], 't_hr'].values - onset) * 60.0
    tr = sess_trials(sess).set_index('trial_id').loc[trial_id]

    titles = ['Stage', 'Criteria C1/C3', 'Single-ch means (demeaned)',
              'HR (BPM)', 'EEG delta', 'Thorax / accel']
    fig = make_subplots(rows=6, cols=1, shared_xaxes=True,
                        row_heights=[0.1, 0.18, 0.2, 0.18, 0.16, 0.18],
                        vertical_spacing=0.03, subplot_titles=titles)
    # stage strip
    codes = w['stage_code'].astype(float).values.copy(); codes[codes < 0] = np.nan
    fig.add_trace(go.Heatmap(z=codes.reshape(1, -1), x=x, y=[''], colorscale=STAGE_CSCALE,
        zmin=-0.5, zmax=4.5, showscale=False,
        text=[[STAGE_LABELS.get(int(c), '?') if np.isfinite(c) else '?' for c in codes]],
        hovertemplate='%{text}<extra></extra>'), row=1, col=1)
    fig.update_yaxes(showticklabels=False, row=1, col=1)
    # criteria
    fig.add_trace(go.Scatter(x=x, y=w['c1_slow_drift'], name='C1', mode='lines',
        line=dict(color='#2980B9')), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=w['c3_slow_thorax'], name='C3', mode='lines',
        line=dict(color='#E67E22')), row=2, col=1)
    fig.add_hline(y=Q_HOLD, line=dict(color='#888', dash='dot', width=1), row=2, col=1)
    fig.update_yaxes(range=[0, 1.02], row=2, col=1)
    # single-ch means demeaned
    for ch, c in [('cle_mean', '#2980B9'), ('cre_mean', '#C0392B'), ('ch_mean', '#16A085')]:
        v = w[ch].values.astype(float)
        fig.add_trace(go.Scatter(x=x, y=v - np.nanmean(v), name=ch.split('_')[0].upper(),
            mode='lines', line=dict(color=c)), row=3, col=1)
    # HR
    fig.add_trace(go.Scatter(x=x, y=w['ecg_hr_hz'] * 60, mode='lines', showlegend=False,
        line=dict(color='#C0392B')), row=4, col=1)
    # EEG delta
    fig.add_trace(go.Scatter(x=x, y=w['eeg_delta_ratio'], mode='lines', showlegend=False,
        line=dict(color='#8E44AD')), row=5, col=1)
    # thorax + accel
    fig.add_trace(go.Scatter(x=x, y=w['thorax_rms'], name='thorax', mode='lines',
        line=dict(color='#27AE60')), row=6, col=1)
    fig.add_trace(go.Scatter(x=x, y=w['acc_rms'], name='accel', mode='lines',
        line=dict(color='#7F8C8D')), row=6, col=1)

    # shade trial + movement markers on all rows
    for r in range(1, 7):
        fig.add_vrect(x0=0.0, x1=x_off, fillcolor='#2ECC71', opacity=0.15,
                      line_width=0, row=r, col=1)
        for mx in move_x:
            fig.add_vline(x=mx, line=dict(color='#E67E22', width=1, dash='dot'),
                          row=r, col=1)
    fig.update_xaxes(title_text='minutes from trial onset', row=6, col=1)
    fig.update_layout(height=720, margin=dict(l=60, r=15, t=55, b=40),
        title=dict(text=f'{sess} · trial {int(trial_id)} — {tr["duration_min"]:.1f} min · '
                        f'dom {tr["dom_stage"]} · N3 {tr["frac_N3"]*100:.0f}% '
                        f'(orange dotted = head movement)', font=dict(size=14)),
        legend=dict(orientation='h', y=1.05, x=0, font=dict(size=9)),
        plot_bgcolor='#ffffff', hovermode='x unified')
    return fig


# ── Matplotlib PNG exporters (kaleido-free) ──────────────────────────────────

def save_wholenight_png(sess):
    g = sess_epochs(sess); ts = sess_trials(sess); t = g['t_hr'].values
    fig, ax = plt.subplots(5, 1, figsize=(19, 10), sharex=True)
    fig.suptitle(f'{sess} — whole night with {len(ts)} CAP-SWA trials (green)',
                 fontsize=13, fontweight='bold')
    trial_spans = [trial_span_hr(g, tid) for tid in ts['trial_id']]
    for a in ax:
        for x0, x1 in trial_spans:
            a.axvspan(x0, x1, color='#2ECC71', alpha=0.18)
    for _, r in g.iterrows():
        ax[0].axvspan(r['t_hr'], r['t_hr'] + EPOCH_SEC / 3600,
                      color=STAGE_COLORS.get(int(r['stage_code']), '#AAA'),
                      alpha=0.9 if r['stage_code'] == N3_CODE else 0.4)
    ax[0].set_yticks([]); ax[0].set_ylabel('stage')
    for x0, _ in trial_spans:
        ax[0].axvline(x0, color='#111', lw=0.6)
    for ch, c in [('cle_mean', '#2980B9'), ('cre_mean', '#C0392B'), ('ch_mean', '#16A085')]:
        ax[1].plot(t, g[ch], lw=0.6, color=c, label=ch.split('_')[0].upper())
    ax[1].legend(fontsize=7, ncol=3); ax[1].set_ylabel('CAP means')
    ax[2].plot(t, g['c1_slow_drift'], lw=0.7, color='#2980B9', label='C1 slow-drift')
    ax[2].plot(t, g['c3_slow_thorax'], lw=0.7, color='#E67E22', label='C3 slow-thorax')
    ax[2].axhline(Q_HOLD, color='#888', ls=':'); ax[2].legend(fontsize=7, ncol=2)
    ax[2].set_ylabel('criteria'); ax[2].set_ylim(0, 1.02)
    ax[3].plot(t, g['ecg_hr_hz'] * 60, lw=0.7, color='#C0392B'); ax[3].set_ylabel('HR (BPM)')
    ax[4].plot(t, g['eeg_delta_ratio'], lw=0.7, color='#8E44AD')
    ax[4].set_ylabel('EEG delta'); ax[4].set_xlabel('Time (hr)')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = EXPORT_DIR / f'{sess}_wholenight.png'
    fig.savefig(out, dpi=110, bbox_inches='tight', facecolor='white'); plt.close(fig)
    return out


def save_trial_png(sess, trial_id):
    g = sess_epochs(sess)
    idx = np.where(g['trial_id'].values == trial_id)[0]
    a, b = idx[0], idx[-1] + 1
    lo, hi = max(0, a - PAD_EPOCHS), min(len(g), b + PAD_EPOCHS)
    w = g.iloc[lo:hi]; onset = g['t_hr'].iloc[a]
    x = (w['t_hr'].values - onset) * 60.0
    x_off = (g['t_hr'].iloc[b - 1] - onset) * 60.0 + EPOCH_SEC / 60.0
    move_x = (w.loc[w['head_move'], 't_hr'].values - onset) * 60.0
    tr = sess_trials(sess).set_index('trial_id').loc[trial_id]
    fig, ax = plt.subplots(6, 1, figsize=(9, 12), sharex=True)
    fig.suptitle(f'{sess} · trial {int(trial_id)} — {tr["duration_min"]:.1f} min · '
                 f'dom {tr["dom_stage"]} · N3 {tr["frac_N3"]*100:.0f}%', fontsize=11)
    def shade(a):
        a.axvspan(0, x_off, color='#2ECC71', alpha=0.15)
        for mx in move_x:
            a.axvline(mx, color='#E67E22', lw=0.8, ls=':')
    for _, r in w.iterrows():
        ax[0].axvspan((r['t_hr'] - onset) * 60, (r['t_hr'] - onset) * 60 + EPOCH_SEC / 60,
                      color=STAGE_COLORS.get(int(r['stage_code']), '#AAA'),
                      alpha=0.9 if r['stage_code'] == N3_CODE else 0.4)
    for mx in move_x:
        ax[0].axvline(mx, color='#E67E22', lw=1.2)
    ax[0].set_yticks([]); ax[0].set_ylabel('stage')
    shade(ax[1]); ax[1].plot(x, w['c1_slow_drift'], color='#2980B9', label='C1')
    ax[1].plot(x, w['c3_slow_thorax'], color='#E67E22', label='C3')
    ax[1].axhline(Q_HOLD, color='#888', ls=':'); ax[1].legend(fontsize=7); ax[1].set_ylabel('criteria')
    shade(ax[2])
    for ch, c in [('cle_mean', '#2980B9'), ('cre_mean', '#C0392B'), ('ch_mean', '#16A085')]:
        v = w[ch].values.astype(float); ax[2].plot(x, v - np.nanmean(v), color=c,
            label=ch.split('_')[0].upper())
    ax[2].legend(fontsize=7, ncol=3); ax[2].set_ylabel('CAP demeaned')
    shade(ax[3]); ax[3].plot(x, w['ecg_hr_hz'] * 60, color='#C0392B'); ax[3].set_ylabel('HR')
    shade(ax[4]); ax[4].plot(x, w['eeg_delta_ratio'], color='#8E44AD'); ax[4].set_ylabel('EEG delta')
    shade(ax[5]); ax[5].plot(x, w['thorax_rms'], color='#27AE60', label='thorax')
    ax5b = ax[5].twinx(); ax5b.plot(x, w['acc_rms'], color='#7F8C8D', alpha=0.8)
    ax[5].set_ylabel('thorax/accel'); ax[5].set_xlabel('minutes from trial onset')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    sd = EXPORT_DIR / sess; sd.mkdir(exist_ok=True)
    out = sd / f'trial_{int(trial_id):03d}.png'
    fig.savefig(out, dpi=110, bbox_inches='tight', facecolor='white'); plt.close(fig)
    return out


def export_all():
    n_w, n_t = 0, 0
    for sess in SESSIONS:
        save_wholenight_png(sess); n_w += 1
        for tid in sess_trials(sess)['trial_id']:
            save_trial_png(sess, tid); n_t += 1
        print(f'  {sess}: whole-night + {sess_trials(sess).shape[0]} trial windows')
    print(f'exported {n_w} whole-night + {n_t} trial-window PNGs -> {EXPORT_DIR}')


# ── Dash app ─────────────────────────────────────────────────────────────────

app = Dash(__name__)
app.title = 'CAP-SWA Trial Viewer'

app.layout = html.Div([
    html.H2('CAP-SWA Trial Viewer', style={'margin': '10px 15px', 'fontFamily': 'sans-serif'}),
    html.Div([
        html.Div([html.Label('Session', style={'fontWeight': 'bold', 'fontSize': '13px'}),
                  dcc.Dropdown(id='sess', options=[{'label': s, 'value': s} for s in SESSIONS],
                               value=SESSIONS[0], clearable=False, style={'width': '140px'})]),
        html.Div([html.Label('Trial', style={'fontWeight': 'bold', 'fontSize': '13px'}),
                  dcc.Dropdown(id='trial', options=[], placeholder='click a marker or pick',
                               style={'width': '360px'})]),
        html.Div([html.Label(' ', style={'display': 'block', 'fontSize': '13px'}),
                  html.Button('Export ALL (whole-night + trial PNGs)', id='export-btn',
                              n_clicks=0, style={'fontSize': '13px', 'padding': '6px 10px'})]),
        html.Div(id='status', style={'fontSize': '12px', 'color': '#16A085',
                                     'alignSelf': 'center', 'marginLeft': '10px'}),
    ], style={'display': 'flex', 'gap': '16px', 'padding': '5px 15px 10px',
              'fontFamily': 'sans-serif', 'flexWrap': 'wrap', 'alignItems': 'flex-start'}),
    dcc.Loading(dcc.Graph(id='whole', config={'scrollZoom': True, 'displaylogo': False})),
    html.Hr(),
    dcc.Loading(dcc.Graph(id='detail', config={'displaylogo': False})),
], style={'backgroundColor': '#fafafa'})


@app.callback(
    Output('whole', 'figure'), Output('trial', 'options'), Output('trial', 'value'),
    Output('status', 'children'),
    Input('sess', 'value'))
def on_session(sess):
    fig = wholenight_fig(sess)
    ts = sess_trials(sess)
    opts = [{'label': f"trial {int(r.trial_id)} · {r.t_start_hr:.2f} hr · "
                      f"{r.duration_min:.1f} min · {r.dom_stage} · N3 {r.frac_N3*100:.0f}%",
             'value': int(r.trial_id)} for r in ts.itertuples()]
    out = save_wholenight_png(sess)   # auto-save whole-night on load
    return fig, opts, None, f'saved {out.name}'


@app.callback(
    Output('detail', 'figure'), Output('trial', 'value', allow_duplicate=True),
    Input('whole', 'clickData'), Input('trial', 'value'), State('sess', 'value'),
    prevent_initial_call=True)
def on_trial(click, trial_val, sess):
    trig = ctx.triggered_id
    tid = None
    if trig == 'whole' and click and click.get('points'):
        cd = click['points'][0].get('customdata')
        if cd is not None:
            tid = int(cd)
    elif trig == 'trial' and trial_val is not None:
        tid = int(trial_val)
    if tid is None:
        return no_update, no_update
    return trial_detail_fig(sess, tid), tid


@app.callback(Output('status', 'children', allow_duplicate=True),
              Input('export-btn', 'n_clicks'), prevent_initial_call=True)
def on_export(n):
    if not n:
        return no_update
    n_w, n_t = 0, 0
    for sess in SESSIONS:
        save_wholenight_png(sess); n_w += 1
        for tid in sess_trials(sess)['trial_id']:
            save_trial_png(sess, tid); n_t += 1
    return f'exported {n_w} whole-night + {n_t} trial PNGs -> {EXPORT_DIR}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--export-all', action='store_true',
                    help='save all whole-night + trial PNGs and exit (no server)')
    ap.add_argument('--port', type=int, default=8051)
    args = ap.parse_args()
    if args.export_all:
        export_all()
        return
    print(f'\nCAP-SWA Trial Viewer at http://localhost:{args.port}\n'
          f'  {len(SESSIONS)} sessions, {len(TRIALS)} trials total\n')
    app.run(debug=True, port=args.port)


if __name__ == '__main__':
    main()

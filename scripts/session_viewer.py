"""
Interactive Sleep Session Viewer — Plotly Dash app.

Usage:
    python scripts/session_viewer.py
    → opens browser at http://localhost:8050
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scipy.signal import spectrogram as scipy_spectrogram
from scipy.signal import butter, sosfiltfilt

from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sleep_monitor import (
    load_session, SESSION_META, FS,
    PSG_CHANNELS,
    EEG_BANDS, BAND_COLORS,
    DELTA_SUB_BANDS, DELTA_SUB_COLORS,
    STAGE_LABELS, STAGE_COLORS, STAGE_ORDER,
    APNEA_LABELS, APNEA_COLORS,
)
from sleep_monitor.loader import load_sleep_profile, load_apnea_events
from sleep_monitor.spectral import compute_band_power_ratios
from sleep_monitor.motion import epoch_motion, classify_position
from sleep_monitor.ground_truth import gt_sliding_rates

# ── Constants ────────────────────────────────────────────────────────────────

CAP_CHS = ['CH', 'CLE', 'CRE']
CAP_SET = {'CH', 'CLE', 'CRE', 'aX', 'aY', 'aZ'}
SPEC_FMAX = 5.0
PSG_SPEC_FMAX = 30.0
PSG_SET = set(PSG_CHANNELS)
SPEC_NPERSEG_SEC = 10.0
SPEC_NOVERLAP_SEC = 5.0
SMOOTH_PTS = 12

STAGE_CSCALE = []
for _i in range(5):
    _lo = _i / 5.0
    _hi = (_i + 1) / 5.0
    STAGE_CSCALE.append([_lo, STAGE_COLORS[_i]])
    STAGE_CSCALE.append([_hi, STAGE_COLORS[_i]])

ROW_H = {
    'hypnogram': 80,
    'apnea': 60,
    'spectrogram': 220,
    'band_power': 220,
    'ridges': 220,
    'feature': 160,
}

PCA_COLORS = ['#E74C3C', '#3498DB', '#2ECC71']

ALL_RAW_CHS = ['CH', 'CLE', 'CRE'] + PSG_CHANNELS

RAW_COLORS = {
    'CH': '#2980B9', 'CLE': '#27AE60', 'CRE': '#8E44AD',
    'aX': '#E74C3C', 'aY': '#F39C12', 'aZ': '#1ABC9C',
    'EEG': '#2C3E50', 'EOGl': '#16A085', 'EOGr': '#27AE60',
    'ECG': '#C0392B', 'Flow': '#2980B9', 'Pleth': '#8E44AD',
    'Thorax': '#D35400', 'Abdomen': '#7D3C98',
}

CUSTOM_COLORS = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12',
                 '#9B59B6', '#1ABC9C', '#E67E22', '#C0392B']

# ── Helpers ──────────────────────────────────────────────────────────────────


def _compute_spectrogram(sig, fs, fmax=SPEC_FMAX):
    nperseg = int(SPEC_NPERSEG_SEC * fs)
    noverlap = int(SPEC_NOVERLAP_SEC * fs)
    f, t, Sxx = scipy_spectrogram(
        sig.astype(np.float64), fs=fs,
        nperseg=nperseg, noverlap=noverlap, window='hann')
    mask = f <= fmax
    return t / 3600.0, f[mask], 10.0 * np.log10(Sxx[mask] + 1e-20)


def _smooth(arr, n=SMOOTH_PTS):
    finite = np.isfinite(arr)
    x = np.where(finite, arr, 0.0)
    kernel = np.ones(n)
    sums = np.convolve(x, kernel, mode='full')[:len(arr)]
    counts = np.convolve(finite.astype(float), kernel, mode='full')[:len(arr)]
    out = np.where(counts > 0, sums / counts, np.nan)
    return out


def _parse_custom_bands(text):
    """Parse 'name:lo-hi, name2:lo2-hi2' into a bands dict."""
    if not text or not text.strip():
        return None
    bands = {}
    for part in text.split(','):
        part = part.strip()
        if ':' not in part or '-' not in part:
            continue
        name, freq_range = part.split(':', 1)
        try:
            lo_s, hi_s = freq_range.split('-', 1)
            lo, hi = float(lo_s), float(hi_s)
        except ValueError:
            continue
        if lo >= hi:
            continue
        bands[name.strip()] = (lo, hi)
    return bands if bands else None


def _parse_ridge_freqs(text):
    """Parse '0.5, 1.0, 1.5' or '0.5:2.5:0.25' into a sorted list of freqs."""
    if not text or not text.strip():
        return None
    freqs = []
    for part in text.split(','):
        part = part.strip()
        if ':' in part:
            pieces = part.split(':')
            if len(pieces) >= 2:
                try:
                    start, stop = float(pieces[0]), float(pieces[1])
                    step = float(pieces[2]) if len(pieces) >= 3 else 0.1
                    if step > 0 and stop > start:
                        freqs.extend(np.arange(start, stop + step / 2, step).tolist())
                except ValueError:
                    continue
        else:
            try:
                freqs.append(float(part))
            except ValueError:
                continue
    return sorted(set(round(f, 3) for f in freqs)) if freqs else None


def _ridge_colors(n):
    """Generate n perceptually-distinct colors via HSL rotation."""
    import colorsys
    colors = []
    for i in range(n):
        h = (i * 0.618033988) % 1.0
        r, g, b = colorsys.hls_to_rgb(h, 0.45, 0.85)
        colors.append(f'rgb({int(r*255)},{int(g*255)},{int(b*255)})')
    return colors


def _apply_filter(sig, lo, hi, fs):
    """Bandpass / highpass / lowpass via 4th-order Butterworth."""
    nyq = fs / 2.0
    if lo is not None and hi is not None:
        if 0 < lo < hi < nyq:
            sos = butter(4, [lo, hi], btype='band', fs=fs, output='sos')
        else:
            return sig
    elif lo is not None:
        if 0 < lo < nyq:
            sos = butter(4, lo, btype='high', fs=fs, output='sos')
        else:
            return sig
    elif hi is not None:
        if 0 < hi < nyq:
            sos = butter(4, hi, btype='low', fs=fs, output='sos')
        else:
            return sig
    else:
        return sig
    return sosfiltfilt(sos, sig)


MAX_RAW_DISPLAY_PTS = 15000
MAX_SPEC_TIME_PTS = 2000


def _downsample_spec(t_hr, freqs, Sxx_db):
    """Downsample spectrogram time axis and round for smaller JSON."""
    n_t = Sxx_db.shape[1]
    if n_t > MAX_SPEC_TIME_PTS:
        step = n_t // MAX_SPEC_TIME_PTS
        t_hr = t_hr[::step]
        Sxx_db = Sxx_db[:, ::step]
    return t_hr, freqs, np.around(Sxx_db, 1)


# ── Session cache ────────────────────────────────────────────────────────────

class SessionCache:
    def __init__(self, idx):
        self.idx = idx
        print(f'\n[viewer] Loading session {idx}...')
        self.session = load_session(idx)
        self.session.sleep_profile = load_sleep_profile(self.session)
        self.session.apnea_events = load_apnea_events(self.session)

        self.spectrograms = {}
        for ch in CAP_CHS:
            self.spectrograms[ch] = _compute_spectrogram(
                self.session.cap[ch], FS)

        acc = self.session.cap['acc_mag']
        self.band_powers = {}
        for ch in CAP_CHS:
            self.band_powers[ch] = {
                'eeg': compute_band_power_ratios(
                    self.session.cap[ch], fs=FS, acc_mag=acc),
                'delta': compute_band_power_ratios(
                    self.session.cap[ch], fs=FS, acc_mag=acc,
                    bands=DELTA_SUB_BANDS, total_range=(0.0, 4.0)),
            }

        self.motion = epoch_motion(self.session)
        self.positions = classify_position(
            self.motion['roll_deg'], self.motion['pitch_deg'])

        # PCA on stacked spectrograms (3 channels x n_freqs at each time step)
        specs = [self.spectrograms[ch][2] for ch in CAP_CHS]
        X = np.column_stack([s.T for s in specs])
        X_c = X - np.nanmean(X, axis=0)
        X_c = np.nan_to_num(X_c)
        U, S, _ = np.linalg.svd(X_c, full_matrices=False)
        self.pca_scores = U * S
        self.pca_var_ratio = S**2 / np.sum(S**2)
        self.pca_t_hr = self.spectrograms[CAP_CHS[0]][0]

        self.time_hr = self.session.time_hr

        try:
            self.gt_rates = gt_sliding_rates(self.session)
        except Exception as e:
            print(f'  [viewer] GT rates unavailable: {e}')
            self.gt_rates = None

        self._custom_band_cache = {}

        print(f'[viewer] Session {self.session.label} ready.')

    def get_spectrogram(self, ch):
        if ch not in self.spectrograms:
            src = self.session.psg if ch in self.session.psg else self.session.cap
            fmax = PSG_SPEC_FMAX if ch in PSG_SET else SPEC_FMAX
            self.spectrograms[ch] = _compute_spectrogram(src[ch], FS, fmax=fmax)
        return self.spectrograms[ch]

    def get_custom_band_power(self, band_ch, bands):
        """Cached custom band power — ratio denominator is full Nyquist range."""
        key = (band_ch, tuple(sorted(bands.items())))
        if key not in self._custom_band_cache:
            self._custom_band_cache[key] = compute_band_power_ratios(
                self.session.cap[band_ch], fs=FS,
                acc_mag=self.session.cap['acc_mag'],
                bands=bands, total_range=(0.0, FS / 2.0))
        return self._custom_band_cache[key]


_cache = None

# ── Dash app ─────────────────────────────────────────────────────────────────

app = Dash(__name__)
app.title = 'Sleep Session Viewer'

session_options = [
    {'label': f"{m['label']}  {m['subject']}-{m['initials']}  {m['date']}",
     'value': m['idx']}
    for m in SESSION_META
]

app.layout = html.Div([
    html.H2('Sleep Session Viewer',
            style={'margin': '10px 15px', 'fontFamily': 'sans-serif'}),

    html.Div([
        html.Div([
            html.Label('Session',
                       style={'fontWeight': 'bold', 'fontSize': '13px'}),
            dcc.Dropdown(id='session-dropdown', options=session_options,
                         value=0, style={'width': '300px'}),
        ]),
        html.Div([
            html.Label('Panels',
                       style={'fontWeight': 'bold', 'fontSize': '13px'}),
            dcc.Checklist(
                id='panel-toggles',
                options=[
                    {'label': ' Hypnogram', 'value': 'hypnogram'},
                    {'label': ' Apnea', 'value': 'apnea'},
                    {'label': ' Band power', 'value': 'band_power'},
                    {'label': ' Ridges', 'value': 'ridges'},
                ],
                value=['hypnogram', 'apnea', 'band_power'], inline=True,
                style={'fontSize': '13px'}),
        ]),
        html.Div([
            html.Label('Spectrogram channels',
                       style={'fontWeight': 'bold', 'fontSize': '13px'}),
            dcc.Checklist(
                id='spec-channels',
                options=[{'label': f' {ch}', 'value': ch}
                         for ch in CAP_CHS + PSG_CHANNELS],
                value=['CH', 'CLE', 'CRE'], inline=True,
                style={'fontSize': '13px'}),
        ]),
        html.Div([
            html.Label('Band channel',
                       style={'fontWeight': 'bold', 'fontSize': '13px'}),
            dcc.Dropdown(id='band-channel',
                         options=[{'label': ch, 'value': ch} for ch in CAP_CHS],
                         value='CH', clearable=False,
                         style={'width': '90px'}),
        ]),
        html.Div([
            html.Label('Band mode',
                       style={'fontWeight': 'bold', 'fontSize': '13px'}),
            dcc.RadioItems(
                id='band-mode',
                options=[{'label': ' Ratio', 'value': 'ratio'},
                         {'label': ' Absolute', 'value': 'absolute'}],
                value='ratio', inline=True, style={'fontSize': '13px'}),
        ]),
        html.Div([
            html.Label('Band set',
                       style={'fontWeight': 'bold', 'fontSize': '13px'}),
            dcc.RadioItems(
                id='band-set',
                options=[{'label': ' EEG', 'value': 'eeg'},
                         {'label': ' Delta sub', 'value': 'delta'},
                         {'label': ' Custom', 'value': 'custom'}],
                value='eeg', inline=True, style={'fontSize': '13px'}),
        ]),
        html.Div([
            html.Label('Custom bands (name:lo-hi, ...)',
                       style={'fontWeight': 'bold', 'fontSize': '13px'}),
            dcc.Input(id='custom-bands-input', type='text',
                      placeholder='resp:0.1-0.5, cardiac:0.8-2.0',
                      debounce=True,
                      style={'width': '280px', 'fontSize': '13px'}),
        ]),
        html.Div([
            html.Label('Ridge freqs (Hz)',
                       style={'fontWeight': 'bold', 'fontSize': '13px'}),
            dcc.Input(id='ridge-freqs-input', type='text',
                      placeholder='0.5, 1.0, 1.5 or 0.5:2.5:0.25',
                      debounce=True,
                      style={'width': '260px', 'fontSize': '13px'}),
        ]),
        html.Div([
            html.Label('Band LP (mHz)',
                       style={'fontWeight': 'bold', 'fontSize': '13px'}),
            dcc.Input(id='band-lp-cutoff', type='number',
                      placeholder='e.g. 5', debounce=True,
                      min=0.1, step=0.5,
                      style={'width': '80px', 'fontSize': '13px'}),
        ]),
        html.Div([
            html.Label('Spec norm',
                       style={'fontWeight': 'bold', 'fontSize': '13px'}),
            dcc.RadioItems(
                id='spec-norm',
                options=[{'label': ' Global', 'value': 'global'},
                         {'label': ' Per-column', 'value': 'per_col'}],
                value='global', inline=True, style={'fontSize': '13px'}),
        ]),
        html.Div([
            html.Label('Feature overlays',
                       style={'fontWeight': 'bold', 'fontSize': '13px'}),
            dcc.Checklist(
                id='feature-overlays',
                options=[
                    {'label': ' Resp rate', 'value': 'resp_rate'},
                    {'label': ' Cardiac rate', 'value': 'card_rate'},
                    {'label': ' Head position', 'value': 'head_pos'},
                    {'label': ' Motion', 'value': 'motion'},
                    {'label': ' PC1', 'value': 'pc_1'},
                    {'label': ' PC2', 'value': 'pc_2'},
                    {'label': ' PC3', 'value': 'pc_3'},
                ],
                value=[], inline=True, style={'fontSize': '13px'}),
        ]),
        html.Div([
            html.Label('CAP raw',
                       style={'fontWeight': 'bold', 'fontSize': '13px'}),
            dcc.Checklist(
                id='raw-cap',
                options=[{'label': f' {ch}', 'value': ch}
                         for ch in ['CH', 'CLE', 'CRE']],
                value=[], inline=True, style={'fontSize': '13px'}),
        ]),
        html.Div([
            html.Label('PSG raw',
                       style={'fontWeight': 'bold', 'fontSize': '13px'}),
            dcc.Checklist(
                id='raw-psg',
                options=[{'label': f' {ch}', 'value': ch}
                         for ch in PSG_CHANNELS],
                value=[], inline=True, style={'fontSize': '13px'}),
        ]),
        html.Div([
            dcc.Checklist(
                id='zscore-toggle',
                options=[{'label': ' Z-score raw signals', 'value': 'zscore'}],
                value=[], inline=True,
                style={'fontSize': '13px', 'marginTop': '18px'}),
        ]),
        html.Div([
            html.Label('Filter (Hz)',
                       style={'fontWeight': 'bold', 'fontSize': '13px'}),
            html.Div([
                dcc.Input(id='filter-lo', type='number', placeholder='Lo',
                          debounce=True, min=0, step=0.05,
                          style={'width': '70px', 'fontSize': '13px'}),
                html.Span(' – ',
                          style={'fontSize': '13px', 'margin': '0 4px'}),
                dcc.Input(id='filter-hi', type='number', placeholder='Hi',
                          debounce=True, min=0, step=0.05,
                          style={'width': '70px', 'fontSize': '13px'}),
            ], style={'display': 'flex', 'alignItems': 'center'}),
        ]),
    ], style={'display': 'flex', 'gap': '18px', 'flexWrap': 'wrap',
              'alignItems': 'flex-start', 'padding': '5px 15px 10px 15px',
              'fontFamily': 'sans-serif'}),

    dcc.Loading(
        children=[dcc.Graph(id='main-graph',
                            config={'scrollZoom': True,
                                    'displaylogo': False})],
        type='default',
    ),
], style={'backgroundColor': '#fafafa'})


# ── Figure builder ───────────────────────────────────────────────────────────

def _build_figure(cache, panels, spec_channels, band_ch, band_mode,
                  band_set, spec_norm, features, custom_bands_text,
                  raw_signals, do_zscore, filter_lo, filter_hi,
                  band_lp_mhz, ridge_freqs_text):
    session = cache.session
    sp = session.sleep_profile
    apnea = session.apnea_events

    show_hypno = 'hypnogram' in panels
    show_apnea = 'apnea' in panels
    show_bands = 'band_power' in panels
    show_ridges = 'ridges' in panels

    # Split PCA toggles from regular features
    pc_indices = sorted(
        int(f.split('_')[1]) - 1 for f in features if f.startswith('pc_'))
    other_features = [f for f in features if not f.startswith('pc_')]

    use_filter = filter_lo is not None or filter_hi is not None

    # ── Row layout ──────────────────────────────────────────────────────
    row_info = []
    if show_hypno:
        row_info.append(('Hypnogram', ROW_H['hypnogram']))
    if show_apnea:
        row_info.append(('Apnea events', ROW_H['apnea']))
    for ch in spec_channels:
        row_info.append((f'Spectrogram — {ch}', ROW_H['spectrogram']))
    if show_bands:
        row_info.append(('Band power', ROW_H['band_power']))
    if show_ridges:
        row_info.append((f'Spectral ridges — {band_ch}', ROW_H['ridges']))

    feat_labels = {
        'resp_rate': 'Resp rate (br/min)',
        'card_rate': 'Cardiac rate (BPM)',
        'head_pos': 'Head position (deg)',
        'motion': 'Motion RMS',
    }
    for f in other_features:
        row_info.append((feat_labels[f], ROW_H['feature']))
    if pc_indices:
        row_info.append(('PCA components', ROW_H['feature']))

    # Build raw signal row titles
    for ch in raw_signals:
        lbl = ch
        if use_filter:
            lo_s = f'{filter_lo}' if filter_lo is not None else ''
            hi_s = f'{filter_hi}' if filter_hi is not None else ''
            lbl += f' [{lo_s}–{hi_s} Hz]'
        if do_zscore:
            lbl += ' (z-scored)'
        row_info.append((lbl, ROW_H['feature']))

    if not row_info:
        row_info.append(('(no panels selected)', 80))

    n_rows = len(row_info)
    titles = [r[0] for r in row_info]
    heights = [r[1] for r in row_info]
    total_px = sum(heights) + 100

    fig = make_subplots(
        rows=n_rows, cols=1, shared_xaxes=True,
        row_heights=heights,
        vertical_spacing=max(0.005, 30.0 / total_px),
        subplot_titles=titles,
    )

    cur_row = 0

    # ── Hypnogram ────────────────────────────────────────────────────────
    if show_hypno:
        cur_row += 1
        if sp is not None:
            t_ep = sp['t_ep_hr']
            codes_f = sp['codes'].astype(float)
            codes_f[codes_f == -1] = np.nan
            fig.add_trace(go.Heatmap(
                z=codes_f.reshape(1, -1),
                x=t_ep.tolist(), y=[''],
                colorscale=STAGE_CSCALE,
                zmin=-0.5, zmax=4.5,
                showscale=False,
                hovertemplate='%{x:.2f} hr  %{text}<extra></extra>',
                text=[sp['labels']],
            ), row=cur_row, col=1)
            for code in STAGE_ORDER:
                fig.add_trace(go.Scatter(
                    x=[None], y=[None], mode='markers',
                    marker=dict(color=STAGE_COLORS[code], size=10,
                                symbol='square'),
                    name=STAGE_LABELS[code], legendgroup='stages',
                ), row=cur_row, col=1)
        fig.update_yaxes(showticklabels=False, row=cur_row, col=1)

    # ── Apnea events ────────────────────────────────────────────────────
    if show_apnea:
        cur_row += 1
        fig.update_yaxes(range=[0, 1], showticklabels=False,
                         row=cur_row, col=1)
        if apnea is not None and len(apnea['codes']) > 0:
            for s, e, code in zip(apnea['start_hr'], apnea['end_hr'],
                                  apnea['codes']):
                fig.add_shape(
                    type='rect', x0=float(s), x1=float(e), y0=0, y1=1,
                    fillcolor=APNEA_COLORS.get(int(code), '#E74C3C'),
                    opacity=0.8, line_width=0,
                    row=cur_row, col=1,
                )
            seen = set()
            for code in apnea['codes']:
                c = int(code)
                if c in seen or c == 0:
                    continue
                seen.add(c)
                fig.add_trace(go.Scatter(
                    x=[None], y=[None], mode='markers',
                    marker=dict(color=APNEA_COLORS[c], size=10,
                                symbol='square'),
                    name=APNEA_LABELS[c], legendgroup='apnea',
                ), row=cur_row, col=1)

    # ── Spectrograms ────────────────────────────────────────────────────
    for ch in spec_channels:
        cur_row += 1
        t_hr, freqs, Sxx_db = _downsample_spec(*cache.get_spectrogram(ch))

        if spec_norm == 'per_col':
            col_min = np.nanmin(Sxx_db, axis=0, keepdims=True)
            col_max = np.nanmax(Sxx_db, axis=0, keepdims=True)
            rng = col_max - col_min
            rng[rng < 1e-6] = 1.0
            z = np.around((Sxx_db - col_min) / rng, 2)
            zmin, zmax = 0.0, 1.0
        else:
            z = Sxx_db
            p2, p98 = np.nanpercentile(Sxx_db, [2, 98])
            zmin, zmax = float(p2), float(p98)

        fig.add_trace(go.Heatmap(
            z=z, x=t_hr.tolist(), y=freqs.tolist(),
            colorscale='Inferno', showscale=False,
            zmin=zmin, zmax=zmax,
            hovertemplate='%{x:.2f} hr  %{y:.2f} Hz  %{z:.1f}'
                          '<extra></extra>',
        ), row=cur_row, col=1)
        fig.update_yaxes(title_text='Hz', row=cur_row, col=1)

    # ── Band power ──────────────────────────────────────────────────────
    pct = 0.0
    if show_bands:
        cur_row += 1

        custom_band_warning = None
        if band_set == 'custom':
            custom_bands = _parse_custom_bands(custom_bands_text)
            if custom_bands:
                bands = custom_bands
                bp_data = cache.get_custom_band_power(band_ch, bands)
                colors = {name: CUSTOM_COLORS[i % len(CUSTOM_COLORS)]
                          for i, name in enumerate(bands)}
            else:
                bands = EEG_BANDS
                bp_data = cache.band_powers[band_ch]['eeg']
                colors = BAND_COLORS
                custom_band_warning = (
                    'Enter bands as "name:lo-hi, …" then press Enter')
        elif band_set == 'delta':
            bands = DELTA_SUB_BANDS
            bp_data = cache.band_powers[band_ch]['delta']
            colors = DELTA_SUB_COLORS
        else:
            bands = EEG_BANDS
            bp_data = cache.band_powers[band_ch]['eeg']
            colors = BAND_COLORS

        bp_t = bp_data['t_hr']
        bp_fs = 1.0 / ((bp_t[1] - bp_t[0]) * 3600.0) if len(bp_t) > 1 else None
        for name, (flo, fhi) in bands.items():
            key = name if band_mode == 'ratio' else f'{name}_abs'
            vals = _smooth(bp_data[key])
            if band_lp_mhz and bp_fs and band_lp_mhz / 1000.0 < bp_fs / 2.0:
                cutoff_hz = band_lp_mhz / 1000.0
                valid = np.isfinite(vals)
                if valid.sum() > 12:
                    interp = np.interp(np.arange(len(vals)),
                                       np.where(valid)[0], vals[valid])
                    sos = butter(2, cutoff_hz, btype='low', fs=bp_fs,
                                 output='sos')
                    interp = sosfiltfilt(sos, interp)
                    vals = np.where(valid, interp, np.nan)
            fig.add_trace(go.Scattergl(
                x=bp_t, y=vals, mode='lines',
                line=dict(color=colors[name], width=1.5),
                name=f'{name} ({flo:.1f}–{fhi:.1f} Hz)',
                legendgroup='bands',
            ), row=cur_row, col=1)

        pct = (bp_data['motion_mask'].sum()
               / len(bp_data['motion_mask']) * 100)
        is_custom = band_set == 'custom' and custom_band_warning is None
        if band_mode == 'ratio':
            y_range = None if is_custom else [0, 1]
            fig.update_yaxes(title_text=f'{band_ch} ratio', range=y_range,
                             row=cur_row, col=1)
        else:
            fig.update_yaxes(title_text=f'{band_ch} abs power (log)',
                             type='log', row=cur_row, col=1)

        if custom_band_warning:
            fig.add_annotation(
                text=custom_band_warning, showarrow=False,
                xref='x domain', yref='y domain', x=0.5, y=0.5,
                font=dict(size=12, color='#E74C3C'),
                row=cur_row, col=1)

    # ── Spectral ridges ────────────────────────────────────────────────
    if show_ridges:
        cur_row += 1
        target_freqs = _parse_ridge_freqs(ridge_freqs_text)
        if target_freqs:
            t_hr, freqs, Sxx_db = cache.get_spectrogram(band_ch)
            colors = _ridge_colors(len(target_freqs))
            for i, tf in enumerate(target_freqs):
                idx = np.argmin(np.abs(freqs - tf))
                actual_f = freqs[idx]
                trace = _smooth(Sxx_db[idx, :])
                if band_lp_mhz:
                    bp_fs = 1.0 / (SPEC_NOVERLAP_SEC)
                    cutoff_hz = band_lp_mhz / 1000.0
                    if cutoff_hz < bp_fs / 2.0:
                        valid = np.isfinite(trace)
                        if valid.sum() > 12:
                            interp = np.interp(
                                np.arange(len(trace)),
                                np.where(valid)[0], trace[valid])
                            sos = butter(2, cutoff_hz, btype='low',
                                         fs=bp_fs, output='sos')
                            interp = sosfiltfilt(sos, interp)
                            trace = np.where(valid, interp, np.nan)
                fig.add_trace(go.Scattergl(
                    x=t_hr, y=trace, mode='lines',
                    line=dict(color=colors[i], width=1.3),
                    name=f'{actual_f:.2f} Hz',
                    legendgroup='ridges',
                ), row=cur_row, col=1)
            fig.update_yaxes(title_text=f'{band_ch} power (dB)',
                             row=cur_row, col=1)
        else:
            fig.add_annotation(
                text='Enter frequencies: "0.5, 1.0, 1.5" or "0.5:2.5:0.25"',
                showarrow=False,
                xref='x domain', yref='y domain', x=0.5, y=0.5,
                font=dict(size=12, color='#E74C3C'),
                row=cur_row, col=1)

    # ── Feature rows ────────────────────────────────────────────────────
    for feat in other_features:
        cur_row += 1

        if feat == 'resp_rate':
            if cache.gt_rates is not None:
                fig.add_trace(go.Scattergl(
                    x=cache.gt_rates['t_hr'],
                    y=cache.gt_rates['resp_hz'] * 60.0,
                    mode='lines', name='Resp rate',
                    line=dict(color='#27AE60', width=1.2),
                    legendgroup='features',
                ), row=cur_row, col=1)
            fig.update_yaxes(title_text='br/min', row=cur_row, col=1)

        elif feat == 'card_rate':
            if cache.gt_rates is not None:
                fig.add_trace(go.Scattergl(
                    x=cache.gt_rates['t_hr'],
                    y=cache.gt_rates['card_hz'] * 60.0,
                    mode='lines', name='Cardiac rate',
                    line=dict(color='#E74C3C', width=1.2),
                    legendgroup='features',
                ), row=cur_row, col=1)
            fig.update_yaxes(title_text='BPM', row=cur_row, col=1)

        elif feat == 'head_pos':
            fig.add_trace(go.Scattergl(
                x=cache.motion['t_hr'], y=cache.motion['roll_deg'],
                mode='lines', name='Roll',
                line=dict(color='#3498DB', width=1),
                legendgroup='features',
            ), row=cur_row, col=1)
            fig.add_trace(go.Scattergl(
                x=cache.motion['t_hr'], y=cache.motion['pitch_deg'],
                mode='lines', name='Pitch',
                line=dict(color='#E67E22', width=1),
                legendgroup='features',
            ), row=cur_row, col=1)
            fig.update_yaxes(title_text='deg', row=cur_row, col=1)

        elif feat == 'motion':
            fig.add_trace(go.Scattergl(
                x=cache.motion['t_hr'], y=cache.motion['movement_rms'],
                mode='lines', name='Motion RMS',
                line=dict(color='#9B59B6', width=1),
                legendgroup='features',
            ), row=cur_row, col=1)
            fig.update_yaxes(title_text='accel', row=cur_row, col=1)

    # ── PCA components row ──────────────────────────────────────────────
    if pc_indices:
        cur_row += 1
        for pi in pc_indices:
            if pi < cache.pca_scores.shape[1]:
                var_pct = cache.pca_var_ratio[pi] * 100
                fig.add_trace(go.Scattergl(
                    x=cache.pca_t_hr,
                    y=cache.pca_scores[:, pi],
                    mode='lines',
                    name=f'PC{pi+1} ({var_pct:.1f}%)',
                    line=dict(color=PCA_COLORS[pi % len(PCA_COLORS)],
                              width=1.2),
                    legendgroup='pca',
                ), row=cur_row, col=1)
        fig.update_yaxes(title_text='PC score', row=cur_row, col=1)

    # ── Raw signal rows ─────────────────────────────────────────────────
    for ch in raw_signals:
        cur_row += 1

        src = cache.session.psg if ch in cache.session.psg else cache.session.cap
        sig = src[ch].astype(np.float64)
        sig = _apply_filter(sig, filter_lo, filter_hi, FS)

        if do_zscore:
            mu = np.mean(sig)
            sd = np.std(sig)
            sig = (sig - mu) / (sd + 1e-12)

        step = max(1, len(sig) // MAX_RAW_DISPLAY_PTS)
        fig.add_trace(go.Scattergl(
            x=cache.time_hr[::step],
            y=sig[::step],
            mode='lines',
            name=ch,
            line=dict(color=RAW_COLORS.get(ch, '#333'), width=1),
            legendgroup='raw',
        ), row=cur_row, col=1)
        fig.update_yaxes(title_text=ch, row=cur_row, col=1)

    # ── Layout ──────────────────────────────────────────────────────────
    s = session
    title = (f"{s.label}  {s.subject}-{s.meta['initials']}  "
             f"{s.meta['date']}  ({s.duration_hr:.1f} hr)")
    if pct > 0:
        title += f'  [motion masked {pct:.0f}%]'

    fig.update_layout(
        height=total_px,
        title=dict(text=title, font=dict(size=14)),
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.01,
                    xanchor='left', x=0, font=dict(size=10)),
        margin=dict(l=55, r=15, t=70, b=35),
        hovermode='closest',
        plot_bgcolor='#ffffff',
    )
    fig.update_xaxes(title_text='Time (hr)', row=n_rows, col=1,
                     range=[0, session.duration_hr])

    return fig


# ── Callback ─────────────────────────────────────────────────────────────────

@app.callback(
    Output('main-graph', 'figure'),
    [Input('session-dropdown', 'value'),
     Input('panel-toggles', 'value'),
     Input('spec-channels', 'value'),
     Input('band-channel', 'value'),
     Input('band-mode', 'value'),
     Input('band-set', 'value'),
     Input('spec-norm', 'value'),
     Input('feature-overlays', 'value'),
     Input('custom-bands-input', 'value'),
     Input('raw-cap', 'value'),
     Input('raw-psg', 'value'),
     Input('zscore-toggle', 'value'),
     Input('filter-lo', 'value'),
     Input('filter-hi', 'value'),
     Input('band-lp-cutoff', 'value'),
     Input('ridge-freqs-input', 'value')],
)
def update_figure(session_idx, panels, spec_channels, band_ch, band_mode,
                  band_set, spec_norm, features, custom_bands_text,
                  raw_cap, raw_psg, zscore_toggle, filter_lo, filter_hi,
                  band_lp_mhz, ridge_freqs_text):
    global _cache
    if _cache is None or _cache.idx != session_idx:
        _cache = SessionCache(session_idx)
    panels = panels or []
    spec_channels = spec_channels or []
    features = features or []
    raw_signals = (raw_cap or []) + (raw_psg or [])
    do_zscore = 'zscore' in (zscore_toggle or [])
    return _build_figure(_cache, panels, spec_channels, band_ch, band_mode,
                         band_set, spec_norm, features, custom_bands_text,
                         raw_signals, do_zscore, filter_lo, filter_hi,
                         band_lp_mhz, ridge_freqs_text)


if __name__ == '__main__':
    print('\nStarting Sleep Session Viewer at http://localhost:8050\n')
    app.run(debug=True, port=8050)

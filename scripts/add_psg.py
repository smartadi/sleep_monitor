import json

with open('analysis_raw.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# ── 1. Cell 3: add PSG_BASE_DIR, STAGE_LABELS, STAGE_COLORS, STAGE_ORDER ─────
cells[3]['source'] = r"""BASE_DIR = Path(r'C:\Users\adity\Documents\sleep monitor\overnight_6subject_pelthupdate_030526\overnight_6subject_pelthupdate_030526')
PSG_BASE_DIR = Path(r'C:\Users\adity\Documents\sleep monitor\overnight_6subject_complete_032626\overnight_6subject_complete_032626')

CAP_CHANNELS   = ['CH', 'CLE', 'CRE', 'aX', 'aY', 'aZ']
PSG_CHANNELS   = ['EEG', 'EOGl', 'EOGr', 'ECG', 'Flow', 'Pleth', 'Thorax', 'Abdomen']
ALL_SIG_COLS   = CAP_CHANNELS + PSG_CHANNELS
FS             = 100.0  # Hz (confirmed from timeMS step = 10 ms)

STAGE_LABELS = {0: 'REM', 1: 'N3', 2: 'N2', 3: 'N1', 4: 'Wake', -1: '?'}
STAGE_COLORS = {0: '#9B59B6', 1: '#2ECC71', 2: '#3498DB', 3: '#F39C12', 4: '#E74C3C', -1: '#AAAAAA'}
STAGE_ORDER  = [0, 1, 2, 3, 4]

EEG_BANDS   = {'delta': (0.5, 4.0), 'theta': (4.0, 8.0), 'alpha': (8.0, 13.0), 'beta': (13.0, 30.0)}
BAND_COLORS = {'delta': '#C0392B', 'theta': '#E67E22', 'alpha': '#27AE60', 'beta': '#2980B9'}

def _subj_dir(subj_id, initials):
    return BASE_DIR / f'{subj_id} - {initials}'

def _session_dir(subj_id, initials, date):
    return _subj_dir(subj_id, initials) / date / f'Sync_{date}'

def _csv_path(subj_id, initials, date, variant=''):
    tag = '_1point_sync' if variant == '1point' else ''
    return _session_dir(subj_id, initials, date) / f'SleepMask_PSG_100Hz{tag}_combined_{date}.csv.gz'

_S = [
    ('OS001','KJK','09-17-2024', ''), ('OS001','KJK','09-18-2024', ''),
    ('OS002','LDI','09-19-2024', ''), ('OS002','LDI','09-20-2024', ''),
    ('OS003','LCW','12-18-2025', ''), ('OS003','LCW','12-19-2025', ''),
    ('OS004','CJH','12-25-2025','1point'), ('OS004','CJH','12-26-2025','1point'),
    ('OS005','CJY','01-03-2026','1point'), ('OS005','CJY','12-27-2025','1point'),
    ('OS006','SK', '01-14-2026', ''),  ('OS006','SK', '01-15-2026', ''),
]

SESSION_META = [
    {
        'idx':      i,
        'subject':  sid,
        'initials': ini,
        'night':    (i % 2) + 1,
        'label':    f'S{(i // 2) + 1}N{(i % 2) + 1}',
        'date':     date,
        'csv':      _csv_path(sid, ini, date, var),
    }
    for i, (sid, ini, date, var) in enumerate(_S)
]

print(f'Session registry built — {len(SESSION_META)} sessions')
for m in SESSION_META:
    csv_ok = '✓' if m['csv'].exists() else '✗'
    print(f"  [{m['idx']:2d}] {m['label']}  {m['subject']}-{m['initials']}  {m['date']}  csv:{csv_ok}")
"""

# ── 2. Cell 5: load_session — also read timeSM[0] for absolute start time ─────
cells[5]['source'] = '''def load_session(idx, dtype=np.float32):
    """
    Load one session from its CSV.GZ file.

    Returns a dict with keys:
      meta       : SESSION_META[idx]
      time_ms    : (N,) float32 — milliseconds from start
      time_hr    : (N,) float32 — hours from start
      time_start : pd.Timestamp — absolute wall-clock start of recording
      cap        : dict {channel -> (N,) array}  CH, CLE, CRE, aX, aY, aZ, acc_mag
      psg        : dict {channel -> (N,) array}  EEG, EOGl, EOGr, ECG, Flow, Pleth, Thorax, Abdomen
      fs         : float  (100.0 Hz)
    """
    meta = SESSION_META[idx]

    df = pd.read_csv(meta[\'csv\'], compression=\'gzip\', dtype={
        col: np.float32 for col in ALL_SIG_COLS + [\'timeMS\']
    }, usecols=[\'timeSM\', \'timeMS\'] + ALL_SIG_COLS)

    time_start = pd.to_datetime(df[\'timeSM\'].iloc[0])

    t_ms  = df[\'timeMS\'].to_numpy(dtype=dtype)
    t_ms -= t_ms[0]
    t_hr  = t_ms / 3_600_000.0

    cap = {ch: df[ch].to_numpy(dtype=dtype) for ch in CAP_CHANNELS}
    aX, aY, aZ = cap[\'aX\'], cap[\'aY\'], cap[\'aZ\']
    cap[\'acc_mag\'] = np.sqrt(aX**2 + aY**2 + aZ**2).astype(dtype)
    psg = {ch: df[ch].to_numpy(dtype=dtype) for ch in PSG_CHANNELS}

    return {
        \'meta\':       meta,
        \'time_ms\':   t_ms,
        \'time_hr\':   t_hr,
        \'time_start\': time_start,
        \'cap\':        cap,
        \'psg\':        psg,
        \'fs\':         FS,
    }
'''

# ── 3. New cells to insert after cell 6 (load all sessions) ───────────────────
psg_section_md = {
    'cell_type': 'markdown',
    'metadata': {},
    'source': '## PSG Sleep Profiles\n\nSleep stage labels from the PSG system, stored as 30-second epoch text files in `PSG_analysis_<date>` folders. Epochs are aligned to the CSV recording window using the absolute `timeSM` clock column.'
}

psg_loader_code = {
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': '''_SP_LABEL_MAP = {
    \'wake\': 4, \'artefact\': 4, \'artifact\': 4, \'a\': -1,
    \'stage 1\': 3, \'stage 2\': 2, \'stage 3\': 1, \'rem\': 0,
}


def load_sleep_profile(session):
    """
    Parse the PSG Sleep Profile text file for one session and align it to
    the CSV recording window.

    Returns dict with:
      t_ep_hr : (N,) float — epoch start times in hours from CSV start
      labels  : (N,) str  — raw stage labels from the file
      codes   : (N,) int  — stage codes (0=REM,1=N3,2=N2,3=N1,4=Wake,-1=?)
    Returns None if no file found or no overlapping epochs.
    """
    m = session[\'meta\']
    subj_dir = PSG_BASE_DIR / f"{m[\'subject\']} - {m[\'initials\']}" / m[\'date\']
    psg_dirs = list(subj_dir.glob(\'PSG_analysis_*\'))
    if not psg_dirs:
        return None
    sp_files = [f for f in psg_dirs[0].glob(\'Sleep Profile*.txt\')
                if \'Reliability\' not in f.name]
    if not sp_files:
        return None

    lines = sp_files[0].read_text(encoding=\'utf-8\', errors=\'replace\').splitlines()

    start_dt = None
    records  = []
    for line in lines:
        line = line.strip()
        if line.startswith(\'Start Time:\'):
            start_dt = pd.to_datetime(line.split(\':\', 1)[1].strip())
        elif \',\' in line and \';\' in line:
            time_part, stage = line.split(\';\', 1)
            hh, mm, rest = time_part.strip().split(\':\')
            total_sec = int(hh) * 3600 + int(mm) * 60 + float(rest.replace(\',\', \'.\'))
            records.append((total_sec, stage.strip()))

    if start_dt is None or not records:
        return None

    base_date = start_dt.normalize()           # midnight of the recording date
    csv_start = session[\'time_start\']
    csv_end   = csv_start + pd.Timedelta(milliseconds=float(session[\'time_ms\'][-1]))

    t_ep_hr, labels = [], []
    for sec, stage in records:
        dt = base_date + pd.Timedelta(seconds=sec)
        if csv_start <= dt <= csv_end:
            t_ep_hr.append((dt - csv_start).total_seconds() / 3600.0)
            labels.append(stage)

    if not t_ep_hr:
        return None

    codes = np.array([_SP_LABEL_MAP.get(lbl.lower(), -1) for lbl in labels], dtype=int)
    return {\'t_ep_hr\': np.array(t_ep_hr), \'labels\': np.array(labels), \'codes\': codes}


def plot_hypnogram(sp, ax, title=None):
    """Draw a colored hypnogram from a sleep profile dict onto ax."""
    t  = sp[\'t_ep_hr\']
    c  = sp[\'codes\']
    for k in range(len(c) - 1):
        ax.axvspan(t[k], t[k + 1], color=STAGE_COLORS.get(c[k], \'#AAA\'), alpha=0.40)
    ax.step(t, c, color=\'k\', lw=0.8, where=\'post\')
    ax.set_yticks(sorted(STAGE_LABELS))
    ax.set_yticklabels([STAGE_LABELS[k] for k in sorted(STAGE_LABELS)], fontsize=7)
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel(\'Time (hours from recording start)\')
    if title:
        ax.set_title(title, fontsize=8)
'''
}

psg_load_all_code = {
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': '''# Load PSG sleep profiles for all sessions and align to CSV timeline
sp_profiles = []
for s in sessions:
    m = s[\'meta\']
    print(f"PSG {m[\'label\']} ({m[\'subject\']})...", end=\' \')
    sp = load_sleep_profile(s)
    sp_profiles.append(sp)
    if sp is not None:
        n = len(sp[\'t_ep_hr\'])
        print(f"{n} epochs  {sp[\'t_ep_hr\'][0]:.2f}–{sp[\'t_ep_hr\'][-1]:.2f} hr")
    else:
        print(\'not found\')
'''
}

psg_hypno_grid_code = {
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': '''# 6×2 PSG hypnogram grid — all subjects × nights
fig, axes = plt.subplots(6, 2, figsize=(16, 14))
axes[0, 0].set_title(\'Night 1\', fontsize=10, fontweight=\'bold\')
axes[0, 1].set_title(\'Night 2\', fontsize=10, fontweight=\'bold\')

for subj in range(6):
    for night in range(2):
        idx = subj * 2 + night
        s   = sessions[idx]
        sp  = sp_profiles[idx]
        ax  = axes[subj, night]
        m   = s[\'meta\']
        if sp is not None:
            plot_hypnogram(sp, ax)
        else:
            ax.text(0.5, 0.5, \'No PSG data\', ha=\'center\', transform=ax.transAxes, color=\'grey\')
        ax.set_title(f"{m[\'label\']}  {m[\'subject\']}-{m[\'initials\']}", fontsize=8)
        if night == 0:
            ax.set_ylabel(f"S{subj+1}", fontsize=8)

# Legend
from matplotlib.patches import Patch
handles = [Patch(color=STAGE_COLORS[k], label=STAGE_LABELS[k]) for k in STAGE_ORDER]
fig.legend(handles=handles, loc=\'lower center\', ncol=5, fontsize=8, frameon=True)
fig.suptitle(\'PSG Hypnograms — All Sessions\', fontsize=12)
plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.show()
'''
}

psg_single_overlay_code = {
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': '''# Single session: PSG hypnogram + all signals
s  = sessions[SESSION]
sp = sp_profiles[SESSION]
t  = s[\'time_hr\']
cap = s[\'cap\']
psg = s[\'psg\']

CH  = cap[\'CH\']  - cap[\'CH\'][0]
CLE = cap[\'CLE\'] - cap[\'CLE\'][0]
CRE = cap[\'CRE\'] - cap[\'CRE\'][0]

signal_panels = [
    (\'CH (cap)\',      t, CH,              \'#2980B9\'),
    (\'CLE (cap)\',     t, CLE,             \'#27AE60\'),
    (\'CRE (cap)\',     t, CRE,             \'#8E44AD\'),
    (\'Acc magnitude\', t, cap[\'acc_mag\'],  \'#E67E22\'),
    (\'EEG\',           t, psg[\'EEG\'],      \'#C0392B\'),
    (\'EOG left\',      t, psg[\'EOGl\'],     \'#16A085\'),
    (\'EOG right\',     t, psg[\'EOGr\'],     \'#1ABC9C\'),
    (\'ECG\',           t, psg[\'ECG\'],      \'#E74C3C\'),
    (\'Flow\',          t, psg[\'Flow\'],     \'#F39C12\'),
    (\'Pleth\',         t, psg[\'Pleth\'],    \'#9B59B6\'),
    (\'Thorax\',        t, psg[\'Thorax\'],   \'#16A085\'),
    (\'Abdomen\',       t, psg[\'Abdomen\'],  \'#2C3E50\'),
]

n_panels = 1 + len(signal_panels)   # hypnogram row on top
height_ratios = [1.2] + [1] * len(signal_panels)

fig, axes = plt.subplots(n_panels, 1, figsize=(14, 24), sharex=True,
                         gridspec_kw={\'height_ratios\': height_ratios})

# Top panel: PSG hypnogram
ax_hyp = axes[0]
if sp is not None:
    plot_hypnogram(sp, ax_hyp)
    ax_hyp.set_xlabel(\'\')
else:
    ax_hyp.text(0.5, 0.5, \'No PSG data\', ha=\'center\', transform=ax_hyp.transAxes, color=\'grey\')
ax_hyp.set_title(\'PSG Hypnogram\', fontsize=8)

# Signal panels
for ax, (label, t_, sig, color) in zip(axes[1:], signal_panels):
    ax.plot(t_, sig, color=color, lw=0.4, alpha=0.9)
    ax.set_ylabel(label, fontsize=8)

axes[-1].set_xlabel(\'Time (hours from recording start)\')
m = s[\'meta\']
fig.suptitle(f"{m[\'label\']}  {m[\'subject\']}-{m[\'initials\']}  {m[\'date\']} — Full Night with PSG", fontsize=11)
plt.tight_layout()
plt.show()
'''
}

# Insert new cells after cell 6 (load all sessions), in order
insert_at = 7
for new_cell in [psg_section_md, psg_loader_code, psg_load_all_code,
                 psg_hypno_grid_code, psg_single_overlay_code]:
    nb['cells'].insert(insert_at, new_cell)
    insert_at += 1

with open('analysis_raw.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print('Done. Cell listing:')
with open('analysis_raw.ipynb', 'r', encoding='utf-8') as f:
    nb2 = json.load(f)
for i, cell in enumerate(nb2['cells']):
    src = ''.join(cell['source'])[:70].replace('\n', ' ')
    print(f'  Cell {i:2d} [{cell["cell_type"]}]: {src}')

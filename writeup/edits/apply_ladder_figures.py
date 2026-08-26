"""Refresh the manuscript's stage figures onto the corrected ladder order.

Eight of the manuscript's live images carry a sleep-stage axis. The repo
convention is defined once in `sleep_monitor.config` -- top to bottom, Wake, N1,
N2, N3, REM, the axis read as depth with REM at the deepest rung -- and every one
of these figures has been regenerated on it.

Nothing but the order moves. Each regenerated figure was checked against the
version it replaces: same estimator, same checkpoint, same statistics, same
printed p-values. `band_ridge_by_stage` and `fig3_per_stage_mae` were compared
panel by panel and cell by cell.

The swap is done at the media level -- the bytes of each existing image part are
replaced in place, keeping its name, its relationship id and its display extent.
No document XML is touched, so figure numbering, captions and layout cannot move.
That makes the aspect ratio load-bearing: the stored extent would stretch a
replacement of a different shape. Where one differs slightly (the per-stage MAE
figure sits in the document a few percent shorter than matplotlib draws it, from
an earlier crop), the replacement is letterboxed with white to the original
shape rather than distorted or forced through an XML edit.

Run from the repo root:  .venv/Scripts/python.exe writeup/edits/apply_ladder_figures.py
"""

import io
import shutil
import sys
import zipfile
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from target_doc import DOC  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIGS = ROOT / "writeup" / "figures"
BACKUP = ROOT / "writeup" / "_archive" / "CAP_sleep_mask_manuscript_main_PRE_LADDER_20260825.docx"

# media part in the docx -> the figure on disk that it came from
SWAPS = {
    "word/media/S1N1_CH_CLE-CRE_39.png": FIGS / "channel_evolution" / "S1N1_CH_CLE-CRE.png",
    "word/media/ladder_S6N1_33.png": FIGS / "harmonics" / "ladders" / "ladder_S6N1.png",
    "word/media/ladder_stage_relationship_34.png": FIGS / "harmonics" / "ladder_stage_relationship.png",
    "word/media/ridge_tune_S1N1_CRE_36.png": FIGS / "harmonics" / "ridges" / "ridge_tune_S1N1_CRE.png",
    "word/media/image7.png": FIGS / "harmonics" / "band_ridge_by_stage.png",
    "word/media/image11.png": FIGS / "mask_rate_detection" / "fig3_per_stage_mae.png",
    "word/media/image15.png": FIGS / "channel_evolution" / "ch_vs_clecre_grid.png",
    "word/media/image16.png": FIGS / "imbalance" / "imbalance_grid.png",
}

# The stored extent is in EMU and is not recomputed here, so a replacement whose
# shape differs would be squashed into the old box. Small differences are
# letterboxed to match; anything larger is a figure that genuinely changed shape
# and needs a look, not a silent pad.
ASPECT_TOL = 0.002
ASPECT_PAD_LIMIT = 0.10


def fit_aspect(png, target):
    """Return PNG bytes letterboxed with white to the target width/height ratio."""
    im = Image.open(png).convert("RGB")
    a = im.width / im.height
    if abs(a - target) / target <= ASPECT_TOL:
        return Path(png).read_bytes()
    if a > target:                      # too wide -> add height
        h = int(round(im.width / target)); w = im.width
    else:                               # too tall -> add width
        w = int(round(im.height * target)); h = im.height
    canvas = Image.new("RGB", (w, h), "white")
    canvas.paste(im, ((w - im.width) // 2, (h - im.height) // 2))
    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return buf.getvalue()


def main():
    missing = [str(p) for p in SWAPS.values() if not p.exists()]
    if missing:
        sys.exit("figure not on disk:\n  " + "\n  ".join(missing))

    src = zipfile.ZipFile(DOC)
    names = src.namelist()
    for part in SWAPS:
        if part not in names:
            sys.exit("no such image part in the document: %s" % part)

    # shape check and letterboxing, before anything is written
    payload = {}
    for part, png in SWAPS.items():
        old = Image.open(io.BytesIO(src.read(part)))
        new = Image.open(png)
        a_old, a_new = old.width / old.height, new.width / new.height
        drift = abs(a_old - a_new) / a_old
        if drift > ASPECT_PAD_LIMIT:
            sys.exit("aspect ratio moved too far for %s: %.3f -> %.3f (%s)"
                     % (part, a_old, a_new, png.name))
        payload[part] = fit_aspect(png, a_old)
        if drift > ASPECT_TOL:
            print("  letterboxed %s (%.3f -> %.3f)" % (png.name, a_new, a_old))

    BACKUP.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DOC, BACKUP)
    print("backup -> %s" % BACKUP)

    tmp = DOC.with_suffix(".ladder.tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as out:
        for item in src.infolist():
            if item.filename in SWAPS:
                png = SWAPS[item.filename]
                out.writestr(item, payload[item.filename])
                print("  %-42s <- %s" % (item.filename.replace("word/media/", ""),
                                         png.relative_to(FIGS)))
            else:
                out.writestr(item, src.read(item.filename))
    src.close()
    tmp.replace(DOC)
    print("\nwrote %s  (%d images refreshed)" % (DOC, len(SWAPS)))


if __name__ == "__main__":
    main()

# Writeup Directory

Single manuscript covering both rate detection and slow-wave spectral analysis from capacitive temple sensors.

## Structure

```
writeup/
├── paper/                       The manuscript — all tracking files live here
│   ├── DRAFT.md                 Source of truth for prose (authored here)
│   ├── build_docx.py            DRAFT.md → the single deliverable docx
│   ├── CAP_sleep_mask_paper.docx  ← THE deliverable (prof drops this into the full paper)
│   ├── OUTLINE.md               Section structure, status, standing rules
│   ├── CLAIMS.md                Each claim → evidence (data, script, figure)
│   ├── FIGURES.md               Figure list with source paths + status
│   ├── TABLES.md                Table list with data sources + status
│   ├── KEY_NUMBERS.md           All quantitative results, grouped by section
│   ├── figures/                 Paper-specific figure copies
│   └── data/                    Extracted CSVs feeding tables
│
├── shared/                      Shared methods text (participants, preprocessing, GT)
├── figures/                     Master figure archive (all topics, existing)
├── _legacy/                     Archived prior drafts (do not edit; reference only)
├── harmonics/                   Legacy harmonics docx (pending archive into _legacy/)
└── SFN 2026 *.docx              Conference abstracts (separate deliverable, kept)
```

## The deliverable

There is **one** paper docx: `paper/CAP_sleep_mask_paper.docx`. Do not hand-edit it —
edit `paper/DRAFT.md` and rebuild:

```
cd writeup/paper
py build_docx.py        # regenerates CAP_sleep_mask_paper.docx from DRAFT.md
```

DRAFT.md is plain markdown; HTML comment blocks in it are internal notes and are not
rendered. The professor takes the generated docx and integrates it into the full paper.

## File Roles

| File | Purpose | When to update |
|------|---------|----------------|
| `JOURNAL.md` | **Your input** — natural language notes and instructions | Anytime; Claude reads this as work orders |
| `OUTLINE.md` | Section list + status tracker | When section status changes |
| `CLAIMS.md` | Claim → evidence chain | When a claim is added, verified, or killed |
| `FIGURES.md` | Figure # → source file | When figures are created or regenerated |
| `TABLES.md` | Table # → data source | When tables are updated |
| `KEY_NUMBERS.md` | All stats by section | When analysis is re-run |
| `DRAFT.md` | Prose | During writing |

## Workflow

### Writing a section
1. Check `OUTLINE.md` for scope and status
2. Pull claims from `CLAIMS.md` — every paragraph traces to a claim
3. Pull statistics from `KEY_NUMBERS.md`
4. Reference figures/tables from `FIGURES.md` / `TABLES.md`
5. Write in `DRAFT.md`
6. Update `OUTLINE.md` status

### Adding a new result
1. Add claim to `CLAIMS.md` with evidence chain
2. Add numbers to `KEY_NUMBERS.md`
3. Add figure/table to `FIGURES.md` / `TABLES.md`
4. Update `OUTLINE.md` if section scope changes

### Talking to Claude
- **Write in `JOURNAL.md`** — add dated freeform notes, thoughts, or instructions. Then tell Claude "check the journal" and it will read your entries and act on them.
- **"Write section 3.2"** → reads OUTLINE + CLAIMS + KEY_NUMBERS, writes prose
- **"Review CLAIMS.md for gaps"** → audits completeness
- **"What's the paper status?"** → scans OUTLINE.md, reports
- **"Generate Fig 7"** → finds script + data from FIGURES.md, runs it
- **"Update KEY_NUMBERS after re-run"** → extracts new stats

## Legacy Files (reference only, do not modify)
- `PAPER_TASK.md` — original monolithic task spec
- `CAP_sleep_analysis_paper.docx` — early combined draft
- `CAP_rate_consolidation_section.docx` — rate consolidation draft
- `SFN 2026 *.docx` — conference abstract versions
- `harmonics/CAP_harmonic_ridge_analysis.docx` — harmonics draft

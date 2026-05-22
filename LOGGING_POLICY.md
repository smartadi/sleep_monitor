# Logging Policy

Every code or analysis change must be logged. This document defines what goes where.

## Log files

| Log | Location | What goes in it |
|-----|----------|-----------------|
| **Analysis Log** | `notebooks/ANALYSIS_LOG.md` | Every analysis run: question, script/notebook, parameters, results, plots, findings. Also records new scripts/notebooks with purpose + outputs. |
| **Code Changelog** | `CHANGELOG.md` | Changes to library modules (`sleep_monitor/*.py`), new scripts, new notebooks, config changes. One entry per change with date, file(s), and what changed. |

`PROGRESS_LOG.md` is deprecated — fold its content into the Analysis Log or Changelog as appropriate.

## When to log

### Analysis Log (`notebooks/ANALYSIS_LOG.md`)

Log an entry when:
- You run an analysis and get results (even negative results)
- You create a new analysis script or notebook
- You change methodology (e.g., switching GT source, changing a default estimator)
- You validate or invalidate a hypothesis

Entry template:
```markdown
## YYYY-MM-DD — Short title

**Question:** What are we trying to learn?

**Script/Notebook:** `path/to/file.py`
**Outputs:** `artifacts/...`, `notebooks/plots/.../...`

### Setup
Parameters, channels, sessions, GT source.

### Results
Tables, metrics, key numbers.

### Key findings
Numbered list of observations.

### Status
Done / in progress / superseded by X.
```

### Code Changelog (`CHANGELOG.md`)

Log an entry when:
- You add or modify a module in `sleep_monitor/`
- You add a new script to `scripts/`
- You add a new notebook to `notebooks/`
- You change project configuration (requirements, setup.py, config.py constants)
- You archive or delete files

Entry template:
```markdown
## YYYY-MM-DD

- **Added** `scripts/new_script.py` — one-line description
- **Changed** `sleep_monitor/rates.py` — added `rate_foo()`, changed default k from 1.67 to 1.70
- **Removed** `scripts/old_script.py` — superseded by `new_script.py`, moved to `archive/`
```

## Rules

1. **Log before moving on.** After completing any code change or analysis run, update the appropriate log before starting the next task.
2. **Negative results count.** If an experiment showed nothing useful, log that — it prevents re-running the same dead end.
3. **Link scripts to logs.** Every script in `scripts/` and every `.py`/`.ipynb` in `notebooks/` should have a corresponding log entry.
4. **Record run status.** Note whether a script has actually been executed and produced outputs, or just been written.
5. **Date everything.** Use ISO dates (YYYY-MM-DD). Convert relative dates ("next Thursday") to absolute.
6. **Keep Next Steps current.** The "Next Steps" section at the bottom of ANALYSIS_LOG.md should reflect actual priorities, not stale ideas.

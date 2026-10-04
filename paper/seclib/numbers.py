"""
The numbers registry: every number the paper quotes, recomputed and compared.

Each stage creates one `Numbers` object and calls `add` for every value the
manuscript states. `paper_value` is the text exactly as it appears in the
manuscript being checked (V11), so a reader can find it; `computed` is what the
code gets now. `run_all.py` gathers every stage's file into
outputs/paper_numbers.csv and outputs/PAPER_NUMBERS.md.

Status
    MATCH     computed rounds to the paper's value at the paper's precision
    DIFF      it does not -- the text needs changing (or the code is wrong)
    NEW       computed, but the manuscript does not state it (yet)
    EXTERNAL  the paper's value comes from outside this code (co-author work)

The comparison is deliberately dumb: the paper value's own number of decimals
sets the tolerance (half a unit in its last place), so "0.55" matches 0.546 but
not 0.542. Anything subtler goes in `note`.
"""

from __future__ import annotations

import math
import re
from typing import Optional

import pandas as pd

from .config import NUM_DIR

# A dash is a minus sign only when it does not follow a digit: "10–50" is a range
# (10 and 50), "−0.22" and "( -3" are negative numbers.
_NUM = re.compile(r'(?<![\d.])[-−–]?\d+(?:\.\d+)?|\d+(?:\.\d+)?')


def _decimals(text: str) -> int:
    m = _NUM.search(text)
    if not m:
        return 0
    s = m.group(0)
    return len(s.split('.')[1]) if '.' in s else 0


def _first_number(text: str) -> Optional[float]:
    m = _NUM.search(text)
    if not m:
        return None
    return float(m.group(0).replace('−', '-').replace('–', '-'))


class Numbers:
    COLUMNS = ['stage', 'id', 'section', 'what', 'paper_value', 'computed',
               'unit', 'status', 'source', 'note']

    def __init__(self, stage: str):
        self.stage = stage
        self.rows: list[dict] = []

    def add(self, id: str, section: str, what: str, computed,
            paper_value: Optional[str] = None, unit: str = '', source: str = '',
            note: str = '', status: Optional[str] = None) -> None:
        """Record one number.

        computed     a float (compared) or a string such as '0.42–0.56' (ranges:
                     every number in it is compared, in order, with the paper's)
        paper_value  the manuscript's text for it, or None if the paper does not
                     state it
        status       force a status ('EXTERNAL'); otherwise it is worked out
        """
        if status is None:
            status = self._compare(computed, paper_value)
        self.rows.append(dict(stage=self.stage, id=id, section=section, what=what,
                              paper_value='' if paper_value is None else paper_value,
                              computed=self._fmt(computed, paper_value), unit=unit,
                              status=status, source=source, note=note))

    @staticmethod
    def _fmt(v, paper_value):
        if isinstance(v, str):
            return v
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return 'nan'
        d = _decimals(paper_value or '')
        return f'{v:.{max(d, 3)}f}' if isinstance(v, float) else str(v)

    @staticmethod
    def _compare(computed, paper_value) -> str:
        if paper_value is None or paper_value == '':
            return 'NEW'
        pv = [float(x.replace('−', '-').replace('–', '-'))
              for x in _NUM.findall(paper_value)]
        if isinstance(computed, str):
            cv = [float(x.replace('−', '-').replace('–', '-'))
                  for x in _NUM.findall(computed)]
        else:
            cv = [float(computed)]
        if not pv or not cv or len(pv) != len(cv):
            return 'DIFF'
        d = _decimals(paper_value)
        tol = 0.5 * 10 ** (-d) + 1e-9
        return 'MATCH' if all(abs(a - b) <= tol for a, b in zip(pv, cv)) else 'DIFF'

    def save(self) -> pd.DataFrame:
        NUM_DIR.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self.rows, columns=self.COLUMNS)
        df.to_csv(NUM_DIR / f'{self.stage}.csv', index=False)
        counts = df.status.value_counts().to_dict()
        print(f'  [{self.stage}] {len(df)} numbers: '
              + ', '.join(f'{k} {v}' for k, v in sorted(counts.items())))
        return df

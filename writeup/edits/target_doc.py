"""Which manuscript file the edit scripts write to.

There is one manuscript: `writeup/main/CAP_sleep_mask_manuscript_main.docx`.
The working copy used through 2026-08-17 was promoted into it and deleted, so
edits land on the canonical file directly again.

Override with the MS_DOC environment variable to target a copy — useful when
the canonical file is open in Word, which locks it against writing:

    MS_DOC=writeup/main/scratch.docx python writeup/edits/<script>.py
"""

import os
from pathlib import Path

CANONICAL = Path("writeup/main/CAP_sleep_mask_manuscript_main.docx")

DOC = Path(os.environ.get("MS_DOC", CANONICAL))

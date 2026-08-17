"""Which manuscript file the edit scripts write to.

The user keeps the canonical file open in Word to read, so edits land on a
working copy and are promoted afterwards. Override with the MS_DOC environment
variable to target a different file.

Promote the working copy when the canonical file is closed:

    cp writeup/main/CAP_sleep_mask_manuscript_main_WORKING.docx \
       writeup/main/CAP_sleep_mask_manuscript_main.docx
"""

import os
from pathlib import Path

CANONICAL = Path("writeup/main/CAP_sleep_mask_manuscript_main.docx")
WORKING = Path("writeup/main/CAP_sleep_mask_manuscript_main_WORKING.docx")

DOC = Path(os.environ.get("MS_DOC", WORKING))

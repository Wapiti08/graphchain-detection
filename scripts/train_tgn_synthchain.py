from __future__ import annotations

from pathlib import Path
import sys

# Ensure local package imports work when invoked as a script.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from graph.tgn_train.runner import cli_main


if __name__ == "__main__":
    cli_main()


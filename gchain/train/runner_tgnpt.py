"""Backward compatibility: delegates to the unified runner."""
from __future__ import annotations

from typing import List, Optional

from gchain.train.runner import cli_main

__all__ = ["cli_main"]

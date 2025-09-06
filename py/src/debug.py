from __future__ import annotations
import os
from typing import Callable


def make_debug_logger(name: str) -> Callable[[str], None]:
	"""Return a simple debug logging function gated by DEBUG.

	When DEBUG=1 the returned function prints messages prefixed with
	the component name. Otherwise it is a noop.
	"""
	enabled = os.environ.get('DEBUG') == '1'
	if not enabled:
		def _noop(msg: str) -> None:
			return
		return _noop

	def _dbg(msg: str) -> None:
		print(f"[debug::{name}] {msg}")

	return _dbg

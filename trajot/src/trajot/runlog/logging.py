"""``log.txt``: tee stdout and stderr to the run directory, leaving them on the console."""

from __future__ import annotations

import sys
from pathlib import Path


class _Tee:
    def __init__(self, stream, sink) -> None:
        self._stream, self._sink = stream, sink

    def write(self, text: str) -> int:
        self._sink.write(text)
        self._sink.flush()
        return self._stream.write(text)

    def flush(self) -> None:
        self._stream.flush()

    def __getattr__(self, name):  # isatty, encoding, fileno, ...
        return getattr(self._stream, name)


class RunLogger:
    """Tee ``sys.stdout`` and ``sys.stderr`` to ``<run_dir>/log.txt`` from construction until ``close()``.

    Only Python-level writes are captured; output written straight to file descriptor 1 or 2
    (subprocesses, joblib workers) is not.
    """

    def __init__(self, run_dir: Path, name: str) -> None:
        self.name = name
        self._file = (Path(run_dir) / "log.txt").open("w", encoding="utf-8")
        self._stdout, self._stderr = sys.stdout, sys.stderr
        sys.stdout = self._tee_out = _Tee(self._stdout, self._file)
        sys.stderr = self._tee_err = _Tee(self._stderr, self._file)

    def log(self, msg: str) -> None:
        print(f"[{self.name}] {msg}")

    def close(self) -> None:
        if self._file.closed:
            return
        if sys.stdout is self._tee_out:
            sys.stdout = self._stdout
        if sys.stderr is self._tee_err:
            sys.stderr = self._stderr
        self._file.close()

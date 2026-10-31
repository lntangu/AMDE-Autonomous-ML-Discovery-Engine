#!/usr/bin/env python3
"""Run a worker and relay output into a size-bounded rotating log."""
from __future__ import annotations
import argparse
import logging
import os
import subprocess
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-path", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("a worker command is required after --")
    path = Path(args.log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("amde2.worker.relay")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = RotatingFileHandler(path, maxBytes=20 * 1024 * 1024, backupCount=5, encoding="utf-8", delay=False)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    logger.info("AMDE 2.0 worker log relay started")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    try:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, cwd=os.getcwd(), env=env,
                                 text=True, encoding="utf-8", errors="replace", bufsize=1)
        assert child.stdout is not None
        for line in child.stdout:
            logger.info(line.rstrip("\r\n"))
        code = child.wait()
        logger.info("AMDE 2.0 worker exited: exit_code=%s", code)
        return int(code)
    except OSError as exc:
        logger.exception("Could not launch worker: %s", exc)
        return 127
    finally:
        for item in list(logger.handlers):
            item.flush()
            item.close()
            logger.removeHandler(item)


if __name__ == "__main__":
    raise SystemExit(main())

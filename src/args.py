"""Argument printing and logging for training runs.

Run configuration is loaded with Hydra (see ``conf/`` and ``train.py`` / ``test.py``).
"""

from __future__ import annotations

from pathlib import Path


def _format_val(val):
    if isinstance(val, Path):
        return str(val)
    return val


def print_args(args):
    print("")
    for arg in sorted(vars(args)):
        val = getattr(args, arg)
        if val is not None:
            print("{0:<21} {1:<}".format(arg, _format_val(val)))
        else:
            print("{0:<21} None".format(arg))
    print("")


def log_args(args, file_path):
    with open(file_path, "w") as outfile:
        for arg in sorted(vars(args)):
            val = getattr(args, arg)
            if val is not None:
                outfile.write("{0:<21} {1:<}\n".format(arg, _format_val(val)))
            else:
                outfile.write("{0:<21} None\n".format(arg))

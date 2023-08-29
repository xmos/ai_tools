import subprocess
import typing
from pathlib import Path
from typing import Union, List, Optional
from .flash import generate_flash
import re

__compilation_output = ""
__arena_size = 0


def convert(
    filename: Union[str, Path],
    outfile: Union[str, Path],
    params: Optional[typing.Dict[str, Optional[str]]],
) -> int:
    args: List[str] = ["xcore-opt", "-o", str(outfile)]

    if params is not None:
        for key, val in params.items():
            if len(key) > 1:
                flag: str = "--" + str(key)
            else:
                flag = "-" + str(key)
            if str(val) == "" or val is None:
                args.append(flag)
            else:
                args.append(f"{flag}={val}")

    args.append(str(filename))

    process_call: subprocess.CompletedProcess = subprocess.run(
        [arg for arg in args],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )

    global __compilation_output, __arena_size
    __compilation_output = process_call.stdout.decode("utf-8")
    size_str = re.sub("((.|\n|\r)*)Tensor arena size :", "", __compilation_output)
    size_str = re.sub("(\n|\r)((.|\n|\r)*)", "", size_str)
    __arena_size = int(size_str.strip())

    return process_call.returncode


def tensor_arena_size() -> int:
    return __arena_size


def print_optimization_report():
    print(__compilation_output)


def print_help(show_hidden: Optional[bool] = False) -> int:
    if show_hidden:
        return subprocess.run(["xcore-opt", "--help-list-hidden"]).returncode

    return subprocess.run(["xcore-opt", "--help-list"]).returncode

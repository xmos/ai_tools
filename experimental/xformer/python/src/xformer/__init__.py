import subprocess
import typing
from pathlib import Path
from typing import Union, List, Optional


def convert(
    filename: Union[str, Path],
    outfile: Union[str, Path],
    params: Optional[typing.Dict[str, Optional[str]]],
) -> int:
    args: List[Optional[str]] = ["xcore-opt", "-o", str(outfile)]

    if params is not None:
        for key, val in params.items():
            if len(key) > 1:
                flag: str = "--" + str(key)
            else:
                flag = "-" + str(key)
            if str(val) == "" or val is None:
                args.append(flag)
            else:
                args.append(f"{flag} {val}")

    args.append(str(filename))

    process_call: subprocess.CompletedProcess = subprocess.run(
        [arg for arg in args], check=True
    )
    return process_call.returncode


def print_help(show_hidden: Optional[bool] = False) -> int:
    if show_hidden:
        return subprocess.run(["xcore-opt", "--help-list-hidden"]).returncode

    return subprocess.run(["xcore-opt", "--help-list"]).returncode

import os
import subprocess
import typing
from pathlib import Path
from platform import uname
from typing import Union, List, Optional


def __find_binary() -> Union[str, None]:
    package_dir: str = os.path.dirname(os.path.realpath(__file__))
    suffix: str = "xcore-opt"
    system, _, release, _, _, _ = uname()
    os_dir: Path = package_dir / Path("binaries/" + system)
    if os_dir.is_dir():
        files = os_dir.rglob("*-" + suffix + "*")

        for file in files:
            filename: str = file.stem
            offset: int = filename.find(suffix) - 1
            binary_version: str = file.stem[:offset].replace("-", ".")

            if binary_version == release:
                return str(file)
            else:
                # TODO: check for closest release?
                raise Exception("Not found")
        raise Exception("Not found")
    else:
        # TODO: is an exception appropriate here?
        raise OSError(f"This operating system is unsupported: {system} {release}")


def run(
        filename: Union[str, Path],
        outfile: Union[str, Path],
        params: Optional[typing.Dict[str, Optional[str]]]
) -> int:
    args: List[Optional[str]] = [__find_binary(), "-o", str(outfile)]

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

    process_call: subprocess.CompletedProcess = subprocess.run([arg for arg in args], check=True)
    return process_call.returncode


if __name__ == "__main__":
    run(Path("../example/testmodel.tflite").resolve(), Path("../example/output.tflite").resolve(), {"mlir-disable-threading": None})